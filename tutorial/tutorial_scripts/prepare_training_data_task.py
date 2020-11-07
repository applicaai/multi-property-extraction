
import enum
import gzip
import io
import logging
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast

import luigi
import regex

import gzip
import logging
import lzma
import os
from pathlib import Path

import luigi
from luigi.target import Target

from .download_dataset_task import DownloadDatasetTask
from .download_model_task import DownloadModelTask, ModelType

logger = logging.getLogger('luigi-interface')


class PrepareTrainingDataTask(luigi.Task):
    """Preprocess challenge data (split)."""

    output_dir = luigi.Parameter('./processed')
    subset = luigi.Parameter()
    model = luigi.EnumParameter(enum=ModelType, description='Model type')

    def requires(self) -> Dict[str, luigi.Task]:
        requirements = {
            'model': DownloadModelTask(model=self.model),
            'dataset': DownloadDatasetTask(),
        }
        return requirements

    def output(self) -> Dict[str, luigi.Target]:
        out = {
            'in': luigi.LocalTarget(Path(self.output_dir).absolute() / self.model.name / f'{self.subset}.in-out.in'),
            'out': luigi.LocalTarget(Path(self.output_dir).absolute() / self.model.name / f'{self.subset}.in-out.out'),
        }
        if self.model == ModelType.T5:
            out['property_names'] = luigi.LocalTarget(Path(self.output_dir).absolute() / self.model.name / f'{self.subset}.property_names')
            out['indices'] = luigi.LocalTarget(Path(self.output_dir).absolute() / self.model.name / f'{self.subset}.indices')
        return out

    def parse_property_names(self, prop_names_line: str) -> List[str]:
        return [item.replace('_', ' ').replace('  ', ' ') for item in prop_names_line.strip().split(' ')]

    def smart_open(self, posix_path: str):
        path = str(posix_path)
        if os.path.exists(path + '.gz'):
            logger.debug(f'Smart-opening {posix_path}.gz')
            return gzip.open(path + '.gz', 'rt', encoding='utf-8')
        elif os.path.exists(str(path) + '.xz'):
            logger.debug(f'Smart-opening {posix_path}.xz')
            return lzma.open(path + '.xz', 'rt', encoding='utf-8')
        logger.debug(f'Smart-opening {posix_path}')
        return open(path, 'rt', encoding='utf-8')

    @property
    def property_column(self) -> int:
        return 0

    @property
    def doc_column(self) -> int:
        return 1

    def run(self):
        os.makedirs(Path(self.output_dir).absolute() / self.model.name, exist_ok=True)

        in_files = [
            self.smart_open(Path(self.input()['dataset'].path) / self.subset / 'in.tsv'),
            self.smart_open(Path(self.input()['dataset'].path) / self.subset / 'expected.tsv'),
        ]

        processed_data_files = {key: open(self.output()[key].path, 'w') for key in self.output()}

        processor = self.requires()['model'].get_processor()

        for line_idx, (input_line, reference_line) in enumerate(zip(*in_files)):
            items = input_line.rstrip('\n').split('\t')
            properties = [property.replace('_', ' ') for property in items[self.property_column].split(' ')]
            document = items[self.doc_column].replace(' <EOL> ', ' ').replace(' <EOS> ', ' ').replace(' <EOP> ', ' ')
            document = regex.sub(r'\s+', ' ', document)
            reference_items = [item.replace('_', ' ').replace('  ', ' ').split('=', maxsplit=1)
                               for item in reference_line.strip().split(' ')]
            pieced_document = ' '.join(processor.encode_as_pieces(document))
            if self.model in (ModelType.DUAL_ROBERTA_TRANSFORMER, ModelType.DUAL_SOURCE_TRANSFORMER):
                properties.sort()
                pieced_properties = ' '.join(processor.encode_as_pieces(' ### '.join(properties)))
                processed_data_files['in'].write(pieced_document + '\n')
                processed_data_files['in'].write(pieced_properties + '\n')

                pieced_property_values = processor.encode_as_pieces(
                    ' ### '.join([f'{item[0]} : {item[1]}' for item in reference_items]))
                processed_data_files['out'].write(' '.join(pieced_property_values) + '\n')
            else:
                for prop in properties:
                    reference = filter(lambda x: x[0] == prop, reference_items)
                    pieced_prop = ' '.join(processor.encode_as_pieces(prop))
                    processed_data_files['in'].write(f'{pieced_prop} : {pieced_document}\n')

                    pieced_property_values = processor.encode_as_pieces(' ### '.join(entity[1] for entity in reference))
                    processed_data_files['out'].write(' '.join(pieced_property_values) + '\n')
                    processed_data_files['indices'].write(f'{line_idx}\n')
                    processed_data_files['property_names'].write(f'{prop}\n')

        for in_file in in_files:
            in_file.close()

        for pd_file in processed_data_files.values():
            pd_file.close()


import gzip
import lzma
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from .generate_predictions_task import GeneratePredictionsTask
from .download_model_task import DownloadModelTask, ModelType
from typing import DefaultDict, Dict, List, Optional, Tuple

import luigi
import regex
from tqdm import tqdm


Prediction = namedtuple('Prediction', ['value', 'score', 'position'])


class PostprocessPredictionsTask(luigi.Task):
    """Postprocess prediction from fairseq."""

    subset = luigi.Parameter()
    model = luigi.EnumParameter(enum=ModelType, description='Model type')
    result_path = luigi.Parameter('./outputs')

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(Path(self.result_path) / self.model.name / self.subset / 'out.tsv')

    def requires(self) -> Dict[str, luigi.Task]:
        requirements = {
            'model': DownloadModelTask(model=self.model),
            'predictions': GeneratePredictionsTask(model=self.model, subset=self.subset),
        }
        return requirements

    @property
    def processor(self):
        if not hasattr(self, '__processor'):
            self.__processor = self.requires()['model'].get_processor()
        return self.__processor

    def extract_predictions(self, line: str, property_names: Optional[List[str]] = None) -> Tuple[int, List[str]]:
        items = line.strip().split('\t')

        line_idx = int(items[0][2:])
        tokens = items[2].strip().split(' ')

        decoded_text = self.processor.decode_pieces(tokens)

        predictions = []
        for prediction in decoded_text.split('###'):
            if property_names:
                property_name = property_names[line_idx]
                prediction = f'{property_name}_:_{prediction.strip()}'

            if len(prediction.strip().replace(' ', '_').split('_:_')) == 2:
                name, value = prediction.strip().replace(' ', '_').split('_:_', maxsplit=1)
                predictions.append(f'{name}={value}')
            else:
                if prediction:
                    predictions.append(prediction.strip().replace(' ', '_'))
        return line_idx, predictions

    def aggregate_by_article(self, predictions: Dict[int, List[str]]) -> Dict[int, List[str]]:
        preprocess_task = self.requires()['predictions'].requires()['data'].requires()['prepare-data']
        with open(preprocess_task.output()['indices'].path) as index_file:
            indices = [int(line.strip()) for line in index_file]

        aggregated_predictions: DefaultDict[int, List[str]] = defaultdict(list)
        for line_idx, doc_idx in enumerate(indices):
            aggregated_predictions[doc_idx].extend(predictions[line_idx])
        return aggregated_predictions

    def remove_duplicates(self, predictions: DefaultDict[int, List[str]]):
        for idx in predictions:
            predictions[idx] = list(set(predictions[idx]))

    def read_property_names(self) -> List[str]:
        preprocess_task = self.requires()['predictions'].requires()['data'].requires()['prepare-data']

        with open(preprocess_task.output()['property_names'].path) as name_file:
            names = [line.strip() for line in name_file]
        return names

    def run(self):
        property_names = self.read_property_names() if self.model == ModelType.T5 else None

        final_predictions = defaultdict(list)
        with open(self.input()['predictions'].path) as generated_file:
            for line in generated_file:
                if line.startswith('H-'):
                    line_idx, predictions = self.extract_predictions(line, property_names)
                    final_predictions[line_idx] = predictions

        if self.model == ModelType.T5:
            final_predictions = self.aggregate_by_article(final_predictions)

        self.remove_duplicates(final_predictions)

        # if self.filter_extra_properties:
        property_names = []
        with gzip.open(Path('dataset') / 'wikireading-recycled' / self.subset / 'in.tsv.gz', 'rt') as source_file:
            for line in source_file:
                property_names.append(line.strip().split('\t')[0].split(' '))

        for doc_idx in range(max(final_predictions.keys()) + 1):
            final_predictions[doc_idx] = [prediction for prediction in final_predictions[doc_idx]
                                            if prediction.split('=')[0] in property_names[doc_idx]]

        with open(self.output().path, 'wt') as out_file:
            for i in range(max(final_predictions.keys()) + 1):
                out_file.write(' '.join(sorted(final_predictions[i])) + '\n')

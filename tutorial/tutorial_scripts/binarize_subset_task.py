import os
import shutil
from typing import List

import luigi
import luigi.contrib.external_program
from fairseq import options
from fairseq_cli import preprocess

from . import DownloadModelTask, PrepareTrainingDataTask
from .download_model_task import ModelType


class BinarizeSubsetTask(luigi.contrib.external_program.ExternalProgramTask):
    """Binarize data (split basically) for fairseq."""

    subset = luigi.Parameter()
    model = luigi.EnumParameter(enum=ModelType, description='Model type')

    def requires(self):
        requirements = {
            'prepare-data': PrepareTrainingDataTask(subset=self.subset, model=self.model),
            'model': DownloadModelTask(model=self.model),
        }
        return requirements

    def output(self):
        output = {}
        for suffix in ('in', 'out'):
            if self.subset == 'train':
                out_subset = 'train'
            elif self.subset.startswith('dev-'):
                out_subset = 'valid'
            elif self.subset.startswith('test-'):
                out_subset = 'test'
            output[f'{suffix}.bin'] = luigi.LocalTarget(f'binarized/{self.model.name}/{self.subset}/{out_subset}.in-out.{suffix}.bin')
            output[f'{suffix}.idx'] = luigi.LocalTarget(f'binarized/{self.model.name}/{self.subset}/{out_subset}.in-out.{suffix}.idx')
        return output

    def program_args(self) -> List[str]:
        sp_vocab_path = self.input()['model']['dict'].path
        binarization_options = [
            'fairseq-preprocess',
            '--destdir', f'./binarized/{self.model.name}/{self.subset}',
            '--joined-dictionary',
            '--dataset-impl', 'mmap',
            '--workers', '60',
            '--seed', '31337',
            '--srcdict', sp_vocab_path,
            '-s', 'in',
            '-t', 'out',
            '--user-dir', './fairseq_modules',
        ]
        if self.model == ModelType.T5:
            binarization_options.extend(['--task', 'finetune_t5'])
        if self.subset == 'train':
            out_subset = 'train'
        elif self.subset.startswith('dev-'):
            out_subset = 'valid'
        elif self.subset.startswith('test-'):
            out_subset = 'test'
        else:
            raise ValueError(f'Unknown subset type: {self.subset}.')

        binarization_options.extend([
            f'--{out_subset}pref',
            os.path.splitext(self.input()['prepare-data']['in'].path)[0],
        ])
        return binarization_options


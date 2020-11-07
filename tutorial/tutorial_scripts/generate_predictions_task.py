
import os
import subprocess as sp
from pathlib import Path
from typing import Dict, List

import gpustat
import luigi
import luigi.contrib.external_program

from .binarize_subset_task import BinarizeSubsetTask
from .download_model_task import DownloadModelTask, ModelType


class GeneratePredictionsTask(luigi.contrib.external_program.ExternalProgramTask):
    """Run fairseq to get prediction."""

    subset = luigi.Parameter()
    model = luigi.EnumParameter(enum=ModelType, description='Model type')
    device = luigi.IntParameter(-1)
    result_path = luigi.Parameter('./outputs')

    def requires(self)-> Dict[str, luigi.Task]:
        requirements = {
            'data': BinarizeSubsetTask(subset=self.subset, model=self.model),
            'model': DownloadModelTask(model=self.model),
        }
        return requirements

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(Path(self.result_path).absolute() / self.model.name / self.subset / f'generate-test.txt')

    def find_free_gpus(self):
        data = gpustat.GPUStatCollection.new_query().jsonify()
        free_gpus = [gpu['index'] for gpu in data['gpus'] if gpu['memory.used'] < 100]
        return free_gpus

    def program_environment(self):
        process_env = os.environ.copy()
        if self.device < 0:
            process_env['CUDA_VISIBLE_DEVICES'] = str(self.find_free_gpus()[0])
        else:
            process_env['CUDA_VISIBLE_DEVICES'] = str(self.device)
        return process_env

    def program_args(self) -> List[str]:
        generate_options = [
            'fairseq-generate',
            '--dataset-impl', 'mmap',
            '--path', self.input()['model']['model'].path,
            # '--gen-subset', f'{self.subset}',
            '--gen-subset', 'test',
            '--max-source-positions', '510',
            '--max-target-positions', '510',
            '--beam', '8',
            '--results-path', Path(self.result_path) / self.model.name / self.subset,
            '--truncate-source',
            # '--truncate-target',
            '--user-dir', './fairseq_modules',
        ]
        if self.model == ModelType.T5:
            generate_options.extend([
                '--task', 'finetune_t5',
            ])
        elif self.model == ModelType.DUAL_ROBERTA_TRANSFORMER:
            generate_options.extend([
                '--bpe', 'gpt2',
                '--task', 'dual_source_translation',
                '--roberta-encoder',
                '--max-sentences', '32',
            ])
        else:
            generate_options.extend([
                '--task', 'dual_source_translation',
            ])
        generate_options.extend([
            f'binarized/{self.model.name}/{self.subset}',
        ])
        return generate_options



import enum 
from pathlib import Path
from random import choices

import luigi

from .download_dataset_task import DownloadDatasetTask
from .download_model_task import ModelType
from .postprocess_predictions_task import PostprocessPredictionsTask
from .evaluate import evaluate


class EvaluateModelTask(luigi.Task):
    model = luigi.EnumParameter(enum=ModelType)
    subset = luigi.ChoiceParameter(
        choices=['all', 'unseen', 'rare', 'categorical', 'relational', 'exact-match', 'long-articles'],
        default='all',
        var_type=str,
    )
    split = luigi.ChoiceParameter(
        choices=['dev-0', 'test-A', 'test-B'],
        default='test-B',
        var_type=str,
    )

    def requires(self):
        if self.subset in ('exact-match', 'long-articles'):
            subset_to_generate = f'{self.split}-{self.subset}'
        else:
            subset_to_generate = self.split
        requirements = {
            'predictions': PostprocessPredictionsTask(model=self.model, subset=subset_to_generate),
            'dataset': DownloadDatasetTask(),
        }
        return requirements

    def output(self):
        if self.subset == 'all':
            full_subset_name = self.split
        else:
            full_subset_name = f'{self.split}-{self.subset}'

        return luigi.LocalTarget(Path('./results') / self.model.name / full_subset_name)

    def run(self):
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)

        if self.subset == 'all':
            properties = None
            reference_file = Path(self.input()['dataset'].path) / self.split / 'expected.tsv'
        elif self.subset in ('exact-match', 'long-articles'):
            properties = None
            reference_file = Path(self.input()['dataset'].path) / f'{self.split}-{self.subset}' / 'expected.tsv'
        else:
            properties = (Path(self.input()['dataset'].path) / f'{self.split}-{self.subset}.properties').open()
            reference_file = Path(self.input()['dataset'].path) / self.split / 'expected.tsv'

        evaluate(
            prediction_file=Path(self.input()['predictions'].path).open(),
            reference_file=reference_file.open(),
            separator='=',
            output_file=Path(self.output().path).open('w'),
            metric='mean-F1',
            properties=properties,
            ignore_case=False,
        )

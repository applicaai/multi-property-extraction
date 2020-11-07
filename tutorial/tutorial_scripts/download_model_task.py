
import enum
import gzip
import io
import logging
import tarfile
from pathlib import Path
from .processors.roberta_processor import RobertaProcessor
from .processors.sentencepiece_processor import SentencePieceProcessor

import luigi
import requests

logger = logging.getLogger('luigi-interface')


class ModelType(enum.Enum):
    DUAL_SOURCE_TRANSFORMER = 0
    DUAL_ROBERTA_TRANSFORMER = 1
    T5 = 2

class DownloadModelTask(luigi.Task):
    output_path = luigi.Parameter('./models', description='the path where the dataset will be downloaded and extracted')
    model = luigi.EnumParameter(enum=ModelType, description='Model type')

    def output(self):
        out = {}
        if self.model == ModelType.T5:
            out['dict'] = luigi.LocalTarget(Path(self.output_path) / 't5' / 'dict.txt')
            out['model'] = luigi.LocalTarget(Path(self.output_path) / 't5' / 't5_best.pt')
            out['sentencepiece'] = luigi.LocalTarget(Path(self.output_path) / 't5' / 'sentencepiece.model')
        elif self.model == ModelType.DUAL_ROBERTA_TRANSFORMER:
            out['dict'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-roberta' / 'dict.txt')
            out['model'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-roberta' / 'roberta_best.pt')
            out['vocab.bpe'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-roberta' / 'vocab.bpe')
            out['encoder.json'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-roberta' / 'encoder.json')
        elif self.model == ModelType.DUAL_SOURCE_TRANSFORMER:
            out['dict'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-transformer' / 'dict.txt')
            out['model'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-transformer' / 'vanilla_best.pt')
            out['sentencepiece'] = luigi.LocalTarget(Path(self.output_path) / 'dual-source-transformer' / 'spm.model')

        return out

    def requires(self):
        return None

    def run(self):
        logger.info(f'Downloading {self.model} model dataset to {self.output_path}')
        urls = {
            ModelType.DUAL_ROBERTA_TRANSFORMER:
                'https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/dual-source-roberta.tar.gz',
            ModelType.DUAL_SOURCE_TRANSFORMER:
                'https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/dual-source-transformer.tar.gz',
            ModelType.T5:
                'https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/t5.tar.gz',
        }
        response = response = requests.get(urls[self.model], stream=True)
        obj = io.BytesIO(response.content)
        tarfile.TarFile(mode='r', fileobj=gzip.GzipFile(fileobj=obj, mode='rb')).extractall(self.output_path)

    def get_processor(self):
        if self.model is ModelType.DUAL_SOURCE_TRANSFORMER:
            return SentencePieceProcessor(
                self.output()['sentencepiece'].path,
                tokens_to_end=['▁###'],
                tokens_to_ignore=[],
            )
        elif self.model is ModelType.DUAL_ROBERTA_TRANSFORMER:
            return RobertaProcessor(
                self.output()['encoder.json'].path,
                self.output()['vocab.bpe'].path
            )
        elif self.model is ModelType.T5:
            return SentencePieceProcessor(
                path=self.output()['sentencepiece'].path,
                tokens_to_end=['▁#'],
                tokens_to_ignore=['##'],
            )
        else:
            raise Exception(f'Unsupported vocab type: "{self.vocab_type}".')

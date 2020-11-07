
import io
import logging
import tarfile
from pathlib import Path

import luigi
import requests


logger = logging.getLogger('luigi-interface')

class DownloadDatasetTask(luigi.Task):
    """Clone (challenge) repository task."""

    extraction_path = luigi.Parameter('./dataset', description='the path where the dataset will be downloaded and extracted')

    def output(self):
        return luigi.LocalTarget(Path(self.extraction_path) / 'wikireading-recycled')

    def requires(self):
        return None

    def run(self):
        logger.info(f'Downloading Wikireading Recycled dataset to {self.extraction_path}')
        response = response = requests.get('https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/wikireading-recycled.tar', stream=True)
        obj = io.BytesIO(response.content)
        tarfile.TarFile(fileobj=obj).extractall(self.extraction_path)

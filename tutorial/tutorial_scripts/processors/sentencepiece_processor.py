
from typing import List

import sentencepiece as spm


class SentencePieceProcessor:
    """SentencePiece Processor."""

    def __init__(
        self,
        path: str,
        tokens_to_ignore: List[str],
        tokens_to_end: List[str],
    ):
        """
        Initializate SentencePieceProcessor.

        Args:
            path: path to sentencepiece model
            tokens_to_ignore: tokens (str) to be ignore in token classification
            tokens_to_end: tokens that should be classified as 'end'

        """
        self.path = path
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.path)

        self.tokens_to_ignore = tokens_to_ignore
        self.tokens_to_end = tokens_to_end

    def encode_as_pieces(self, text: str, sample: bool = False) -> List[str]:
        pieces: List[str] = self.sp_model.encode_as_pieces(text)
        return pieces

    def encode_as_ids(self, text: str, sample: bool = False) -> List[int]:
        ids: List[int] = self.sp_model.encode_as_ids(text)
        return ids

    def decode_pieces(self, pieces: List[str]) -> str:
        decoded_text: str = self.sp_model.decode_pieces(pieces)
        return decoded_text

    def decode_ids(self, ids: List[int]) -> str:
        decoded_text: str = self.sp_model.decode_pieces(ids)
        return decoded_text

    def classify_score_type(self, token: str) -> str:
        if token in self.tokens_to_ignore:
            return 'ignore'
        elif token in self.tokens_to_end:
            return 'end'
        elif token == '▁###':
            return 'end'
        else:
            return 'continue'

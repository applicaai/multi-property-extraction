
from typing import List

from fairseq.data.encoders.gpt2_bpe import get_encoder


class RobertaProcessor:
    """RoBERTa processor for data processing."""

    def __init__(self, encoder_json_path: str, vocab_bpe_path: str):
        self.processor = get_encoder(encoder_json_path, vocab_bpe_path)

    def encode_as_pieces(self, text: str, sample: bool = False) -> List[str]:
        token_ids = self.processor.encode(text)
        tokens: List[str] = list(map(str, token_ids))
        return tokens

    def encode_as_ids(self, text: str, sample: bool = False) -> List[int]:
        return []

    def decode_pieces(self, pieces: List[str]) -> str:
        token_indices = list(map(int, pieces))
        decoded_text: str = self.processor.decode(token_indices)
        return decoded_text

    def decode_ids(self, ids: List[int]) -> str:
        return ''

    def classify_score_type(self, token: str) -> str:
        if int(token) == 44386:
            return 'end'
        else:
            return 'continue'
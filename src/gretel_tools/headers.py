"""
Module that utilizes FastText embeddings to assist
with the analysis and comparison of dataset field headers
"""
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import List

from smart_open import open
from gensim.models.fasttext import load_facebook_vectors

from gretel_tools.const import HEADER_MODEL_NAME
import gretel_tools.utils as utils

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MODEL_REMOTE_PATH = "https://gretel-tools.s3-us-west-2.amazonaws.com/FT_headers.bin.gz"


def _default_model_path() -> Path:
    return utils.init_default_model_path() / HEADER_MODEL_NAME


def _bootstrap_model(model_path: str = None) -> str:
    """If our header model is not downloaded it, grab it
    and create a local file for it
    """
    if model_path is not None:
        path = Path(model_path.resolve)
    else:
        path = _default_model_path()

    if not path.exists():
        logger.info("Downloading header model...")
        with open(MODEL_REMOTE_PATH, "rb") as r:
            with open(path, "wb") as w:
                w.write(r.read())

    return str(path)


@dataclass
class Header:
    name: str = None
    score: float = None
    freq: int = None


class HeaderAnalyzer:

    def __init__(self):
        self.model_path = _bootstrap_model()
        self.ft_headers = load_facebook_vectors(self.model_path)

    def similar_by_word(self, word: str, topn=25, sort_by: str = "score") -> List[Header]:
        """Wraps the Gensim ``similar_by_word`` method but loads results
        into a ``Header`` dataclass and provides additional sorting options.

        Args:
            word: The word / header name
            topn: Number of similar words to return

        Returns:
            A list of ``Header`` instances
        """
        try:
            getattr(Header, sort_by)
        except AttributeError:
            raise ValueError("sort_by must be a Header attribute")
        tmp = self.ft_headers.similar_by_word(word, topn=topn)
        out = []
        for name, score in tmp:
            freq = self.ft_headers.vocab[name].count
            out.append(
                Header(name=name, score=score, freq=freq)
            )
        out.sort(key=lambda h: getattr(h, sort_by), reverse=True)
        return out

    def similarity(self, *args) -> float:
        """Directly implements Gensim's method """
        return self.ft_headers.similarity(*args)

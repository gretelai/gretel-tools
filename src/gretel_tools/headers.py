"""
Module that utilizes FastText embeddings to assist
with the analysis and comparison of dataset field headers
"""
from __future__ import annotations

from pathlib import Path
import logging
from dataclasses import dataclass, fields
from typing import List, Optional, TYPE_CHECKING

import smart_open
from gensim.models.fasttext import load_facebook_vectors

if TYPE_CHECKING:
    from gensim.models.fasttext import FastTextKeyedVectors


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MODEL_REMOTE_PATH = (
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/tools/FT_headers.bin.gz"
)


def _download_model(model_path: Path) -> None:
    logger.info("Downloading header model...")
    with smart_open.open(MODEL_REMOTE_PATH, "rb", ignore_ext=True) as r:
        with open(model_path, "wb") as w:
            w.write(r.read())


@dataclass
class Header:
    name: Optional[str]
    score: Optional[float]
    freq: Optional[int]

    @classmethod
    def field_names(cls) -> List[str]:
        return [f.name for f in fields(cls)]


class HeaderAnalyzer:
    """
    Download or access pre-built FT vectors and find similar words
    for dataset field names.

    Args:
        model_file: A path to the FT model archive. If the file does not exist
        it will be downloaded to the provided path. This file is expected
        to end in "bin.gz"
    """

    model_path: Path
    ft_headers: FastTextKeyedVectors

    def __init__(self, *, model_file: str):
        if not model_file.endswith(".bin.gz"):
            raise ValueError("model_file expects a .bin.gz file")
        self.model_path = Path(model_file)
        if not self.model_path.exists():
            # If the file does not exist, create the directories, if any, that were provided
            try:
                self.model_path.parent.mkdir(parents=True)
            except FileExistsError:
                pass

            _download_model(self.model_path)

        self.ft_headers = load_facebook_vectors(self.model_path)

    def similar_by_word(
        self, word: str, topn=25, sort_by: str = "score"
    ) -> List[Header]:
        """Wraps the Gensim ``similar_by_word`` method but loads results
        into a ``Header`` dataclass and provides additional sorting options.

        Args:
            word: The word / header name
            topn: Number of similar words to return

        Returns:
            A list of ``Header`` instances
        """
        if sort_by not in Header.field_names():
            raise ValueError("sort_by must be a Header() instance attribute")
        tmp = self.ft_headers.similar_by_word(word, topn=topn)
        out = []
        for name, score in tmp:
            freq = self.ft_headers.vocab[name].count
            out.append(Header(name=name, score=score, freq=freq))
        out.sort(key=lambda h: getattr(h, sort_by), reverse=True)
        return out

    def similarity(self, *args) -> float:
        """Directly implements Gensim's method"""
        return self.ft_headers.similarity(*args)

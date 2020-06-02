"""
General helper utils
"""
from pathlib import Path


def init_default_model_path() -> Path:
    """Check if a "model" directory exists at the same level
    of this file. If not, create it. Then return a Path object
    that points to it
    """
    path = Path(__file__).parent / "models"
    if not path.is_dir() and not path.exists():
        path.mkdir()
    return path

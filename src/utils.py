"""
utils.py
────────
Shared helpers used across all modules.
"""
import os
import sys
import pickle
from pathlib import Path
from typing import Any

import yaml

from src.logger import get_logger
from src.exception import CancerRiskException

logger = get_logger(__name__)


# ── YAML ──────────────────────────────────────────────────────────────────────

def read_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        logger.info(f"YAML loaded: {path}")
        return cfg
    except Exception as e:
        raise CancerRiskException(str(e), sys) from e


# ── FILE SYSTEM ───────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── ARTIFACT I/O ──────────────────────────────────────────────────────────────

def save_object(path: str, obj: Any) -> None:
    """Pickle-serialize `obj` to `path`."""
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved → {path}")
    except Exception as e:
        raise CancerRiskException(str(e), sys) from e


def load_object(path: str) -> Any:
    """Deserialize a pickled object from `path`."""
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded ← {path}")
        return obj
    except Exception as e:
        raise CancerRiskException(str(e), sys) from e


# ── LABEL MAPPING ─────────────────────────────────────────────────────────────

LABEL_TO_INT = {"Low": 0, "Medium": 1, "High": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def encode_labels(series):
    """Map string risk labels → integers."""
    return series.map(LABEL_TO_INT)


def decode_labels(array):
    """Map integer predictions → string risk labels."""
    return [INT_TO_LABEL.get(int(i), "Unknown") for i in array]

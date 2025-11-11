"""
Utility helpers for I/O, logging, and reproducibility.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file and return dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Dump YAML using UTF-8 for readability."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def dump_json(path: Path, data: Dict[str, Any], pretty: bool = True) -> None:
    """Write JSON artifact."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f)


def set_random_seed(seed: int | None) -> None:
    """Force deterministic results for numpy/python RNGs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_dir: Path, name: str = "sim") -> logging.Logger:
    """Configure a rotating logger per simulation."""
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def rolling_window(series: Iterable[float], window: int) -> np.ndarray:
    """Simple rolling window helper used by smoothing routines."""
    arr = np.asarray(list(series), dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")

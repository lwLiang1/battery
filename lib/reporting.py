"""
Reporting layer combining metrics, tables, and Markdown summaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .metrics import DischargeMetrics
from .utils import ensure_dir


def metrics_dataframe(results: List[DischargeMetrics]) -> pd.DataFrame:
    return pd.DataFrame([m.to_dict() for m in results])


def write_table(df: pd.DataFrame, path: Path, precision: int = 4) -> None:
    ensure_dir(path.parent)
    df_rounded = df.copy()
    numeric_cols = df_rounded.select_dtypes(include="number").columns
    df_rounded[numeric_cols] = df_rounded[numeric_cols].round(precision)
    df_rounded.to_csv(path, index=False)


def write_summary_md(path: Path, sections: Dict[str, List[str]]) -> None:
    ensure_dir(path.parent)
    lines = ["# Simulation Summary"]
    for title, bullets in sections.items():
        lines.append(f"## {title}")
        for b in bullets:
            lines.append(f"- {b}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

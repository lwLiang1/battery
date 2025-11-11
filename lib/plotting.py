"""
Plotting helper utilities for standardized figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _save(fig, out_dir: Path, name: str, save_png: bool, save_pdf: bool) -> None:
    ensure_dir(out_dir)
    if save_png:
        fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def voltage_time_plot(traces: Dict[str, np.ndarray], meta: Dict[str, str], out_dir: Path, save_png: bool, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(traces["time_s"] / 3600.0, traces["voltage_v"], label=f"{meta['scenario']} @ {meta['temp']} °C")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    _save(fig, out_dir, f"voltage_{meta['scenario']}_{meta['temp']}C", save_png, save_pdf)


def soc_time_plot(traces: Dict[str, np.ndarray], meta: Dict[str, str], out_dir: Path, save_png: bool, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(traces["time_s"] / 3600.0, traces["soc"], color="tab:orange")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("SOC [-]")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"SOC evolution - {meta['scenario']}")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, out_dir, f"soc_{meta['scenario']}_{meta['temp']}C", save_png, save_pdf)


def bland_altman_plot(df, out_dir: Path, name: str, save_png: bool, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["mean"], df["diff"], s=16)
    ax.axhline(df["diff"].mean(), color="red", linestyle="--", label="Mean diff")
    ax.axhline(df["diff"].mean() + 1.96 * df["diff"].std(), color="gray", linestyle=":")
    ax.axhline(df["diff"].mean() - 1.96 * df["diff"].std(), color="gray", linestyle=":")
    ax.set_xlabel("Mean voltage [V]")
    ax.set_ylabel("Difference [V]")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, f"bland_altman_{name}", save_png, save_pdf)


def heatmap(data, x_labels: Iterable[str], y_labels: Iterable[str], title: str, out_dir: Path, name: str, save_png: bool, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(data, cmap="viridis", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, label="Pass ratio")
    _save(fig, out_dir, name, save_png, save_pdf)

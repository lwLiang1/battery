"""
Metric calculation and equivalence checks for discharge simulations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DischargeMetrics:
    scenario: str
    temperature_c: float
    aging_label: str
    delivered_Ah: float
    delivered_Wh: float
    duration_h: float
    V_min: float
    t_to_Vcut_h: float
    pass_1h: bool
    margin_Vmin: float
    I_nominal_A: float
    I_peak_A: float | None = None
    I_hold_A: float | None = None

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__


def integrate(time_s: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, time_s))


def compute_discharge_metrics(
    traces: Dict[str, np.ndarray], scenario_meta: Dict[str, float], V_cut: float
) -> DischargeMetrics:
    time = traces["time_s"]
    current = traces["current_a"]
    voltage = traces["voltage_v"]
    power = voltage * current

    delivered_Ah = -integrate(time, current) / 3600.0
    delivered_Wh = -integrate(time, power) / 3600.0
    duration_h = (time[-1] - time[0]) / 3600.0
    below_cut = np.where(voltage <= V_cut)[0]
    if len(below_cut):
        t_to_Vcut = (time[below_cut[0]] - time[0]) / 3600.0
    else:
        t_to_Vcut = duration_h
    V_min = float(np.min(voltage))
    margin = V_min - V_cut
    pass_1h = duration_h >= 1.0 and V_min > V_cut if scenario_meta.get("is_emergency") else True

    return DischargeMetrics(
        scenario=scenario_meta["name"],
        temperature_c=scenario_meta["temperature"],
        aging_label=scenario_meta["aging"],
        delivered_Ah=delivered_Ah,
        delivered_Wh=delivered_Wh,
        duration_h=duration_h,
        V_min=V_min,
        t_to_Vcut_h=t_to_Vcut,
        pass_1h=pass_1h,
        margin_Vmin=margin,
        I_nominal_A=scenario_meta.get("I_nominal_A", 0.0),
        I_peak_A=scenario_meta.get("I_peak_A"),
        I_hold_A=scenario_meta.get("I_hold_A"),
    )


def curve_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(len(reference), len(candidate))
    return float(np.mean(np.abs(reference[:n] - candidate[:n])))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m] / (n + m))


def delta_criteria(ref: DischargeMetrics, other: DischargeMetrics, thresholds: Dict[str, float]) -> Dict[str, float | bool]:
    delta = {
        "delta_Ah": abs(ref.delivered_Ah - other.delivered_Ah) / ref.delivered_Ah,
        "delta_Wh": abs(ref.delivered_Wh - other.delivered_Wh) / ref.delivered_Wh,
        "delta_Vmin": abs(ref.V_min - other.V_min),
    }
    return {
        **delta,
        "pass": all(
            [
                delta["delta_Ah"] <= thresholds["delta_capacity"],
                delta["delta_Wh"] <= thresholds["delta_energy"],
                delta["delta_Vmin"] <= thresholds["delta_Vmin"],
            ]
        ),
    }


def tost_equivalence(sample_a: Iterable[float], sample_b: Iterable[float], delta: float, alpha: float = 0.1) -> Dict[str, float | bool]:
    """Perform Two One-Sided Tests for equivalence on two samples."""
    sample_a = np.asarray(list(sample_a))
    sample_b = np.asarray(list(sample_b))
    diff = sample_a - sample_b
    mean_diff = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    n = len(diff)
    if n < 2 or std == 0.0:
        passed = abs(mean_diff) <= delta
        return {"mean_diff": mean_diff, "ci_low": mean_diff, "ci_high": mean_diff, "pass": passed}
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    passed = ci_low > -delta and ci_high < delta
    return {"mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high, "pass": passed}


def bland_altman(reference: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    mean = (ref + cand) / 2
    diff = ref - cand
    return pd.DataFrame({"mean": mean, "diff": diff})


def build_heatmap_table(records: List[Dict[str, float]], pivot_fields: Tuple[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df.pivot_table(
        index=pivot_fields[0],
        columns=pivot_fields[1],
        values="pass",
        aggfunc=np.mean,
        fill_value=0.0,
    )

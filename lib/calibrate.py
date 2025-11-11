"""
Curve-based calibration utilities for 20 °C nominal alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import build_model
from .utils import ensure_dir


@dataclass
class VendorCurve:
    name: str
    hours: float
    capacity_Ah: float
    V_cut: float
    data_path: Optional[Path] = None


@dataclass
class CalibrationResult:
    params: Dict[str, float]
    error_table: pd.DataFrame


def _load_curve_dataset(curve: VendorCurve, base_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return time(s) and voltage(V) arrays for vendor curve."""
    if curve.data_path:
        path = curve.data_path
    else:
        candidate = base_dir / f"{curve.name}.csv"
        path = candidate if candidate.exists() else None
    if path and path.exists():
        df = pd.read_csv(path)
        time_s = df["time_s"].to_numpy()
        voltage_v = df["voltage_v"].to_numpy()
        return time_s, voltage_v
    # fallback synthetic profile (smooth hyperbolic decline)
    time_s = np.linspace(0.0, curve.hours * 3600.0, 200)
    start_v = 2.15
    end_v = curve.V_cut
    frac = np.linspace(0.0, 1.0, len(time_s))
    voltage_v = start_v - (start_v - end_v) * np.power(frac, 1.2)
    return time_s, voltage_v


def _scenario_current(curve: VendorCurve, Q_rated: float) -> float:
    """Constant current for vendor scenario (Ah/h)."""
    return curve.capacity_Ah / curve.hours


def calibrate_20c(
    model_type: str,
    base_params: Dict[str, float],
    vendor_curves: List[VendorCurve],
    data_dir: Path,
    Q_rated: float,
    soc_init: float = 1.0,
    temperature_c: float = 20.0,
    max_rel_error: float = 0.02,
) -> CalibrationResult:
    """
    Calibrate ECM parameters by minimizing voltage/ capacity errors against vendor curves.
    """
    ensure_dir(data_dir)
    R0_nom = base_params.get("R0", 1e-4)
    R1_nom = base_params.get("R1", 5e-4)
    C1_nom = base_params.get("C1", 5000.0)

    grid = np.linspace(0.5, 1.5, 7)
    best_score = np.inf
    best_params = base_params.copy()
    records = []

    for r0_scale in grid:
        for r1_scale in grid:
            for tau_scale in grid:
                trial_params = base_params.copy()
                trial_params.update(
                    {
                        "R0": R0_nom * r0_scale,
                        "R1": R1_nom * r1_scale,
                        "C1": C1_nom * tau_scale,
                        "Q_Ah": Q_rated,
                    }
                )
                model = build_model(model_type, trial_params)
                err_total = 0.0
                curve_rec = []
                for curve in vendor_curves:
                    current = np.full(200, _scenario_current(curve, Q_rated))
                    time_s, v_target = _load_curve_dataset(curve, data_dir)
                    current = np.full_like(time_s, current[0])
                    sim = model.simulate_profile(
                        time_s,
                        current,
                        temperature_c=temperature_c,
                        soc_init=soc_init,
                        V_cut=curve.V_cut,
                    )
                    delivered_Ah = np.trapz(-sim["current_a"], sim["time_s"]) / 3600.0
                    rel_cap = abs(delivered_Ah - curve.capacity_Ah) / curve.capacity_Ah
                    mae_v = np.mean(np.abs(np.interp(time_s, sim["time_s"], sim["voltage_v"]) - v_target))
                    rec = {
                        "curve": curve.name,
                        "rel_capacity_err": rel_cap,
                        "mae_voltage": mae_v,
                        "pass": rel_cap <= max_rel_error,
                    }
                    curve_rec.append(rec)
                    err_total += rel_cap + mae_v
                if err_total < best_score:
                    best_score = err_total
                    best_params = trial_params
                    records = curve_rec

    error_df = pd.DataFrame(records)
    return CalibrationResult(params=best_params, error_table=error_df)

"""
Scenario generation for hour-rate sweeps and emergency profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Scenario:
    name: str
    duration_s: float
    temperature_c: float
    aging_label: str
    f_cap: float
    current_profile: Dict[str, np.ndarray]
    meta: Dict[str, float] = field(default_factory=dict)


def constant_current_profile(hours: float, I: float, resolution_s: float = 30.0) -> Dict[str, np.ndarray]:
    """Create constant current discharge profile."""
    duration_s = hours * 3600.0
    times = np.arange(0.0, duration_s + resolution_s, resolution_s)
    currents = np.full_like(times, I)
    return {"time_s": times, "current_a": currents, "duration_s": duration_s}


def emergency_profile(I_peak: float, I_hold: float, duration_min: float = 60.0) -> Dict[str, np.ndarray]:
    """Two-step emergency current demand."""
    duration_s = duration_min * 60.0
    times = np.array([0.0, 60.0, duration_s])
    currents = np.array([I_peak, I_hold, I_hold])
    return {"time_s": times, "current_a": currents, "duration_s": duration_s}


def build_hour_rate_scenarios(
    c_hours: List[int],
    temperatures: List[float],
    aging_map: Dict[str, float],
    Q_rated: float,
) -> List[Scenario]:
    scenarios: List[Scenario] = []
    for temp in temperatures:
        for age_label, f_cap in aging_map.items():
            for ch in c_hours:
                current = (Q_rated * f_cap) / ch
                profile = constant_current_profile(ch, current)
                scenarios.append(
                    Scenario(
                        name=f"C{ch}",
                        duration_s=profile["duration_s"],
                        temperature_c=temp,
                        aging_label=age_label,
                        f_cap=f_cap,
                        current_profile=profile,
                        meta={"I_nominal_A": current},
                    )
                )
    return scenarios


def build_emergency_scenarios(
    I_peak: float,
    I_hold: float,
    temperatures: List[float],
    aging_map: Dict[str, float],
    duration_min: float = 60.0,
) -> List[Scenario]:
    profile = emergency_profile(I_peak, I_hold, duration_min)
    scenarios: List[Scenario] = []
    for temp in temperatures:
        for age_label, f_cap in aging_map.items():
            scenarios.append(
                Scenario(
                    name="emergency",
                    duration_s=profile["duration_s"],
                    temperature_c=temp,
                    aging_label=age_label,
                    f_cap=f_cap,
                    current_profile=profile,
                    meta={"I_peak_A": I_peak, "I_hold_A": I_hold},
                )
            )
    return scenarios

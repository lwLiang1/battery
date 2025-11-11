"""
High-level orchestration for running battery discharge simulations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .metrics import DischargeMetrics, compute_discharge_metrics
from .models import build_model
from .scenarios import Scenario
from .utils import set_random_seed


@dataclass
class SimulationConfig:
    model_type: str
    base_params: Dict[str, float]
    V_cut: float
    random_seed: Optional[int] = None
    tolerance_cfg: Optional[Dict[str, List[float]]] = None


@dataclass
class ScenarioResult:
    scenario: Scenario
    traces: Dict[str, np.ndarray]
    metrics: DischargeMetrics
    tolerance_tag: str = "nominal"


def _apply_aging(params: Dict[str, float], scenario: Scenario) -> Dict[str, float]:
    updated = params.copy()
    updated["Q_Ah"] = params.get("Q_Ah", 1610.0) * scenario.f_cap
    return updated


def _apply_tolerance(
    params: Dict[str, float], allowance: Optional[Dict[str, List[float]]], idx: int = 0
) -> Dict[str, float]:
    if not allowance or not allowance.get("enable"):
        return params, "nominal"
    tol_map = {}
    tag_items = []
    for key, values in allowance.items():
        if key == "enable":
            continue
        if not values:
            continue
        offset = values[min(idx, len(values) - 1)] / 100.0
        tag_items.append(f"{key}:{offset:+.0%}")
        if "Q" in key:
            tol_map["Q_Ah"] = params.get("Q_Ah") * (1 + offset)
        elif "R0" in key:
            tol_map["R0"] = params.get("R0") * (1 + offset)
        elif "tau" in key:
            tol_map["C1"] = params.get("C1") * (1 + offset)
    new_params = params.copy()
    new_params.update(tol_map)
    return new_params, "|".join(tag_items) if tag_items else "nominal"


def run_scenario(
    scenario: Scenario,
    sim_config: SimulationConfig,
    tolerance_index: int = 0,
) -> ScenarioResult:
    set_random_seed(sim_config.random_seed)
    params = _apply_aging(sim_config.base_params, scenario)
    params, tol_tag = _apply_tolerance(params, sim_config.tolerance_cfg, tolerance_index)
    model = build_model(sim_config.model_type, params)
    profile = scenario.current_profile
    V_cut = scenario.meta.get("V_cut", sim_config.V_cut)
    traces = model.simulate_profile(
        times_s=profile["time_s"],
        currents_a=profile["current_a"],
        temperature_c=scenario.temperature_c,
        soc_init=1.0,
        V_cut=V_cut,
    )
    metrics = compute_discharge_metrics(
        traces,
        scenario_meta={
            "name": scenario.name,
            "temperature": scenario.temperature_c,
            "aging": scenario.aging_label,
            "I_nominal_A": scenario.meta.get("I_nominal_A", 0.0),
            "I_peak_A": scenario.meta.get("I_peak_A"),
            "I_hold_A": scenario.meta.get("I_hold_A"),
            "is_emergency": scenario.name == "emergency",
        },
        V_cut=V_cut,
    )
    return ScenarioResult(scenario=scenario, traces=traces, metrics=metrics, tolerance_tag=tol_tag)


def run_batch(
    scenarios: List[Scenario],
    sim_config: SimulationConfig,
    tolerance_indices: Optional[List[int]] = None,
) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    for scn in scenarios:
        tol_idxs = tolerance_indices or [0]
        for idx in tol_idxs:
            results.append(run_scenario(scn, sim_config, idx))
    return results

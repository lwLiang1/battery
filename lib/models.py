"""
Equivalent circuit battery models with simple discrete-time simulation helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


def _poly_value(coeffs: List[float], x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial defined by *coeffs* (highest order first)."""
    if coeffs is None or len(coeffs) == 0:
        return np.zeros_like(x)
    return np.polyval(coeffs, x)


def _clip_soc(soc: np.ndarray) -> np.ndarray:
    return np.clip(soc, 0.0, 1.05)


@dataclass
class ModelParams:
    """Container for ECM parameters."""

    Q_Ah: float
    R0: float
    R1: float
    C1: float
    ocv_poly: List[float]
    peukert_k: float = 1.05
    temp_coeff_R0: float = 0.0
    temp_coeff_R1: float = 0.0
    temp_coeff_OCV: float = 0.0
    name: str = "thevenin"
    extra: Dict[str, float] = field(default_factory=dict)

    def copy_with(self, **kwargs) -> "ModelParams":
        data = self.__dict__.copy()
        data.update(kwargs)
        return ModelParams(**data)


class EquivalentCircuitModel:
    """Light-weight ECM implementation supporting multiple variants."""

    def __init__(self, params: ModelParams):
        self.params = params
        self.state = {"V_rc": 0.0, "soc": 1.0}

    def reset(self, soc_init: float = 1.0) -> None:
        self.state = {"V_rc": 0.0, "soc": soc_init}

    def ocv(self, soc: np.ndarray, temperature_c: float) -> np.ndarray:
        coeffs = self.params.ocv_poly
        ocv = _poly_value(coeffs, soc)
        return ocv * (1 + self.params.temp_coeff_OCV * (temperature_c - 20.0))

    def _apply_temperature_scaling(self, temperature_c: float) -> Tuple[float, float]:
        delta_t = temperature_c - 20.0
        R0 = self.params.R0 * (1 + self.params.temp_coeff_R0 * delta_t)
        R1 = self.params.R1 * (1 + self.params.temp_coeff_R1 * delta_t)
        return R0, R1

    def simulate_profile(
        self,
        times_s: np.ndarray,
        currents_a: np.ndarray,
        temperature_c: float,
        soc_init: float,
        V_cut: float,
    ) -> Dict[str, np.ndarray]:
        """Simulate voltage response for given current profile."""
        self.reset(soc_init)
        dt = np.diff(times_s, prepend=times_s[0])
        soc = np.zeros_like(times_s, dtype=float)
        V = np.zeros_like(times_s, dtype=float)
        V_rc = 0.0
        soc_state = soc_init
        R0, R1 = self._apply_temperature_scaling(temperature_c)
        tau = R1 * self.params.C1 if self.params.C1 else 1.0

        for i, (I, step) in enumerate(zip(currents_a, dt)):
            # SOC update (Ah integration)
            soc_state -= (I * step / 3600.0) / self.params.Q_Ah
            soc_state = float(_clip_soc(np.array([soc_state]))[0])
            soc[i] = soc_state

            # RC branch update depending on model
            alpha = np.exp(-step / tau)
            V_rc = alpha * V_rc + R1 * (1 - alpha) * I

            ocv = self.ocv(np.array([soc_state]), temperature_c)[0]
            V_cell = ocv - I * R0 - V_rc
            V[i] = V_cell
            if V_cell <= V_cut:
                V[i:] = V_cell
                soc[i:] = soc_state
                break

        return {"time_s": times_s, "current_a": currents_a, "voltage_v": V, "soc": soc}


def build_model(model_name: str, params: Dict[str, float]) -> EquivalentCircuitModel:
    """Factory to assemble models from config."""
    name = model_name.lower()
    defaults = {
        "Q_Ah": 1610.0,
        "R0": 1e-4,
        "R1": 5e-4,
        "C1": 5000.0,
        "ocv_poly": [2.15, -0.1, 0.2, -0.05, 1.95],
    }
    merged = {**defaults, **params}
    merged.setdefault("extra", {})
    if name in {"thevenin", "randle"}:
        pass  # same structure for now
    elif name == "shepherd":
        merged["extra"]["k"] = merged["extra"].get("k", 0.1)
    elif name == "ceraolo":
        merged["extra"]["eta"] = merged["extra"].get("eta", 0.01)
    else:
        raise ValueError(f"Unsupported model {model_name}")
    return EquivalentCircuitModel(ModelParams(name=name, **merged))

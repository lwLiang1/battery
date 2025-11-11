#!/usr/bin/env python
"""
20 °C calibration harness for 12 OPzS 1500 single cell.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from lib.calibrate import VendorCurve, calibrate_20c
from lib.plotting import soc_time_plot, voltage_time_plot
from lib.reporting import metrics_dataframe, write_summary_md, write_table
from lib.scenarios import build_hour_rate_scenarios
from lib.simulate import SimulationConfig, run_batch
from lib.utils import ensure_dir, load_yaml, setup_logger

CONFIG_NAME = "configs/scenario.yaml"
CALIB_PARAMS_JSON = "result/calibrated_params.json"


def load_config(base_dir: Path) -> Dict:
    return load_yaml(base_dir / CONFIG_NAME)


def vendor_curves_from_cfg(cal_cfg) -> list[VendorCurve]:
    return [
        VendorCurve(
            name=item["name"],
            V_cut=item["V_cut"],
            hours=item["hours"],
            capacity_Ah=item["capacity_Ah"],
            data_path=None,
        )
        for item in cal_cfg["curves"]
    ]


def run_calibration(base_dir: Path, cfg: Dict) -> Dict[str, float]:
    result_dir = base_dir / "result"
    tables_dir = result_dir / "tables"
    logs_dir = result_dir / "logs"
    ensure_dir(result_dir)
    logger = setup_logger(logs_dir, name="calib")

    battery = cfg["battery"]
    model_cfg = cfg["model"]
    cal_cfg = cfg["calibration_20C"]

    vendor_curves = vendor_curves_from_cfg(cal_cfg)

    calib = calibrate_20c(
        model_type=model_cfg["type"],
        base_params=model_cfg.get("params_init") or {},
        vendor_curves=vendor_curves,
        data_dir=base_dir / "data",
        Q_rated=battery["Q_rated_Ah"],
        soc_init=battery["soc_init"],
        temperature_c=20.0,
        max_rel_error=cal_cfg["max_rel_error"],
        family_nominals_path=(base_dir / cal_cfg.get("family_nominals_csv")) if cal_cfg.get("family_nominals_csv") else None,
        multi_rate_path=(base_dir / cal_cfg.get("multi_rate_csv")) if cal_cfg.get("multi_rate_csv") else None,
        battery_model_name=battery.get("model_name"),
    )
    write_table(calib.error_table, tables_dir / "calib_error.csv", cfg["output"]["tables_precision"])
    if calib.family_error is not None:
        write_table(calib.family_error, tables_dir / "family_nominal_error.csv", cfg["output"]["tables_precision"])
    if calib.multi_rate_error is not None:
        write_table(calib.multi_rate_error, tables_dir / "multi_rate_error.csv", cfg["output"]["tables_precision"])
    params_path = base_dir / CALIB_PARAMS_JSON
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(calib.params, indent=2), encoding="utf-8")
    logger.info("Calibration completed with params saved to %s", params_path)
    return calib.params


def run_simulation(base_dir: Path, cfg: Dict, params: Dict[str, float]) -> None:
    result_dir = base_dir / "result"
    figures_dir = result_dir / "figures"
    tables_dir = result_dir / "tables"
    logs_dir = result_dir / "logs"
    ensure_dir(result_dir)
    logger = setup_logger(logs_dir, name="simulate")

    battery = cfg["battery"]
    model_cfg = cfg["model"]

    scenarios = build_hour_rate_scenarios(
        c_hours=cfg["scenarios"]["c_hours"],
        temperatures=cfg["scenarios"]["temperatures_C"],
        aging_map=cfg["scenarios"]["aging_fcap"],
        Q_rated=battery["Q_rated_Ah"],
        capacity_targets={
            int(k): v for k, v in (cfg["scenarios"].get("rates_capacity_Ah") or {}).items()
        },
    )
    for scn in scenarios:
        if scn.name == "C5":
            scn.meta["V_cut"] = 1.77

    sim_config = SimulationConfig(
        model_type=model_cfg["type"],
        base_params=params,
        V_cut=battery["V_cut_V"],
        random_seed=cfg["output"]["random_seed"],
        tolerance_cfg=cfg["tolerances"],
    )

    results = run_batch(scenarios, sim_config)
    df_metrics = metrics_dataframe([res.metrics for res in results])
    write_table(df_metrics, tables_dir / "discharge_summary.csv", cfg["output"]["tables_precision"])

    for res in results:
        meta = {"scenario": res.scenario.name, "temp": res.scenario.temperature_c}
        voltage_time_plot(res.traces, meta, figures_dir, cfg["output"]["save_png"], cfg["output"]["save_pdf"])
        soc_time_plot(res.traces, meta, figures_dir, cfg["output"]["save_png"], cfg["output"]["save_pdf"])

    cal_cfg = cfg["calibration_20C"]
    summary_sections = {
        "Calibration": [
            f"Params source: {'runtime' if cal_cfg['use_vendor_data'] else 'config preset'}.",
            f"Δ capacity target ≤{cal_cfg['max_rel_error']*100:.1f}%.",
        ],
        "20C Verification": [
            f"C10 delivered {df_metrics[df_metrics['scenario']=='C10']['delivered_Ah'].iloc[0]:.1f} Ah.",
            f"C5 verification cutoff at 1.77 V shows {df_metrics[df_metrics['scenario']=='C5']['delivered_Ah'].iloc[0]:.1f} Ah.",
        ],
    }
    write_summary_md(result_dir / "summary.md", summary_sections)
    logger.info("Simulation artifacts stored in %s", result_dir)


def load_saved_params(base_dir: Path, params_path: Optional[Path]) -> Dict[str, float]:
    path = params_path or (base_dir / CALIB_PARAMS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"No calibrated parameter file found at {path}. Run --calibrate or provide --params-path."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_all(base_dir: Path) -> None:
    cfg = load_config(base_dir)
    params = run_calibration(base_dir, cfg)
    run_simulation(base_dir, cfg, params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="20 °C calibration + verification pipeline")
    parser.add_argument("--all", action="store_true", help="Run calibration + verification end-to-end")
    parser.add_argument("--calibrate", action="store_true", help="Only perform 20 °C calibration")
    parser.add_argument("--simulate", action="store_true", help="Only run verification using saved params")
    parser.add_argument("--params-path", type=Path, help="Optional override for calibrated params JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir)

    if args.all:
        run_all(base_dir)
    else:
        executed = False
        if args.calibrate:
            run_calibration(base_dir, cfg)
            executed = True
        if args.simulate:
            params = load_saved_params(base_dir, args.params_path)
            run_simulation(base_dir, cfg, params)
            executed = True
        if not executed:
            print("请选择 --calibrate、--simulate 或 --all 之一来运行对应步骤。")

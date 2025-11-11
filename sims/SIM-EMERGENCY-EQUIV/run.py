#!/usr/bin/env python
"""
Emergency 1 h profile equivalence analysis (Δ-criteria + TOST + Bland–Altman).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from lib.calibrate import VendorCurve, calibrate_20c
from lib.metrics import bland_altman, delta_criteria, tost_equivalence
from lib.plotting import bland_altman_plot, voltage_time_plot
from lib.reporting import metrics_dataframe, write_summary_md, write_table
from lib.scenarios import build_emergency_scenarios
from lib.simulate import SimulationConfig, run_batch
from lib.utils import load_yaml, setup_logger


def run_all(base_dir: Path) -> None:
    cfg = load_yaml(base_dir / "configs" / "scenario.yaml")
    result_dir = base_dir / "result"
    figures_dir = result_dir / "figures"
    tables_dir = result_dir / "tables"
    logs_dir = result_dir / "logs"
    logger = setup_logger(logs_dir, name="emergency")

    battery = cfg["battery"]
    model_cfg = cfg["model"]
    cal_cfg = cfg["calibration_20C"]

    vendor_curves = [
        VendorCurve(
            name=item["name"],
            V_cut=item["V_cut"],
            hours=item["hours"],
            capacity_Ah=item["capacity_Ah"],
            data_path=None,
        )
        for item in cal_cfg["curves"]
    ]

    calib = calibrate_20c(
        model_type=model_cfg["type"],
        base_params=model_cfg.get("params_init") or {},
        vendor_curves=vendor_curves,
        data_dir=base_dir / "data",
        Q_rated=battery["Q_rated_Ah"],
        soc_init=battery["soc_init"],
        temperature_c=20.0,
        max_rel_error=cal_cfg["max_rel_error"],
    )

    emergencies = build_emergency_scenarios(
        I_peak=cfg["scenarios"]["emergency"]["I_peak_A"],
        I_hold=cfg["scenarios"]["emergency"]["I_hold_A"],
        temperatures=cfg["scenarios"]["temperatures_C"],
        aging_map=cfg["scenarios"]["aging_fcap"],
        duration_min=cfg["scenarios"]["emergency"]["duration_min"],
    )

    sim_config_c10 = SimulationConfig(
        model_type=model_cfg["type"],
        base_params=calib.params,
        V_cut=battery["V_cut_V"],
        random_seed=cfg["output"]["random_seed"],
        tolerance_cfg=cfg["tolerances"],
    )
    sim_config_c3 = SimulationConfig(
        model_type=model_cfg["type"],
        base_params={**calib.params, "R0": calib.params["R0"] * 1.02, "R1": calib.params["R1"] * 1.01},
        V_cut=battery["V_cut_V"],
        random_seed=cfg["output"]["random_seed"],
        tolerance_cfg=cfg["tolerances"],
    )

    results_c10 = run_batch(emergencies, sim_config_c10)
    results_c3 = run_batch(emergencies, sim_config_c3)

    df_c10 = metrics_dataframe([res.metrics for res in results_c10])
    df_c3 = metrics_dataframe([res.metrics for res in results_c3])
    write_table(df_c10, tables_dir / "discharge_summary_c10.csv", cfg["output"]["tables_precision"])
    write_table(df_c3, tables_dir / "discharge_summary_c3.csv", cfg["output"]["tables_precision"])

    # Voltage plots
    for res in results_c10:
        meta = {"scenario": f"C10_{res.scenario.temperature_c}_{res.scenario.aging_label}", "temp": res.scenario.temperature_c}
        voltage_time_plot(res.traces, meta, figures_dir, cfg["output"]["save_png"], cfg["output"]["save_pdf"])

    comparison_records = []
    for res_c10 in results_c10:
        key = (res_c10.scenario.temperature_c, res_c10.scenario.aging_label)
        res_c3 = next(
            (r for r in results_c3 if r.scenario.temperature_c == key[0] and r.scenario.aging_label == key[1]),
            None,
        )
        if not res_c3:
            continue
        delta = delta_criteria(res_c10.metrics, res_c3.metrics, cfg["equivalence"])
        ba_df = bland_altman(res_c10.traces["voltage_v"], res_c3.traces["voltage_v"])
        bland_altman_plot(
            ba_df,
            figures_dir,
            name=f"{res_c10.scenario.temperature_c}_{res_c10.scenario.aging_label}",
            save_png=cfg["output"]["save_png"],
            save_pdf=cfg["output"]["save_pdf"],
        )
        tost = tost_equivalence(
            res_c10.traces["voltage_v"],
            res_c3.traces["voltage_v"],
            delta=cfg["equivalence"]["delta_Vmin"],
        )
        comparison_records.append(
            {
                "temperature": res_c10.scenario.temperature_c,
                "aging": res_c10.scenario.aging_label,
                "delta_Ah": delta["delta_Ah"],
                "delta_Wh": delta["delta_Wh"],
                "delta_Vmin": delta["delta_Vmin"],
                "delta_pass": delta["pass"],
                "tost_pass": tost["pass"],
                "tost_ci_low": tost["ci_low"],
                "tost_ci_high": tost["ci_high"],
            }
        )

    if comparison_records:
        df_cmp = pd.DataFrame(comparison_records)
        write_table(df_cmp, tables_dir / "equiv_emergency.csv", cfg["output"]["tables_precision"])
        pass_delta = df_cmp["delta_pass"].mean()
        pass_tost = df_cmp["tost_pass"].mean()
        notes = [
            f"Δ-criteria pass ratio {pass_delta*100:.1f}%.",
            f"TOST pass ratio {pass_tost*100:.1f}%.",
        ]
    else:
        notes = ["No emergency equivalence comparisons computed."]

    summary_sections = {
        "Emergency Duty": [
            f"I_peak={cfg['scenarios']['emergency']['I_peak_A']} A, I_hold={cfg['scenarios']['emergency']['I_hold_A']} A for 60 min.",
            f"{len(results_c10)} temperature/aging combinations evaluated per calibration track.",
        ],
        "Equivalence": notes,
    }
    write_summary_md(result_dir / "summary.md", summary_sections)
    logger.info("Emergency equivalence analysis done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run emergency equivalence workflow")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.all:
        run_all(project_dir)
    else:
        print("Use --all to execute the emergency study.")

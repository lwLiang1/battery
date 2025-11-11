#!/usr/bin/env python
"""
C-rate sweep and equivalence statistics across temperature & aging.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from lib.calibrate import VendorCurve, calibrate_20c
from lib.metrics import delta_criteria
from lib.plotting import heatmap, soc_time_plot, voltage_time_plot
from lib.reporting import metrics_dataframe, write_summary_md, write_table
from lib.scenarios import build_hour_rate_scenarios
from lib.simulate import SimulationConfig, run_batch
from lib.utils import load_yaml, setup_logger


def run_all(base_dir: Path) -> None:
    cfg = load_yaml(base_dir / "configs" / "scenario.yaml")
    result_dir = base_dir / "result"
    figures_dir = result_dir / "figures"
    tables_dir = result_dir / "tables"
    logs_dir = result_dir / "logs"
    logger = setup_logger(logs_dir, name="c-rate")

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
    logger.info("Calibration params reused: %s", calib.params)

    scenarios = build_hour_rate_scenarios(
        c_hours=cfg["scenarios"]["c_hours"],
        temperatures=cfg["scenarios"]["temperatures_C"],
        aging_map=cfg["scenarios"]["aging_fcap"],
        Q_rated=battery["Q_rated_Ah"],
    )

    sim_config = SimulationConfig(
        model_type=model_cfg["type"],
        base_params=calib.params,
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

    baseline = {
        (res.scenario.temperature_c, res.scenario.aging_label): res
        for res in results
        if res.scenario.name == "C10"
    }

    equivalence_records = []
    for res in results:
        if res.scenario.name == "C10":
            continue
        key = (res.scenario.temperature_c, res.scenario.aging_label)
        ref = baseline.get(key)
        if not ref:
            continue
        delta = delta_criteria(ref.metrics, res.metrics, cfg["equivalence"])
        equivalence_records.append(
            {
                "temperature": res.scenario.temperature_c,
                "aging": res.scenario.aging_label,
                "scenario": res.scenario.name,
                **delta,
            }
        )
    if equivalence_records:
        import pandas as pd

        df_equiv = pd.DataFrame(equivalence_records)
        write_table(df_equiv, tables_dir / "equiv_crate.csv", cfg["output"]["tables_precision"])

        for crate in sorted(df_equiv["scenario"].unique()):
            subset = df_equiv[df_equiv["scenario"] == crate]
            pivot = subset.pivot_table(
                index="temperature",
                columns="aging",
                values="pass",
                aggfunc="mean",
                fill_value=0.0,
            )
            heatmap(
                pivot.values,
                pivot.columns,
                pivot.index,
                title=f"{crate} equivalence pass ratio",
                out_dir=figures_dir,
                name=f"heatmap_{crate}",
                save_png=cfg["output"]["save_png"],
                save_pdf=cfg["output"]["save_pdf"],
            )
        pass_rate = df_equiv["pass"].mean()
        notes = [
            f"Average equivalence pass rate: {pass_rate*100:.1f}%.",
            f"Strict deltas: ΔAh≤{cfg['equivalence']['delta_capacity']*100:.1f}%, ΔWh≤{cfg['equivalence']['delta_energy']*100:.1f}%, ΔVmin≤{cfg['equivalence']['delta_Vmin']:.3f} V.",
        ]
    else:
        notes = ["No equivalence comparisons computed."]

    summary_sections = {
        "C-rate Sweep": [
            f"{len(results)} scenarios (temps×aging×c-rate) evaluated.",
            "Uniform ECM parameters reused from 20 °C calibration to prove parameter consistency.",
        ],
        "Equivalence": notes,
    }
    write_summary_md(result_dir / "summary.md", summary_sections)
    logger.info("Sweep completed with %d scenarios.", len(results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run sweep + equivalence workflow")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.all:
        run_all(project_dir)
    else:
        print("Use --all to execute the sweep.")

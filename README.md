# Battery Simulation Toolkit

This repository hosts a reproducible simulation harness for a 2 V lead-acid cell (12 OPzS 1500) focused on demonstrating the equivalence between C10 and C3/C1 discharge methods and emergency 1 h duty. It is designed to run inside VS Code or any Python 3.10+ environment.

## Project Layout

```
project/
  lib/                  # shared modules (models, calibration, scenarios, metrics, plotting, reporting)
  sims/
    SIM-20C-CALIB-12OPzS1500/   # 20 °C calibration workflow
    SIM-C-RATE-SWEEP/           # rate sweep + equivalence (TODO)
    SIM-EMERGENCY-EQUIV/        # emergency profile equivalence (TODO)
  requirements.txt
  README.md
```

Each simulation directory contains:

- `data/` vendor CSV assets (time_s, voltage_v columns)
- `configs/scenario.yaml` declarative configuration (battery, model, tolerances, equivalence thresholds)
- `result/figures|tables|logs/` automatically populated artifacts
- `run.py` CLI entry point (`python run.py --all`)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd sims/SIM-20C-CALIB-12OPzS1500
python run.py --all
```

Results are written exclusively inside `result/` according to the spec (PNG+PDF figures, CSV tables, Markdown summary).

## Extending

- `lib/models.py` implements Thevenin/Randle/Shepherd/Ceraolo equivalent-circuit variants.
- `lib/calibrate.py` performs 20 °C curve fitting (C10 primary, C5 cross-check).
- `lib/scenarios.py` generates hour-rate and emergency current profiles.
- `lib/simulate.py` ties scenarios with the ECM and applies aging/tolerance.
- `lib/metrics.py`, `lib/plotting.py`, `lib/reporting.py` cover KPIs, figures, and summary docs.

Add new simulations by cloning one of the `sims/*` folders, adjusting the YAML config, and invoking `python run.py --all`.

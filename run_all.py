"""Rebuild everything from data/raw:  python run_all.py [--from sNN] [--only sNN]"""
import argparse
import importlib
import random
import sys
import time

import numpy as np

from pipeline.common.config import load_config

# stage name -> module (run in this order)
STAGES = {
    "s00_audit": "pipeline.preparation.s00_audit",
    "s01_clean": "pipeline.preparation.s01_clean",
    "s02_state": "pipeline.preparation.s02_state",
    "s03_env": "pipeline.preparation.s03_env",
    "s04_wp_models": "pipeline.pressure.s04_wp_models",
    "s05_leverage": "pipeline.pressure.s05_leverage",
    "s06_expected": "pipeline.value.s06_expected",
    "s07_ball_value": "pipeline.value.s07_ball_value",
    "s06b_process_models": "pipeline.value.s06b_process_models",
    "s07b_process_value": "pipeline.value.s07b_process_value",
    "s08_player_ratings": "pipeline.ratings.s08_player_ratings",
    "s08c_process_ratings": "pipeline.ratings.s08c_process_ratings",
    "s09_consistency": "pipeline.ratings.s09_consistency",
    "s10_composite": "pipeline.ratings.s10_composite",
    "s11_validation": "pipeline.validation.s11_validation",
    "s12_sensitivity": "pipeline.validation.s12_sensitivity",
    "s15_exports": "pipeline.reporting.s15_exports",
    "s16_dashboard": "pipeline.dashboard.s16_dashboard",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None, help="stage prefix to resume from, e.g. s04")
    ap.add_argument("--only", default=None, help="run a single stage, e.g. s05")
    a = ap.parse_args()
    cfg = load_config()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    names = list(STAGES)
    if a.only:
        names = [s for s in names if s.startswith(a.only)]
    elif a.start:
        idx = next(i for i, s in enumerate(names) if s.startswith(a.start))
        names = names[idx:]
    for s in names:
        path = STAGES[s]
        try:
            mod = importlib.import_module(path)
        except ModuleNotFoundError as e:
            if e.name and path.startswith(e.name):      # optional stage not present (e.g. dashboard)
                print(f"[skip] {path} not available")
                continue
            raise
        t0 = time.time()
        mod.run(cfg)
        print(f"[done] {s} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

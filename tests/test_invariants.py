"""Invariant tests on the built pipeline outputs (run after `python run_all.py`)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.common.config import load_config

CFG = load_config()
INTERIM = Path(CFG["paths"]["interim"])
OUT = Path(CFG["paths"]["out"])

pytestmark = pytest.mark.skipif(not (INTERIM / "ball_value.parquet").exists(),
                                reason="pipeline outputs not built (run python run_all.py)")


@pytest.fixture(scope="module")
def balls():
    return pd.read_parquet(INTERIM / "balls.parquet")


@pytest.fixture(scope="module")
def bv():
    b = pd.read_parquet(INTERIM / "ball_value.parquet")
    return b[~b.data_suspect]


@pytest.fixture(scope="module")
def finals():
    return {r: pd.read_csv(OUT / "final" / f"top10_{r}.csv") for r in ("batters", "bowlers")}


def test_six_legal_balls_per_completed_over(balls):
    last = balls.groupby(["match_id", "innings"]).overid.transform("max")
    comp = balls[balls.overid < last].groupby(["match_id", "innings", "overid"]).is_legal.sum()
    assert (comp == 6).mean() >= 0.99


def test_margins_reconcile():
    m = pd.read_parquet(INTERIM / "matches.parquet")
    chk = m.margin_ok.dropna()
    assert len(chk) > 500 and chk.mean() >= 0.99


def test_top10_lists_unique_and_complete(finals):
    for t in finals.values():
        assert len(t) == 10
        assert t.player_id.is_unique
        assert list(t["rank"]) == list(range(1, 11))


def test_top10_eligible_and_active(finals):
    rup = pd.read_parquet(INTERIM / "rup.parquet")
    el = CFG["eligibility"]
    for role, key in (("bat", "batters"), ("bowl", "bowlers")):
        r = rup[rup.role == role].set_index("player_id").loc[finals[key].player_id]
        spell = "innings" if role == "bat" else "spells"
        assert (r.last_match_date >= pd.Timestamp(el["active_since"])).all()
        assert (r.pressure_balls >= el[f"{role}_min_pressure_balls"]).all()
        assert (r.pressure_innings >= el[f"{role}_min_pressure_{spell}"]).all()
        assert (r.P_raw >= CFG["composite"]["top_guard_min_P_raw"]).all()
        assert r.eligible.all() and r.guard_ok.all()


def test_value_means_near_zero(bv):
    assert abs(bv.loc[bv.is_faced, "v_bat"].mean()) <= 0.02
    assert abs(bv.v_bowl.mean()) <= 0.02


def test_process_value_means_near_zero(bv):
    pv = pd.read_parquet(INTERIM / "process_value.parquet")
    b = pd.read_parquet(INTERIM / "ball_value.parquet")[["data_suspect", "is_faced"]]
    ok = ~b.data_suspect.values
    faced = ok & b.is_faced.values
    for c in ("bat_decision", "bat_difficulty", "bat_adjusted_outcome"):
        assert abs(np.nanmean(pv[c].values[faced])) <= 0.02, c
    assert abs(np.nanmean(pv.bowl_exec.values[ok])) <= 0.02


@pytest.mark.parametrize("fname", ["expected_folds.json", "process_folds.json"])
def test_out_of_fold(fname):
    """No expectation / process prediction for fold f comes from a model trained on fold f."""
    log = json.load(open(INTERIM / fname, encoding="utf-8"))
    assert log
    for model, folds in log.items():
        for f, trained_on in folds.items():
            assert int(f) not in trained_on, (model, f)


def test_wp_folds_are_by_match(bv):
    folds = pd.read_parquet(INTERIM / "folds.parquet")
    assert folds.match_id.is_unique
    chk = bv[["match_id", "fold_id"]].drop_duplicates()
    assert chk.match_id.is_unique                     # every ball of a match is in one fold
    assert (chk.merge(folds, on="match_id", suffixes=("", "_s4")).eval("fold_id == fold_id_s4")).all()


def test_pressure_share(bv):
    for inn, d in bv.groupby("innings"):
        assert abs(d.is_pressure.mean() - CFG["pressure"]["top_pct"]) < 0.01


def test_dropdown_names_match_lists(finals):
    txt = (OUT / "final" / "dropdown_check.txt").read_text(encoding="utf-8")
    for t in finals.values():
        for n in t["name"]:
            assert n in txt


def test_no_raw_ball_export_beyond_30_matches():
    tl = pd.read_csv(OUT / "dashboard" / "match_timeline.csv", low_memory=False)
    assert tl.match_id.nunique() <= 30

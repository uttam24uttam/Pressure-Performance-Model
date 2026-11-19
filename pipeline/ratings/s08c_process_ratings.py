"""Stage 8c: process ratings (what players control under pressure), same EB machinery and weights as Stage 8.

  bowlers: bowl_exec (delivery execution value);  batters: bat_decision (shot-choice value)
  proc_P_raw / proc_B_raw / proc_P_star / proc_B_star / proc_D_star  (runs above expected per 100 balls),
  REML k values (k_P from the 95% upper bound of the process pressure-gap spread, as in Stage 8).
Descriptive process stats on pressure vs normal balls:
  bowlers: yorker%, full-toss%, wide+NB%, slower-ball% (pace_off), bouncer%
  batters: aerial%, defence%, dismissals per aerial shot, mean difficulty faced (bat_difficulty x 100)
Bootstrap: the Stage 9 match resamples (same seed, same matches) for proc P*/B*/D*.

Outputs: interim/process_ratings.parquet, interim/bootstrap_process.npz, tables/process_k_values.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.value.s07_ball_value import display_names
from pipeline.ratings.s08_player_ratings import ball_weights, compute_ratings, ROLE_COLS
from pipeline.ratings.s09_consistency import bootstrap

PROC_COL = {"bat": "bat_decision", "bowl": "bowl_exec"}


def load_balls(cfg) -> pd.DataFrame:
    """ball_value + process values, non-suspect balls (row-aligned before filtering)."""
    b = pd.read_parquet(interim(cfg, "ball_value"))
    pv = pd.read_parquet(interim(cfg, "process_value"))
    assert (pv.match_id.values == b.match_id.values).all()
    b = pd.concat([b, pv.drop(columns=["match_id", "innings", "innings_ball_num"])], axis=1)
    return b[~b.data_suspect].reset_index(drop=True)


def process_ratings(b, cfg, role, w, vcol=None, k_override=None):
    r, k = compute_ratings(b, cfg, role, w, vcol=vcol or PROC_COL[role], k_override=k_override)
    keep = ["player_id", "P_raw", "B_raw", "P_star", "B_star", "D_star", "n_P", "n_B", "shrink_P"]
    return r[keep].rename(columns={c: f"proc_{c}" for c in keep if c != "player_id"}), k


def descriptives(b, role) -> pd.DataFrame:
    if role == "bowl":
        g = b.groupby(["bowler_id", "is_pressure"])
        d = pd.DataFrame({"yorker_pct": g.is_yorker.mean(), "full_toss_pct": g.is_full_toss.mean(),
                          "wide_nb_pct": g.is_wide_nb.mean(), "slower_pct": g.is_pace_off.mean(),
                          "bouncer_pct": g.is_bouncer.mean()}) * 100
    else:
        f = b[b.is_faced]
        g = f.groupby(["striker_id", "is_pressure"])
        d = pd.DataFrame({"aerial_pct": 100 * g.is_aerial.mean(), "defence_pct": 100 * g.is_defence.mean(),
                          "outs_per_aerial": g.apply(lambda x: x.batter_out[x.is_aerial].sum() / max(x.is_aerial.sum(), 1),
                                                     include_groups=False),
                          "difficulty_faced": 100 * g.bat_difficulty.mean(),
                          "decision_value": 100 * g.bat_decision.mean()})
    d = d.unstack("is_pressure")
    d.columns = [f"{c}_{'pressure' if p else 'normal'}" for c, p in d.columns]
    return d.rename_axis("player_id").reset_index()


def run(cfg: dict) -> None:
    header("STAGE 8c - PROCESS RATINGS")
    b = load_balls(cfg)
    out = Path(cfg["paths"]["out"])
    R = []
    w = ball_weights(b, cfg)
    names = display_names(b)
    outcome = pd.read_parquet(interim(cfg, "player_ratings"))
    prior = json.load(open(out / "tables" / "consistency_prior.json", encoding="utf-8"))
    el = cfg["eligibility"]
    allr, allk, boot = [], {}, {}
    for role in ("bat", "bowl"):
        header(f"{'Batting: bat_decision' if role == 'bat' else 'Bowling: bowl_exec'} (process track)")
        r, k = process_ratings(b, cfg, role, w)
        allk[role] = k
        sd = lambda t: 100 * np.sqrt(t)
        print(f"  sigma^2 per ball: pressure {k['sigma2_P']:.4f}, normal {k['sigma2_B']:.4f}")
        print(f"  [REML] true baseline spread sd(B) = {sd(k['tau2_B']):.2f} per 100 balls "
              f"(95% CI {sd(k['tau2_B_ci'][0]):.2f}-{sd(k['tau2_B_ci'][1]):.2f})")
        print(f"  [REML] true pressure-gap spread sd(P-B) = {sd(k['tau2_D']):.2f} per 100 balls "
              f"(95% CI {sd(k['tau2_D_ci'][0]):.2f}-{sd(k['tau2_D_ci'][1]):.2f}) -> k_P from the upper bound")
        print(f"  k_B = {k['k_B']:.1f}, k_P = {k['k_P']:.1f} effective balls")
        check(np.isfinite([k["k_B"], k["k_P"]]).all() and k["k_B"] > 0 and k["k_P"] > 0, f"{role}: finite positive process k", R)
        r = r.merge(descriptives(b, role), on="player_id", how="left")
        r["role"] = role
        o = outcome[outcome.role == role][["player_id", "P_star", "B_star", "pressure_balls", "pressure_innings",
                                          "last_match_date"]]
        q = r.merge(o, on="player_id")
        mi = el[f"{role}_min_pressure_" + ("innings" if role == "bat" else "spells")]
        q = q[(q.pressure_balls >= el[f"{role}_min_pressure_balls"]) & (q.pressure_innings >= mi) &
              (q.last_match_date >= pd.Timestamp(el["active_since"]))]
        print(f"  provisional pool {len(q)}: sd proc P* {q.proc_P_star.std():.2f}, sd outcome P* {q.P_star.std():.2f}; "
              f"Spearman(proc P*, outcome P*) {q[['proc_P_star', 'P_star']].corr('spearman').iloc[0, 1]:.2f}; "
              f"Spearman(proc B*, outcome B*) {q[['proc_B_star', 'B_star']].corr('spearman').iloc[0, 1]:.2f}")
        q["name"] = q.player_id.map(names)
        dcols = (["yorker_pct_pressure", "full_toss_pct_pressure", "wide_nb_pct_pressure", "slower_pct_pressure"]
                 if role == "bowl" else ["aerial_pct_pressure", "outs_per_aerial_pressure", "difficulty_faced_pressure"])
        print("  top 15 by process P*:")
        print(q.nlargest(15, "proc_P_star")[["name", "pressure_balls", "proc_P_raw", "proc_P_star", "proc_B_star",
                                             "proc_D_star", "P_star"] + dcols].round(2).to_string(index=False))
        pool_desc = q[[c for c in r.columns if c.endswith("_pressure") or c.endswith("_normal")]].median()
        print("  pool medians (pressure vs normal): " + ", ".join(f"{c} {v:.2f}" for c, v in pool_desc.items()))

        print(f"  bootstrap: {cfg['validation']['bootstrap_reps']} reps on the Stage 9 resamples ...", flush=True)
        mb = el[f"match_min_pressure_balls_{role}"]
        pidx, point, arr = bootstrap(b, role, w, k, (prior[role]["a"], prior[role]["b"]), 0.0, mb,
                                     cfg["validation"]["bootstrap_reps"], cfg["seed"] + (0 if role == "bat" else 1),
                                     vcol=PROC_COL[role])
        rr = r.set_index("player_id").reindex(pidx)
        check(np.allclose(point[0], rr.proc_P_star.values, atol=1e-6), f"{role}: process bootstrap reproduces proc P*", R)
        boot[role] = (pidx, arr)
        allr.append(r)

    pr = pd.concat(allr, ignore_index=True)
    pr.to_parquet(interim(cfg, "process_ratings"), index=False)
    np.savez_compressed(Path(cfg["paths"]["interim"]) / "bootstrap_process.npz",
                        **{f"{role}_proc_{n}": v for role, (pidx, arr) in boot.items() for n, v in arr.items() if n != "C"},
                        **{f"{role}_player_id": np.asarray(pidx) for role, (pidx, _) in boot.items()})
    with open(out / "tables" / "process_k_values.json", "w", encoding="utf-8") as fh:
        json.dump(allk, fh, indent=2)
    print("\n  wrote process_ratings.parquet, bootstrap_process.npz, tables/process_k_values.json")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 8c: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 8c checks failed - review.")

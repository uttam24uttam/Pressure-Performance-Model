"""Stage 10: final rating - Reliability Under Pressure (RUP).

Process and outcome values are both runs above expected per 100 balls.
  R_P = a Process_P* + (1-a) Outcome_P*      R_B = a Process_B* + (1-a) Outcome_B*      D = R_P - R_B
  RUP = wP z(R_P) + wD z(D) + wC z(C), z-scored within the eligible pool.
Eligible: last match >= 2025-01-01 AND min pressure balls AND min pressure innings/spells.
Top-list guard: ranked only if outcome P_raw >= 0 (actual raw pressure outcome at or above expectation).

a and the weights by the temporal test: Stages 8/8c/9 rebuilt on balls <= 2025-12-31 (recency reference
2025-12-31, REML k and Beta prior re-fitted); target = IPL 2026 raw pressure outcome (outcome P_raw on IPL 2026
pressure balls, unweighted) for players with >= 40 IPL 2026 pressure balls and >= 30 train pressure balls;
Spearman; player bootstrap CIs. a in {0, 0.1, ..., 1} (argmax). Weights: simplex step 0.05, each >= 0.15,
bowler C fixed at 0.15; defaults kept unless the bootstrap CI of (rho_best - rho_default) excludes 0.

Outputs: interim/rup.parquet, interim/temporal.parquet, tables/a_search_{role}.csv, weight_search_{role}.csv,
         rup_weights.json, guard_removed.csv, eligibility_removals.csv
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.value.s07_ball_value import display_names
from pipeline.ratings.s08_player_ratings import ball_weights, compute_ratings
from pipeline.ratings.s08c_process_ratings import load_balls
from pipeline.ratings.scoring import (build_role, compose, eligible, score, zscore, rup_from_z, active_map, player_info, weight_grid,
                  default_weights, SPELL)

A_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]


def temporal_frames(b: pd.DataFrame, cfg: dict, role: str, train_until: str | None = None,
                    test_prefixes: tuple | None = None, out_vcol=None, proc_vcol=None):
    """Train ratings (<= train_until) + test-season raw pressure outcome."""
    vc = cfg["validation"]
    train_until = train_until or vc["train_until"]
    test_prefixes = test_prefixes or (vc["test_comp_prefix"],)
    cut = pd.Timestamp(train_until)
    ctr = copy.deepcopy(cfg)
    ctr["reference_date"] = train_until
    btr = b[pd.to_datetime(b.match_date) <= cut].reset_index(drop=True)
    tr, ktr = build_role(btr, ctr, role, ball_weights(btr, ctr), out_vcol=out_vcol, proc_vcol=proc_vcol)
    bte = b[b.comp_name.str.startswith(tuple(test_prefixes)) & (pd.to_datetime(b.match_date) > cut)].reset_index(drop=True)
    te, _ = compute_ratings(bte, cfg, role, np.ones(len(bte)), k_override=ktr["out"])
    te = te[["player_id", "P_raw", "P_star", "pressure_balls"]].rename(
        columns={"P_raw": "P_raw_test", "P_star": "P_star_test", "pressure_balls": "pressure_balls_test"})
    t = tr.merge(te, on="player_id", how="left")
    t["train_pool"] = eligible(t, cfg, role)
    t["test_ok"] = t.pressure_balls_test >= cfg["composite"]["test_min_pressure_balls"]
    t["train_ok"] = t.pressure_balls >= cfg["composite"]["train_min_pressure_balls"]
    t["role"] = role
    return t, ktr, btr, bte


def _boot_idx(n, reps, rng):
    return [rng.integers(0, n, n) for _ in range(reps)]


def _rho(x, y):
    return spearmanr(x, y)[0]


def run(cfg: dict) -> None:
    header("STAGE 10 - FINAL RATING: RELIABILITY UNDER PRESSURE (RUP)")
    b = load_balls(cfg)
    ratings = pd.read_parquet(interim(cfg, "player_ratings"))
    proc = pd.read_parquet(interim(cfg, "process_ratings"))
    cons = pd.read_parquet(interim(cfg, "consistency"))
    out = Path(cfg["paths"]["out"])
    R = []
    rng = np.random.default_rng(cfg["seed"])
    act = active_map(cfg)
    info = player_info(cfg)
    names = display_names(b)
    guard = cfg["composite"]["top_guard_min_P_raw"]
    print("  availability, nationality, Pakistan flag and RR membership are informational columns only.")

    chosen, wlog, temporal, allr, removed, elrem = {}, {}, [], [], [], []
    for role in ("bat", "bowl"):
        lab = "Batting" if role == "bat" else "Bowling"
        header(f"{lab}: temporal test (train <= {cfg['validation']['train_until']}, target = "
               f"{cfg['validation']['test_comp_prefix']} raw pressure outcome)")
        t, ktr, btr, bte = temporal_frames(b, cfg, role)
        S0 = t[t.train_ok & t.test_ok]
        print(f"  train: {len(btr):,} balls; outcome k_B {ktr['out']['k_B']:.0f} / k_P {ktr['out']['k_P']:.0f}; process "
              f"k_B {ktr['proc']['k_B']:.0f} / k_P {ktr['proc']['k_P']:.0f}; test: {len(bte):,} balls")
        print(f"  analysis set: {len(S0)} players (>= {cfg['composite']['test_min_pressure_balls']} test and >= "
              f"{cfg['composite']['train_min_pressure_balls']} train pressure balls); Spearman 95% CI half-width "
              f"~{1.96 / np.sqrt(max(len(S0) - 3, 1)):.2f}")
        check(len(S0) >= 15, f"{role}: analysis set >= 15 players ({len(S0)})", R)
        y = S0.P_raw_test.values
        idx = _boot_idx(len(S0), 2000, rng)
        # ---- a ----
        arow = []
        base0 = compose(S0, 0.0).R_P.values
        for a in A_GRID:
            x = compose(S0, a).R_P.values
            bs = np.array([_rho(x[i], y[i]) for i in idx])
            d0 = np.array([_rho(x[i], y[i]) - _rho(base0[i], y[i]) for i in idx])
            arow.append({"a": a, "rho": _rho(x, y), "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5),
                         "boot_mean": np.nanmean(bs), "vs_outcome_only_lo": np.nanpercentile(d0, 2.5),
                         "vs_outcome_only_hi": np.nanpercentile(d0, 97.5)})
        arow = pd.DataFrame(arow)
        a_best = float(arow.loc[arow.rho.idxmax(), "a"])
        print("  blend weight a (0 = outcome only, 1 = process only): Spearman(train R_P, IPL 2026 raw pressure outcome)")
        print(arow.round(3).to_string(index=False))
        print(f"  -> chosen a = {a_best} (argmax of the point Spearman)")
        arow.to_csv(out / "tables" / f"a_search_{role}.csv", index=False)
        # ---- weights ----
        t = compose(t, a_best)
        t[["z_R_P", "z_D_R", "z_C"]] = zscore(t, t.train_pool)
        S = t[t.train_ok & t.test_ok]
        grid = weight_grid(role, cfg)
        res = pd.DataFrame([{**w_, "rho": _rho(rup_from_z(S, w_), y)} for w_ in grid]).sort_values("rho", ascending=False)
        dw = default_weights(role, cfg)
        best = res.iloc[0][["P", "D", "C"]].to_dict()
        rb, rd = rup_from_z(S, best).values, rup_from_z(S, dw).values
        diffs = np.array([_rho(rb[i], y[i]) - _rho(rd[i], y[i]) for i in idx])
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        use_best = cfg["composite"]["weight_search"] and lo > 0
        wts = best if use_best else dw
        rho_def = _rho(rd, y)
        comp_rho = {c: float(_rho(S[c].values, y)) for c in ("R_P", "D_R", "C")}
        print(f"  weight grid: {len(grid)} vectors (step 0.05, each >= {cfg['composite']['min_weight']}"
              f"{', C fixed 0.15' if role == 'bowl' else ''}); components alone: "
              + ", ".join(f"{k_} {v:+.3f}" for k_, v in comp_rho.items()))
        print(pd.concat([res.head(5), res.tail(3)]).round(3).to_string(index=False))
        print(f"  best {best} rho {res.rho.iloc[0]:.3f} vs default {dw} rho {rho_def:.3f}; bootstrap 95% CI of "
              f"difference [{lo:+.3f}, {hi:+.3f}] -> USING {'BEST' if use_best else 'DEFAULT'} {wts}")
        res.to_csv(out / "tables" / f"weight_search_{role}.csv", index=False)
        chosen[role] = {"a": a_best, "weights": {k_: float(v) for k_, v in wts.items()}}
        wlog[role] = {"analysis_n": int(len(S)), "a_search": arow.to_dict("records"), "a": a_best,
                      "best_weights": best, "rho_best": float(res.rho.iloc[0]), "default_weights": dw,
                      "rho_default": float(rho_def), "diff_ci95": [float(lo), float(hi)],
                      "weights_used": "best" if use_best else "default", "rho_components": comp_rho,
                      "train_k": {"out": {k_: ktr["out"][k_] for k_ in ("k_B", "k_P")},
                                  "proc": {k_: ktr["proc"][k_] for k_ in ("k_B", "k_P")}}}
        temporal.append(t)

        # ---------------- full data ----------------
        header(f"{lab}: full-data RUP (a = {a_best}, weights {wts})")
        r = ratings[ratings.role == role].merge(
            proc[proc.role == role].drop(columns=["role"]), on="player_id", how="left").merge(
            cons[cons.role == role][["player_id", "C", "units", "failure_rate", "se_C", "prior_a", "prior_b"]],
            on="player_id", how="left")
        pa, pb = cons[cons.role == role].prior_a.iloc[0], cons[cons.role == role].prior_b.iloc[0]
        r["C"] = r.C.fillna(pa / (pa + pb))
        r = compose(r, a_best)
        el = cfg["eligibility"]
        mb, mi = el[f"{role}_min_pressure_balls"], el[f"{role}_min_pressure_{SPELL[role]}"]
        has_p = r.pressure_balls > 0
        is_act = r.player_id.isin(act[act].index)
        okb, oki = r.pressure_balls >= mb, r.pressure_innings >= mi
        pool = is_act & okb & oki
        r = score(r, pool, wts, guard)
        r = r.merge(info, on="player_id", how="left")
        r["name"] = r.player_id.map(names).fillna(r.name)
        steps = [("players with any pressure ball", has_p),
                 (f"last match >= {el['active_since']}", has_p & is_act),
                 (f">= {mb} pressure balls", has_p & is_act & okb),
                 (f">= {mi} pressure {SPELL[role]}", pool),
                 (f"guard P_raw >= {guard} (ranked)", pool & r.guard_ok)]
        print("  eligibility funnel:")
        prev = None
        for lbl, m_ in steps:
            print(f"    {lbl:34s} {int(m_.sum()):5d}" + (f"  (removes {int((prev & ~m_).sum())})" if prev is not None else ""))
            if prev is not None:
                gone = r[prev & ~m_].sort_values("pressure_balls", ascending=False)
                elrem.append(gone.assign(rule=lbl, role=role)[["role", "rule", "player_id", "name", "pressure_balls",
                                                               "pressure_innings", "last_match_date", "P_raw", "RUP"]])
            prev = m_
        inact = r[has_p & ~is_act].nlargest(8, "pressure_balls")
        print("    largest pressure records removed by the activity rule: " +
              ", ".join(f"{n_} ({int(p_)}, last {str(d_)[:10]})" for n_, p_, d_ in
                        zip(inact.name, inact.pressure_balls, inact.last_match_date)))
        g = r[r.eligible & ~r.guard_ok].sort_values("rank_unguarded")
        print(f"    guard removes {len(g)} eligible players: " + ", ".join(
            f"{n_} (#{int(k_)}, P_raw {p_:+.1f})" for n_, k_, p_ in zip(g.name, g.rank_unguarded, g.P_raw)))
        print(f"    of whom would otherwise be top 20: {', '.join(g[g.rank_unguarded <= 20].name) or 'none'}")
        removed.append(g.assign(role=role)[["role", "player_id", "name", "rank_unguarded", "RUP", "P_raw", "P_star",
                                            "proc_P_star", "R_P", "D_R", "C", "pressure_balls"]])
        check(60 <= pool.sum() <= 150, f"{role}: eligible count {int(pool.sum())} within 60-150", R)
        print(f"  pool sds: R_P {r.loc[pool, 'R_P'].std():.2f}, D {r.loc[pool, 'D_R'].std():.2f}, C {r.loc[pool, 'C'].std():.4f}; "
              f"Spearman(RUP, outcome P*) {r.loc[pool, ['RUP', 'P_star']].corr('spearman').iloc[0, 1]:.2f}, "
              f"(RUP, process P*) {r.loc[pool, ['RUP', 'proc_P_star']].corr('spearman').iloc[0, 1]:.2f}")
        header(f"{lab}: top 20 by RUP")
        cols = ["rank", "name", "RUP", "R_P", "P_star", "proc_P_star", "B_star", "D_R", "C", "P_raw", "pressure_balls",
                "pressure_innings"]
        top = r[r["rank"] <= 20].sort_values("rank")
        print(top[cols].round(2).to_string(index=False))
        check(top.player_id.is_unique and len(top) == 20, f"{role}: 20 unique ranked players", R)
        check((top.P_raw >= guard).all() and top.eligible.all() and (top.last_match_date >= pd.Timestamp(el["active_since"])).all(),
              f"{role}: every ranked player eligible, active and P_raw >= 0", R)
        r["a"] = a_best
        allr.append(r)

    rup = pd.concat(allr, ignore_index=True)
    rup.to_parquet(interim(cfg, "rup"), index=False)
    pd.concat(temporal, ignore_index=True).to_parquet(interim(cfg, "temporal"), index=False)
    pd.concat(removed, ignore_index=True).to_csv(out / "tables" / "guard_removed.csv", index=False)
    pd.concat(elrem, ignore_index=True).to_csv(out / "tables" / "eligibility_removals.csv", index=False)
    with open(out / "tables" / "rup_weights.json", "w", encoding="utf-8") as f:
        json.dump({"chosen": chosen, "search": wlog}, f, indent=2, default=float)
    print("\n  wrote rup.parquet, temporal.parquet, a_search_*.csv, weight_search_*.csv, rup_weights.json, "
          "guard_removed.csv, eligibility_removals.csv")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 10: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 10 checks failed - review.")

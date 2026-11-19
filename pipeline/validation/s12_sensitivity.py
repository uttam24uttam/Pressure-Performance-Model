"""Stage 12: sensitivity variants and consensus top 10.

Each variant re-runs pressure flags (Stage 5 normalisation via apply_pressure, cached LI_raw), ratings (outcome +
process, REML k re-estimated), consistency, eligibility, RUP, guard and rank. Cached WP / expectation / process
models are reused. Eligibility minimums scale with top_pct / 0.20 (pressure-ball counts scale with the share).
One-at-a-time from the default, plus random combinations (fixed seed).
Consensus: median rank across variants (unranked = that variant's ranked-pool size + 1); final top 10 = lowest
median rank among default-eligible players passing the guard, tie-break P(top 10), then default RUP.
Fragile = default top 10 but outside the top 20 in > 30% of variants.

Outputs: interim/consensus.parquet, tables/sensitivity_ranks.csv, sensitivity_variants.csv,
         consensus_{bat,bowl}.csv; figures/sensitivity_heatmap_{bat,bowl}.png
"""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.pressure.s05_leverage import apply_pressure
from pipeline.ratings.s08_player_ratings import ball_weights
from pipeline.ratings.s08c_process_ratings import load_balls
from pipeline.ratings.scoring import build_role, compose, eligible, score, active_map, default_weights, PROC_COL
from pipeline.common import plotstyle as ps

BASE = {"top_pct": 0.20, "scope": "per_innings", "stakes": False, "half_life": 730, "data": "all",
        "a": "chosen", "weights": "chosen", "labels": "family", "bat_outcome": "v_bat", "bowl_exec": "type"}
OAT = [("top_pct_10", {"top_pct": 0.10}), ("top_pct_30", {"top_pct": 0.30}), ("scope_global", {"scope": "global"}),
       ("stakes_on", {"stakes": True}), ("half_life_365", {"half_life": 365}), ("half_life_inf", {"half_life": np.inf}),
       ("ipl_only", {"data": "IPL"}), ("default_weights", {"weights": "default"}), ("process_only_a1", {"a": 1.0}),
       ("outcome_only_a0", {"a": 0.0}), ("fine_labels", {"labels": "fine"}),
       ("bat_adjusted_outcome", {"bat_outcome": "adjusted"}), ("bowl_exec_literal", {"bowl_exec": "literal"})]
GRID = {"top_pct": [0.10, 0.20, 0.30], "scope": ["per_innings", "global"], "stakes": [False, True],
        "half_life": [365, 730, np.inf], "data": ["all", "IPL"], "a": ["chosen", 0.0, 1.0], "weights": ["chosen", "default"],
        "labels": ["family", "fine"], "bat_outcome": ["v_bat", "adjusted"], "bowl_exec": ["type", "literal"]}


def variant_list(n_random: int, seed: int):
    out = [("default", dict(BASE))] + [(n, {**BASE, **d}) for n, d in OAT]
    seen = {json.dumps(v, sort_keys=True, default=str) for _, v in out}
    rng = np.random.default_rng(seed)
    k = 0
    while k < n_random:
        v = {key: vals[rng.integers(len(vals))] for key, vals in GRID.items()}
        js = json.dumps(v, sort_keys=True, default=str)
        if js in seen or sum(v[key] != BASE[key] for key in BASE) < 2:
            continue
        seen.add(js)
        k += 1
        out.append((f"random_{k:02d}", v))
    return out


def run_variant(b0, cfg, v, chosen, act):
    pcfg = copy.deepcopy(cfg["pressure"])
    pcfg["top_pct"] = v["top_pct"]
    pcfg["percentile_scope"] = v["scope"]
    pcfg["stakes"]["enabled"] = bool(v["stakes"])
    b = b0
    if (v["top_pct"], v["scope"], v["stakes"]) != (BASE["top_pct"], BASE["scope"], BASE["stakes"]):
        b = apply_pressure(b0, pcfg)
    if v["data"] == "IPL":
        b = b[b.comp_group == "IPL"].reset_index(drop=True)
    w = ball_weights(b, cfg, half_life=v["half_life"])
    scale = v["top_pct"] / BASE["top_pct"]
    res = {}
    for role in ("bat", "bowl"):
        out_vcol = "bat_adjusted_outcome" if (role == "bat" and v["bat_outcome"] == "adjusted") else None
        pv = PROC_COL[role]
        if role == "bowl" and v["bowl_exec"] == "literal":
            pv = "bowl_exec_literal"
        if v["labels"] == "fine":
            pv = pv + "_fine"
        r, _ = build_role(b, cfg, role, w, out_vcol=out_vcol, proc_vcol=pv)
        a = chosen[role]["a"] if v["a"] == "chosen" else float(v["a"])
        wts = chosen[role]["weights"] if v["weights"] == "chosen" else default_weights(role, cfg)
        r = compose(r, a)
        pool = eligible(r, cfg, role, scale=scale, active=act)
        r = score(r, pool, wts, cfg["composite"]["top_guard_min_P_raw"])
        res[role] = r.set_index("player_id")[["rank", "RUP", "eligible", "guard_ok"]]
    return res


def heatmap(ranks: pd.DataFrame, names: pd.Series, order: list, path: Path, title: str):
    ps.apply()
    M = ranks.loc[order]
    fig, ax = ps.plt.subplots(figsize=(min(2 + 0.42 * M.shape[1], 18), 0.42 * len(order) + 2.2))
    cap = 40
    data = np.where(M.isna(), np.nan, np.minimum(M.values, cap))
    cmap = ps.plt.get_cmap(ps.SEQ_CMAP + "_r").copy()
    cmap.set_bad("#F1F2F4")
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=1, vmax=cap, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M.values[i, j]
            txt = "–" if np.isnan(val) else str(int(val))
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if (not np.isnan(val) and val <= 12) else ps.INK)
    ax.set_yticks(range(len(order)), [names.get(p, p) for p in order], fontsize=8)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(f"rank (capped at {cap}); – = not eligible / not ranked", fontsize=8)
    ax.set_title(title)
    fig.savefig(path)
    ps.plt.close(fig)


def run(cfg: dict) -> None:
    header("STAGE 12 - SENSITIVITY AND CONSENSUS")
    out = Path(cfg["paths"]["out"])
    R = []
    b0 = load_balls(cfg)
    rup = pd.read_parquet(interim(cfg, "rup"))
    rs = pd.read_parquet(interim(cfg, "rank_stability"))
    chosen = json.load(open(out / "tables" / "rup_weights.json", encoding="utf-8"))["chosen"]
    act = active_map(cfg)
    variants = variant_list(15, cfg["seed"])
    pd.DataFrame([{"variant": n, **{k_: str(v_) for k_, v_ in v.items()}} for n, v in variants]).to_csv(
        out / "tables" / "sensitivity_variants.csv", index=False)
    print(f"  {len(variants)} variants: default + {len(OAT)} one-at-a-time + {len(variants) - 1 - len(OAT)} random")
    results = {}
    for i, (name, v) in enumerate(variants):
        diff = {k_: v_ for k_, v_ in v.items() if v_ != BASE[k_]}
        res = run_variant(b0, cfg, v, chosen, act)
        results[name] = res
        print(f"  [{i + 1:2d}/{len(variants)}] {name:22s} eligible bat {int(res['bat'].eligible.sum()):3d} / bowl "
              f"{int(res['bowl'].eligible.sum()):3d}  {diff if diff else ''}", flush=True)

    long, cons_all = [], []
    for role in ("bat", "bowl"):
        lab = "Batters" if role == "bat" else "Bowlers"
        d0 = rup[rup.role == role].set_index("player_id")
        dflt = results["default"][role]
        same = (dflt["rank"].reindex(d0.index).fillna(-1) == d0["rank"].fillna(-1)).all()
        check(same, f"{role}: default variant reproduces the Stage 10 ranking exactly", R)
        ranks = pd.DataFrame({n: res[role]["rank"] for n, res in results.items()})
        for n in ranks:
            long.append(pd.DataFrame({"role": role, "variant": n, "player_id": ranks.index, "rank": ranks[n].values}))
        cand = d0.index[d0.eligible & d0.guard_ok]
        filled = ranks.copy()
        for n in filled:
            filled[n] = filled[n].fillna(results[n][role]["rank"].max() + 1)
        c = pd.DataFrame({"median_rank_variants": filled.loc[cand].median(axis=1),
                          "share_top10": (filled.loc[cand] <= 10).mean(axis=1),
                          "share_outside_top20": (filled.loc[cand] > 20).mean(axis=1),
                          "best_rank": filled.loc[cand].min(axis=1), "worst_rank": filled.loc[cand].max(axis=1)})
        c["default_rank"] = d0.loc[cand, "rank"]
        c["default_RUP"] = d0.loc[cand, "RUP"]
        c["p_top10"] = rs[rs.role == role].set_index("player_id").p_top10.reindex(cand)
        c = c.sort_values(["median_rank_variants", "p_top10", "default_RUP"], ascending=[True, False, False])
        c["consensus_rank"] = np.arange(1, len(c) + 1)
        c["fragile"] = (c.default_rank <= 10) & (c.share_outside_top20 > 0.30)
        c["name"] = d0.loc[c.index, "name"]
        c["role"] = role
        cons_all.append(c.reset_index())
        header(f"{lab}: consensus top 20 (median rank over {ranks.shape[1]} variants)")
        print(c.head(20)[["consensus_rank", "name", "median_rank_variants", "p_top10", "default_rank", "share_top10",
                          "share_outside_top20", "best_rank", "worst_rank", "fragile"]].round(2).to_string(index=False))
        top10 = c.head(10)
        dtop = set(d0.index[d0["rank"] <= 10])
        print(f"  consensus top 10 vs default top 10: {len(set(top10.index) & dtop)} in common; "
              f"in: {', '.join(d0.loc[list(set(top10.index) - dtop), 'name']) or 'none'}; "
              f"out: {', '.join(d0.loc[list(dtop - set(top10.index)), 'name']) or 'none'}")
        fr = c[c.fragile]
        print(f"  fragile picks (default top 10, outside top 20 in > 30% of variants): "
              f"{', '.join(f'{n_} ({s_:.0%})' for n_, s_ in zip(fr.name, fr.share_outside_top20)) or 'none'}")
        check(top10.index.is_unique and len(top10) == 10, f"{role}: consensus top 10 has 10 unique players", R)
        check(bool(d0.loc[top10.index, "eligible"].all() and d0.loc[top10.index, "guard_ok"].all()),
              f"{role}: every consensus top-10 player is eligible (active, minimums) and passes the guard", R)
        order = list(c.head(20).index)
        heatmap(ranks, d0.name, order, out / "figures" / f"sensitivity_heatmap_{role}.png",
                f"{lab}: rank in each sensitivity variant (consensus top 20, ordered)")
        c.to_csv(out / "tables" / f"consensus_{role}.csv", index=False)
    pd.concat(long, ignore_index=True).to_csv(out / "tables" / "sensitivity_ranks.csv", index=False)
    pd.concat(cons_all, ignore_index=True).to_parquet(interim(cfg, "consensus"), index=False)
    print("\n  wrote consensus.parquet, sensitivity_ranks.csv, sensitivity_variants.csv, consensus_*.csv, "
          "sensitivity_heatmap_*.png")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 12: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 12 checks failed - review.")

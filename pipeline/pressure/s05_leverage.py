"""Stage 5: Leverage Index (LI), Pressure Index (PI), pressure-ball flags, descriptive flags.

LI_raw = sum_o p(o | bucket) * |WP(s_o) - WP(s)|, evaluated with the same fold's WP model.
Outputs: interim/leverage.parquet
         tables: outcome_probs.csv, li_continuity.json, li_concentration.csv,
                 li_within_bucket_spread.csv, li_outcome_rates.csv, li_top_balls.csv
         figures: li_heatmap_inn{1,2}.png, pressure_share_by_phase.png, example_match_*.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.pressure.s04_wp_models import wp, load_models
from pipeline.pressure.outcomes import OUTCOMES, BUCKET_KEYS, add_buckets, classify_outcome, outcome_probs
from pipeline.common import plotstyle as ps


# --------------------------------------------------------------------------- leverage
def leverage_raw(b: pd.DataFrame, models: dict, probs: pd.DataFrame) -> np.ndarray:
    P = b[BUCKET_KEYS].merge(probs, on=BUCKET_KEYS, how="left")
    assert P[list(OUTCOMES)].notna().all().all(), "missing outcome probabilities for some buckets"
    base = b.wp_before.values
    li = np.zeros(len(b))
    for o, (r, w, legal) in OUTCOMES.items():
        nxt = b.assign(score_before=b.score_before + r, wkts_before=b.wkts_before + w,
                       balls_left=b.balls_left - legal)
        li += P[o].values * np.abs(wp(nxt, models) - base)
    return li


def apply_pressure(b: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    """LI normalisation, LI_capped, PI, and top-x% flags. Reused by Stage 12 for variants.

    Flags are rank-based on uncapped PI. LI_capped / PI_capped are for any use of LI as a weight.
    """
    b = b.copy()
    scope = b.innings if pcfg["percentile_scope"] == "per_innings" else pd.Series(0, index=b.index)
    b["LI"] = b.LI_raw / b.LI_raw.groupby(scope).transform("mean")
    cap = b.LI.groupby(scope).transform(lambda s: s.quantile(pcfg.get("li_cap_quantile", 0.995)))
    b["LI_capped"] = np.minimum(b.LI, cap)
    st = pcfg["stakes"]
    stakes = np.where(b.is_final, st["final"], np.where(b.is_knockout, st["knockout"], 1.0))
    use_stakes = st.get("enabled", False) or pcfg.get("method") == "leverage_x_stakes"
    mult = stakes if use_stakes else 1.0
    b["PI"] = b.LI * mult
    b["PI_capped"] = b.LI_capped * mult
    pct = b.PI.groupby(scope).rank(pct=True, method="average")
    for q in (10, 20, 30):
        b[f"pressure_top{q}"] = pct > 1 - q / 100
    b["is_pressure"] = pct > 1 - pcfg["top_pct"]
    return b


def add_wp_after(b: pd.DataFrame, models: dict) -> pd.DataFrame:
    """WP after the ball (batting side) = WP of the actual post-ball state, from the same wp().

    Terminal post states get exact values (chase won 1 / lost 0 / tie 0.5; innings-1 end ->
    1 - chase-start value). Equals the next ball's wp_before within an innings.
    """
    post = b.assign(score_before=b.score_before + b.team_runs,
                    wkts_before=b.wkts_before + b.is_wicket.astype(int),
                    balls_left=b.balls_left - b.is_legal.astype(int))
    b = b.copy()
    b["wp_after"] = wp(post, models)
    b["dwp"] = b.wp_after - b.wp_before
    return b


# --------------------------------------------------------------------------- figures
def _heatmap(b, inn, path, vmax):
    import matplotlib.pyplot as plt
    d = b[b.innings == inn]
    g = d.groupby(["wkts_before", "over_num"]).LI.agg(["mean", "size"])
    mean = g["mean"].unstack().reindex(index=range(10), columns=range(1, 21))
    n = g["size"].unstack().reindex(index=range(10), columns=range(1, 21))
    mean = mean.where(n >= 30)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.grid(False)
    im = ax.imshow(mean.values, aspect="auto", cmap=ps.SEQ_CMAP, vmin=0, vmax=vmax, origin="lower")
    ax.set_xticks(range(20)); ax.set_xticklabels(range(1, 21))
    ax.set_yticks(range(10)); ax.set_yticklabels(range(10))
    ax.set_xlabel("Over"); ax.set_ylabel("Wickets lost before the ball")
    title = "first innings" if inn == 1 else "chase"
    ax.set_title(f"Mean Leverage Index by over and wickets lost: {title}\n"
                 f"(1.0 = average ball of this innings; blank = fewer than 30 balls)", fontsize=11)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Mean LI")
    for (i, j), v in np.ndenumerate(mean.values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if v >= vmax * 0.55 else ps.INK)
    fig.savefig(path); plt.close(fig)


def _phase_share(b, path, phases):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    order = list(phases)
    x = np.arange(len(order))
    for ax, inn in zip(axes, (1, 2)):
        d = b[b.innings == inn]
        allb = d.phase.value_counts(normalize=True).reindex(order)
        pres = d[d.is_pressure].phase.value_counts(normalize=True).reindex(order)
        w = 0.38
        ax.bar(x - w / 2 - 0.01, allb.values * 100, w, color=ps.BLUE, label="All balls")
        ax.bar(x + w / 2 + 0.01, pres.values * 100, w, color=ps.PINK, label="Pressure balls (20% highest Leverage Index)")
        for xi, v in zip(x - w / 2 - 0.01, allb.values):
            ax.text(xi, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=8.5, color=ps.MUTED)
        for xi, v in zip(x + w / 2 + 0.01, pres.values):
            ax.text(xi, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9, color=ps.INK, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in order])
        ax.set_title("First innings" if inn == 1 else "Chase")
        ax.set_ylabel("Share of balls (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Where pressure balls occur, by phase", fontweight="bold")
    fig.savefig(path); plt.close(fig)


def _example_match(b, m, mid, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    d = b[b.match_id == mid].sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
    mm = m.set_index("match_id").loc[mid]
    bf = mm.bat_first_team_id
    bf_name = mm.home_team if mm.home_team_id == bf else mm.away_team
    wpf = np.where(d.innings == 1, d.wp_before, 1 - d.wp_before)
    x = np.arange(len(d))
    brk = int((d.innings == 1).sum())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(x, wpf * 100, color=ps.BLUE, lw=2)
    a1.axhline(50, color=ps.GREY, lw=1, ls="--")
    a1.axvline(brk - 0.5, color=ps.MUTED, lw=1)
    a1.set_ylim(0, 100); a1.set_ylabel(f"Win prob. {bf_name} (%)")
    a1.text(brk + 1, 92, "Chase begins", color=ps.MUTED, fontsize=9)
    cols = np.where(d.is_pressure, ps.PINK, ps.BLUE)
    a2.bar(x, d.LI, color=cols, width=0.9)
    a2.axvline(brk - 0.5, color=ps.MUTED, lw=1)
    a2.set_ylabel("Leverage Index")
    a2.legend(handles=[Patch(color=ps.PINK, label="Pressure ball (20% highest Leverage Index in its innings)"), Patch(color=ps.BLUE, label="Other")],
              loc="upper left")
    a2.set_xlabel("Ball (both innings)")
    a1.set_title(f"{mm.comp_name} ({mm.stage}) {mm.match_date.date()}: {mm.home_team} v {mm.away_team}\n{mm.result}",
                 fontsize=11)
    fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------------------- run
def run(cfg: dict) -> None:
    header("STAGE 5 - LEVERAGE INDEX")
    ps.apply()
    b = pd.read_parquet(interim(cfg, "wp"))
    m = pd.read_parquet(interim(cfg, "matches"))
    models = load_models(cfg)
    pcfg = cfg["pressure"]
    out = Path(cfg["paths"]["out"])
    tabs, figs = out / "tables", out / "figures"
    R = []

    b = add_buckets(b)
    b["outcome"] = classify_outcome(b)
    probs = outcome_probs(b, alpha=pcfg["outcome_smoothing_alpha"])
    probs.to_csv(tabs / "outcome_probs.csv", index=False)
    print(f"outcome-probability buckets: {len(probs)} (smallest n {probs.n_balls.min()}; "
          f"{(probs.n_balls < pcfg['outcome_smoothing_alpha']).sum()} buckets with n < alpha={pcfg['outcome_smoothing_alpha']} "
          f"lean mostly on their parent)")
    ex = probs[(probs.innings == 2) & (probs.phase == "death") & (probs.wkt_bucket == "4-5")]
    print("  e.g. chase, death overs, 4-5 down: p(dot) / p(4) / p(6) / p(wicket) by required-rate band")
    for _, r_ in ex.iterrows():
        print(f"    rrr {r_['rrr_band']:6s} n={r_['n_balls']:5d}  {r_['dot']:.3f} / {r_['4']:.3f} / "
              f"{r_['6']:.3f} / {r_['wicket']:.3f}")

    print(f"computing LI_raw (8 hypothetical outcomes x {len(b):,} balls)...")
    b["LI_raw"] = leverage_raw(b, models, probs)
    b = apply_pressure(b, pcfg)
    b = add_wp_after(b, models)

    # ---------------- 1. innings-break continuity ----------------
    header("Innings-break continuity")
    i1 = b[b.innings == 1]
    last = i1.groupby("match_id").tail(1).set_index("match_id")
    first2 = b[(b.innings == 2) & (b.innings_ball_num == 1)].set_index("match_id")
    done = ((last.legal_before + last.is_legal.astype(int)) >= last.max_balls) | \
           ((last.wkts_before + last.is_wicket.astype(int)) >= 10)
    ids = last.index[done].intersection(first2.index)
    gap = last.loc[ids, "wp_before"] - (1 - first2.loc[ids, "wp_before"])
    gap_term = last.loc[ids, "wp_after"] - (1 - first2.loc[ids, "wp_before"])
    full = i1[i1.max_balls == 120]
    lb = full.legal_before
    li_last6 = full.loc[lb >= 114, "LI"].mean()
    li_mid = full.loc[(lb >= 102) & (lb <= 113), "LI"].mean()
    li_final = full.loc[lb == 119, "LI"].mean()
    li_115 = full.loc[(lb >= 114) & (lb <= 118), "LI"].mean()
    cont = dict(n_innings=int(len(ids)), mean_gap=float(gap.mean()), mean_abs_gap=float(gap.abs().mean()),
                p90_abs_gap=float(gap.abs().quantile(0.9)),
                terminal_vs_first_chase_ball_mean_abs=float(gap_term.abs().mean()),
                li_last6=float(li_last6), li_balls_103_114=float(li_mid), li_final_ball=float(li_final),
                li_balls_115_119=float(li_115))
    print(f"  WP1 before last ball vs 1 - WP2 at first chase ball ({len(ids)} completed innings): "
          f"mean gap {gap.mean():+.4f}, mean |gap| {gap.abs().mean():.4f}, p90 |gap| {gap.abs().quantile(.9):.4f}")
    print(f"  innings-end value (smooth chase-start curve) vs tree WP2 at first chase ball: "
          f"mean |gap| {gap_term.abs().mean():.4f}, mean {gap_term.mean():+.4f}")
    print(f"  mean LI: balls 103-114 {li_mid:.3f} | last 6 balls {li_last6:.3f} (ratio {li_last6 / li_mid:.2f}) | "
          f"balls 115-119 {li_115:.3f} | final ball {li_final:.3f}")
    check(abs(gap.mean()) <= 0.03, f"no systematic WP jump across the break (|mean gap| {abs(gap.mean()):.4f} <= 0.03)", R)
    check(li_final / li_115 < 1.5, f"final ball of innings 1 not inflated vs balls 115-119 (ratio {li_final / li_115:.2f})", R)
    with open(tabs / "li_continuity.json", "w") as f:
        json.dump(cont, f, indent=2)

    # ---------------- 2. LI distribution and flags ----------------
    header("LI distribution and pressure flags")
    for inn in (1, 2):
        d = b[b.innings == inn]
        print(f"  innings {inn}: mean LI {d.LI.mean():.3f} | quantiles "
              f"{d.LI.quantile([.1, .25, .5, .75, .8, .9, .99, .995]).round(2).to_dict()} | max {d.LI.max():.1f} "
              f"| LI_capped max {d.LI_capped.max():.2f} ({(d.LI > d.LI_capped).sum()} balls capped)")
    if pcfg["percentile_scope"] == "per_innings":
        check(np.allclose(b.groupby("innings").LI.mean(), 1.0), "mean LI = 1 within each innings", R)
    else:
        check(np.isclose(b.LI.mean(), 1.0), "mean LI = 1 globally", R)
    share = b.groupby("innings")[["pressure_top10", "pressure_top20", "pressure_top30"]].mean()
    print("  pressure-flag shares by innings:\n" + share.round(3).to_string())
    check(((share.pressure_top20 - 0.20).abs() < 0.005).all(), "top-20% flag covers 20% of each innings", R)
    print("\n  mean LI by over (both innings):")
    print(b.groupby(["innings", "over_num"]).LI.mean().unstack().round(2).to_string())
    print("\n  % of pressure balls by phase (within innings):")
    tab = (b[b.is_pressure].groupby("innings").phase.value_counts(normalize=True).unstack()
           .reindex(columns=list(cfg["phases"])) * 100)
    print(tab.round(1).to_string())
    rate = b.groupby(["innings", "phase"]).is_pressure.mean().unstack().reindex(columns=list(cfg["phases"])) * 100
    print("  % of balls in each phase that are pressure balls:\n" + rate.round(1).to_string())
    mid_share = tab.loc[:, "middle"].min()
    check(tab.loc[:, "death"].max() < 90 and mid_share > 10,
          f"pressure balls not all at the death (middle overs >= {mid_share:.0f}% in both innings)", R)
    for inn in (1, 2):
        mid = b[(b.innings == inn) & (b.phase == "middle")]
        print(f"  innings {inn} middle overs: pressure-ball rate after a collapse (>=2 wkts in 18 balls) "
              f"{mid[mid.wkts_last_18_balls >= 2].is_pressure.mean():.1%} vs otherwise "
              f"{mid[mid.wkts_last_18_balls < 2].is_pressure.mean():.1%}")

    # ---------------- 3. descriptive flags ----------------
    b["flag_collapse"] = b.wkts_last_18_balls >= 2
    b["flag_steep_chase"] = (b.innings == 2) & ((b.rrr - b.crr) >= 2).fillna(False)
    b["flag_new_batter"] = b.striker_balls_so_far < 6
    b["flag_death_defence"] = (b.innings == 2) & (b.over_num >= 16) & b.wp_before.between(0.2, 0.8)
    b["flag_knockout"] = b.is_knockout
    fl = [c for c in b.columns if c.startswith("flag_")]
    print("\n  descriptive flags: share of all balls / share of pressure balls / mean LI when flagged")
    for c in fl:
        print(f"    {c:20s} {b[c].mean():6.1%} / {b.loc[b.is_pressure, c].mean():6.1%} / {b.loc[b[c], 'LI'].mean():.2f}")

    # ---------------- 4. concentration of realised WP movement ----------------
    header("Concentration: share of total |dWP| on the top-LI balls")
    ok = b.dwp.notna()
    rows = []
    for inn in (1, 2):
        d = b[ok & (b.innings == inn)]
        tot = d.dwp.abs().sum()
        r_ = {"innings": inn, "n_balls": len(d), "total_abs_dwp": tot}
        for q in (10, 20, 30):
            r_[f"top{q}_share_of_abs_dwp"] = d.loc[d[f"pressure_top{q}"], "dwp"].abs().sum() / tot
        rows.append(r_)
    conc = pd.DataFrame(rows)
    conc.to_csv(tabs / "li_concentration.csv", index=False)
    for r_ in rows:
        print(f"  innings {r_['innings']}: top 10% of balls by LI carry {r_['top10_share_of_abs_dwp']:.0%}, "
              f"top 20% carry {r_['top20_share_of_abs_dwp']:.0%}, top 30% carry {r_['top30_share_of_abs_dwp']:.0%} "
              f"of all win-probability movement")
    # informational headline; the only gate is "better than random" (a random 20% would carry 20%)
    check((conc.top20_share_of_abs_dwp > 0.20).all(),
          "top-20% LI balls carry more |dWP| than a random 20% of balls (sanity gate; numbers above are the headline)", R)

    # ---------------- 5. LI spread within buckets ----------------
    header("LI spread within innings x phase x wickets buckets")
    cell = ["innings", "phase", "wkt_bucket"]
    sp = b.groupby(cell).agg(n=("LI", "size"), li_p10=("LI", lambda s: s.quantile(.1)),
                             li_p50=("LI", "median"), li_p90=("LI", lambda s: s.quantile(.9)),
                             pct_pressure=("is_pressure", "mean")).reset_index()
    sp["p90_over_p10"] = sp.li_p90 / sp.li_p10.clip(lower=0.01)
    sp = sp[sp.n >= 200]
    sp.to_csv(tabs / "li_within_bucket_spread.csv", index=False)
    show = sp.copy()
    show["pct_pressure"] = (show.pct_pressure * 100).round(1)
    print(show.round(2).to_string(index=False))
    mixed = ((sp.pct_pressure > 0.05) & (sp.pct_pressure < 0.95)).mean()
    print(f"  {mixed:.0%} of buckets contain both pressure and non-pressure balls (5-95% pressure); "
          f"median p90/p10 LI ratio within a bucket: {sp.p90_over_p10.median():.1f}x")

    # ---------------- 6. outcome rates (informational) ----------------
    header("Outcome rates: pressure vs other balls within the same innings x phase x wickets bucket (info)")
    lg = b[b.is_legal].copy()
    lg["dismissal"] = lg.is_wicket.astype(float)
    lg["dot"] = (lg.team_runs == 0) & ~lg.is_wicket
    lg["boundary"] = lg.is_four | lg.is_six
    lg["rpb"] = lg.team_runs
    metr = ["dismissal", "dot", "boundary", "rpb"]
    g = lg.groupby(cell + ["is_pressure"])[metr].mean().unstack("is_pressure")
    n = lg.groupby(cell + ["is_pressure"]).size().unstack("is_pressure").fillna(0)
    rows = []
    for idx in g.index:
        if n.loc[idx].min() < 100:
            continue
        r_ = dict(zip(cell, idx), n_pressure=int(n.loc[idx, True]), n_other=int(n.loc[idx, False]))
        for mt in metr:
            r_[f"{mt}_pressure"] = g.loc[idx, (mt, True)]
            r_[f"{mt}_other"] = g.loc[idx, (mt, False)]
        rows.append(r_)
    ot = pd.DataFrame(rows)
    ot.to_csv(tabs / "li_outcome_rates.csv", index=False)
    show = ot.copy()
    for mt in ["dismissal", "dot", "boundary"]:
        show[mt] = (show[f"{mt}_pressure"] * 100).round(1).astype(str) + " vs " + (show[f"{mt}_other"] * 100).round(1).astype(str)
    show["runs/ball"] = show.rpb_pressure.round(2).astype(str) + " vs " + show.rpb_other.round(2).astype(str)
    print("  (pressure vs other, %; cells with >= 100 balls in both groups)")
    print(show[cell + ["n_pressure", "n_other", "dismissal", "dot", "boundary", "runs/ball"]].to_string(index=False))
    wts = ot.n_pressure
    print("\n  standardised to the pressure-ball mix of cells:")
    for mt in metr:
        pv, ov = np.average(ot[f"{mt}_pressure"], weights=wts), np.average(ot[f"{mt}_other"], weights=wts)
        f_, unit = (1, "") if mt == "rpb" else (100, "%")
        print(f"    {mt:10s} pressure {pv * f_:.2f}{unit} vs other {ov * f_:.2f}{unit}")

    # ---------------- 7. top-LI balls ----------------
    header("Top 5 highest-LI balls")
    mm = m.set_index("match_id")
    top = b.nlargest(5, "LI").copy()
    top["comp"] = top.match_id.map(mm.comp_name)
    top["teams"] = top.match_id.map(mm.home_team) + " v " + top.match_id.map(mm.away_team)
    top["ball"] = top.over_num.astype(str) + "." + top.over_ball_num.astype(str)
    top["state"] = top.score_before.astype(int).astype(str) + "/" + top.wkts_before.astype(int).astype(str)
    top["need"] = np.where(top.innings == 2, top.runs_needed.astype("Int64").astype(str) + " off " +
                           top.balls_left.astype(int).astype(str), "-")
    cols = ["match_id", "comp", "teams", "innings", "ball", "state", "need", "wp_before", "wp_after", "LI",
            "striker_name", "bowler_name", "outcome"]
    print(top[cols].round(3).to_string(index=False))
    b.nlargest(50, "LI")[["match_id", "innings", "over_num", "over_ball_num", "score_before", "wkts_before",
                          "runs_needed", "balls_left", "wp_before", "wp_after", "LI", "striker_name",
                          "bowler_name", "outcome"]].to_csv(tabs / "li_top_balls.csv", index=False)
    tense = top.wp_before.between(0.15, 0.85).mean()
    check(tense >= 0.6, f"top-5 LI balls are in-the-balance moments (WP 0.15-0.85): {tense:.0%}", R)

    # ---------------- figures ----------------
    vmax = float(np.nanpercentile(b.groupby(["innings", "wkts_before", "over_num"]).LI.mean(), 97))
    _heatmap(b, 1, figs / "li_heatmap_inn1.png", vmax)
    _heatmap(b, 2, figs / "li_heatmap_inn2.png", vmax)
    _phase_share(b, figs / "pressure_share_by_phase.png", cfg["phases"])
    for old in figs.glob("example_match_*.png"):
        old.unlink()
    ms = b.groupby("match_id").agg(li_sum=("LI", "sum"))
    mm2 = mm.join(ms)
    ipl_ko = mm2[(mm2.comp_group == "IPL") & mm2.is_knockout & (mm2.result_type == "decisive")]
    t20 = mm2[(mm2.tier == "T20I_FM") & (mm2.result_type == "decisive") & ~mm2.is_dls]
    t20 = t20[(t20.win_team_id == t20.chase_team_id) & (t20.margin_kind == "wickets")]
    last2 = b[b.innings == 2].groupby("match_id").tail(1).set_index("match_id")
    close = t20.index[t20.index.isin(last2.index[last2.balls_left <= 6])]
    for mid, tag in [(ipl_ko.li_sum.idxmax(), "ipl_playoff"), (t20.loc[close].li_sum.idxmax(), "t20i_close_chase")]:
        _example_match(b, m, mid, figs / f"example_match_{tag}_{mid}.png")
        r_ = mm.loc[mid]
        print(f"  example match ({tag}): {mid} {r_.comp_name} {r_.stage}: {r_.result}")

    b.to_parquet(interim(cfg, "leverage"), index=False)
    print(f"\n  wrote leverage.parquet ({len(b):,} rows)")
    n_fail = sum(not ok_ for ok_, _ in R)
    print(f"\nStage 5: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 5 checks failed - STOP and review.")

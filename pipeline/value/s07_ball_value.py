"""Stage 7: per-ball value in two currencies.

A. Runs Above Expected (primary)
   v_bat  = (runs - xR_bat) - RV * (batter_out - xO_bat)              on balls faced by the striker
   v_bowl = (xR_bowl - bowler_conceded) + RV * (bowler_wicket - xW_bowl)   on all deliveries
   Non-striker run-out: the dismissed non-striker gets ns_runout_penalty = -RV (striker unaffected).
B. Win Probability Added (secondary), batting-side dwp = wp_after - wp_before
   striker:      wpa_bat  = dwp on faced balls, excluding byes / leg-byes / penalty and NS run-outs
   non-striker:  wpa_ns   = dwp on a non-striker run-out
   bowler:       wpa_bowl = -dwp on deliveries excluding byes / leg-byes / penalty / run-outs
   *_over_LI = wpa / max(LI_capped, floor)  (for the baseball-style clutch cross-check)

Outputs: interim/ball_value.parquet, tables raw_sr_trap.csv, value_face_validity.csv,
         figures raw_sr_trap.png
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.common import plotstyle as ps

NO_CREDIT_EXTRAS = {"byes", "legbyes", "Penalty"}


def display_names(b: pd.DataFrame) -> pd.Series:
    """player_id -> name from the most recent occurrence (striker, non-striker or bowler)."""
    cols = [("striker_id", "striker_name"), ("non_striker_id", "non_striker_name"), ("bowler_id", "bowler_name")]
    parts = [b[["match_date", "innings_ball_num", i, n]].set_axis(["d", "k", "pid", "name"], axis=1) for i, n in cols]
    a = pd.concat(parts).dropna(subset=["pid"]).sort_values(["d", "k"])
    return a.groupby("pid").name.last()


def run(cfg: dict) -> None:
    header("STAGE 7 - BALL VALUE")
    ps.apply()
    b = pd.read_parquet(interim(cfg, "expected"))
    m = pd.read_parquet(interim(cfg, "matches"))
    out = Path(cfg["paths"]["out"])
    floor = cfg["value"]["li_floor_for_ratio"]
    R = []

    faced = b.is_faced.values
    bo = b.batter_out.values.astype(float)
    # ---------------- A. runs above expected ----------------
    b["v_bat"] = np.where(faced, (b.runs - b.xR_bat) - b.RV * (bo - b.xO_bat), np.nan)
    b["v_bowl"] = (b.xR_bowl - b.bowler_conceded) + b.RV * (b.bowler_wicket - b.xW_bowl)
    nso = b.nonstriker_out.values
    b["ns_runout_penalty"] = np.where(nso, -b.RV, 0.0)
    b["ns_player_id"] = np.where(nso, b.non_striker_id, np.nan)

    # ---------------- B. win probability added ----------------
    ext = b.extras_type
    no_credit = ext.isin(NO_CREDIT_EXTRAS).values
    dwp = b.dwp.values
    b["wpa_bat"] = np.where(faced & ~no_credit & ~nso, dwp, np.nan)
    b["wpa_ns"] = np.where(nso, dwp, np.nan)
    runout = (b.dismissal_type == "RunOut").values
    b["wpa_bowl"] = np.where(~no_credit & ~runout, -dwp, np.nan)
    li_w = np.maximum(b.LI_capped.values, floor)
    b["LI_w"] = li_w
    b["wpa_bat_over_LI"] = b.wpa_bat / li_w
    b["wpa_bowl_over_LI"] = b.wpa_bowl / li_w

    # ---------------- sanity: centring ----------------
    header("Centring (by construction ~0)")
    ok = ~b.data_suspect.values
    mb, mw = np.nanmean(b.v_bat.values[ok & faced]), b.v_bowl.values[ok].mean()
    print(f"  mean v_bat per ball faced:  {mb:+.4f} runs (sum {np.nansum(b.v_bat.values[ok]):+,.0f} over {(ok & faced).sum():,} balls)")
    print(f"  mean v_bowl per delivery:   {mw:+.4f} runs (sum {b.v_bowl.values[ok].sum():+,.0f} over {ok.sum():,} deliveries)")
    print(f"  non-striker run-outs: {nso.sum()} balls, total penalty {b.ns_runout_penalty.sum():.0f} runs "
          f"({b.ns_runout_penalty.sum() / faced.sum():+.4f} per ball faced if folded in)")
    check(abs(mb) <= 0.02 and abs(mw) <= 0.02, "mean v_bat and v_bowl within ±0.02 runs per ball", R)
    # decomposition: runs part vs wicket part
    print(f"  v_bat parts: runs {np.nanmean((b.runs - b.xR_bat).values[faced]):+.4f}, wicket "
          f"{np.nanmean((-b.RV * (bo - b.xO_bat)).values[faced]):+.4f}")
    print(f"  v_bowl parts: runs {(b.xR_bowl - b.bowler_conceded).mean():+.4f}, wicket "
          f"{(b.RV * (b.bowler_wicket - b.xW_bowl)).mean():+.4f}")

    # WPA telescoping: per innings sum of dwp = WP at end - WP at start
    g = b.groupby(["match_id", "innings"])
    tel = (g.dwp.sum() - (g.wp_after.last() - g.wp_before.first())).abs().max()
    check(tel < 1e-9, f"dwp telescopes within each innings (max error {tel:.2e})", R)
    cov_bat = np.nansum(np.abs(b.wpa_bat)) + np.nansum(np.abs(b.wpa_ns))
    print(f"  share of |dwp| credited to batters: {cov_bat / np.nansum(np.abs(dwp)):.1%}; "
          f"to bowlers: {np.nansum(np.abs(b.wpa_bowl)) / np.nansum(np.abs(dwp)):.1%} (rest: byes/leg-byes/penalty, run-outs)")
    print(f"  wpa/LI: LI floor {floor} binds on {(b.LI_capped < floor).mean():.1%} of balls; "
          f"|wpa_bat/LI| p99 {np.nanquantile(np.abs(b.wpa_bat_over_LI), .99):.3f}")

    # ---------------- face validity ----------------
    header("Face validity: top 10 by total runs above expected (all balls, no tuning)")
    names = display_names(b)
    d = b[ok]
    bat = d[faced[ok]].groupby("striker_id").agg(balls=("v_bat", "size"), v=("v_bat", "sum"),
                                                  runs=("runs", "sum"), wpa=("wpa_bat", "sum"))
    ns = d[d.nonstriker_out].groupby("non_striker_id").ns_runout_penalty.sum()
    bat["v"] = bat.v.add(ns, fill_value=0)
    bat["v_per100"] = bat.v / bat.balls * 100
    bat["name"] = bat.index.map(names)
    bowl = d.groupby("bowler_id").agg(balls=("v_bowl", "size"), v=("v_bowl", "sum"), wkts=("bowler_wicket", "sum"),
                                      conceded=("bowler_conceded", "sum"), wpa=("wpa_bowl", "sum"))
    bowl["v_per100"] = bowl.v / bowl.balls * 100
    bowl["name"] = bowl.index.map(names)
    tb = bat.nlargest(10, "v")
    tw = bowl.nlargest(10, "v")
    print("  batters:")
    print(tb[["name", "balls", "runs", "v", "v_per100", "wpa"]].round(2).to_string())
    print("  bowlers:")
    print(tw[["name", "balls", "wkts", "conceded", "v", "v_per100", "wpa"]].round(2).to_string())
    print("  bottom 3 batters / bowlers by total v:",
          bat.nsmallest(3, "v").name.tolist(), "/", bowl.nsmallest(3, "v").name.tolist())
    fv = pd.concat([tb.assign(role="bat"), tw.assign(role="bowl")])
    fv.to_csv(out / "tables" / "value_face_validity.csv")
    rho_bat = bat[bat.balls >= 300][["v_per100", "wpa"]].corr(method="spearman").iloc[0, 1]
    rho_bowl = bowl[bowl.balls >= 300][["v_per100", "wpa"]].corr(method="spearman").iloc[0, 1]
    print(f"  Spearman(runs above expected per 100 balls, total WPA), players with >= 300 balls: "
          f"batters {rho_bat:.2f}, bowlers {rho_bowl:.2f}")

    # ---------------- raw SR trap ----------------
    header("The raw strike-rate trap: pressure vs normal balls (balls faced)")
    f_ = b[faced & ok].copy()
    f_["bnd"] = (f_.is_four | f_.is_six).astype(float)
    f_["dot0"] = (f_.runs == 0).astype(float)

    def summ(x):
        return pd.Series({"balls": len(x), "raw_SR": 100 * x.runs.mean(), "expected_SR": 100 * x.xR_bat.mean(),
                          "dismissal_pct": 100 * x.batter_out.mean(), "expected_dismissal_pct": 100 * x.xO_bat.mean(),
                          "boundary_pct": 100 * x.bnd.mean(), "expected_boundary_pct": 100 * x.xBnd_bat.mean(),
                          "dot_pct": 100 * x.dot0.mean(), "expected_dot_pct": 100 * x.xDot_bat.mean(),
                          "v_bat_per100": 100 * x.v_bat.mean()})
    f_["ball_type"] = np.where(f_.is_pressure, "pressure", "normal")
    t_all = f_.groupby("ball_type").apply(summ, include_groups=False)
    t_inn = f_.groupby(["innings", "ball_type"]).apply(summ, include_groups=False)
    t_ph = f_.groupby(["phase", "ball_type"]).apply(summ, include_groups=False)
    trap = pd.concat([t_all.assign(scope="all").reset_index(),
                      t_inn.reset_index().assign(scope=lambda x: "innings " + x.innings.astype(str)).drop(columns="innings"),
                      t_ph.reset_index().assign(scope=lambda x: x.phase).drop(columns="phase")])
    trap = trap[["scope", "ball_type"] + [c for c in trap.columns if c not in ("scope", "ball_type")]]
    trap.to_csv(out / "tables" / "raw_sr_trap.csv", index=False)
    print(trap.round(1).to_string(index=False))
    gap_raw = t_all.loc["pressure", "raw_SR"] - t_all.loc["normal", "raw_SR"]
    gap_exp = t_all.loc["pressure", "expected_SR"] - t_all.loc["normal", "expected_SR"]
    print(f"\n  raw SR gap (pressure - normal): {gap_raw:+.1f}; expected SR gap: {gap_exp:+.1f}; "
          f"runs above expected per 100 balls: pressure {t_all.loc['pressure', 'v_bat_per100']:+.2f}, "
          f"normal {t_all.loc['normal', 'v_bat_per100']:+.2f}")
    # bowlers' version
    w_ = b[ok].copy()
    w_["legal_ball"] = w_.is_legal.astype(float)
    bw = w_.groupby(np.where(w_.is_pressure, "pressure", "normal")).apply(
        lambda x: pd.Series({"deliveries": len(x), "raw_econ": 6 * x.bowler_conceded.sum() / x.legal_ball.sum(),
                             "expected_econ": 6 * x.xR_bowl.sum() / x.legal_ball.sum(),
                             "wkt_pct": 100 * x.bowler_wicket.mean(), "expected_wkt_pct": 100 * x.xW_bowl.mean()}),
        include_groups=False)
    print("  bowlers:\n" + bw.round(2).to_string())

    # figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    x = np.arange(2)
    w = 0.36
    cats = ["normal", "pressure"]
    for ax, (raw, exp_, lab, unit) in zip(axes, [("raw_SR", "expected_SR", "Strike rate", "runs per 100 balls"),
                                                  ("dismissal_pct", "expected_dismissal_pct", "Dismissal rate", "% of balls faced")]):
        rv = t_all.loc[cats, raw].values
        ev = t_all.loc[cats, exp_].values
        ax.bar(x - w / 2 - 0.01, rv, w, color=ps.BLUE, label="Actual (raw)")
        ax.bar(x + w / 2 + 0.01, ev, w, color=ps.PINK, label="Expected for the situation")
        for xi, v in zip(x - w / 2 - 0.01, rv):
            ax.text(xi, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=ps.INK)
        for xi, v in zip(x + w / 2 + 0.01, ev):
            ax.text(xi, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=ps.INK)
        ax.set_xticks(x); ax.set_xticklabels(["Normal balls", "Pressure balls\n(20% highest Leverage Index)"])
        ax.set_title(lab); ax.set_ylabel(unit)
        ax.set_ylim(0, max(rv.max(), ev.max()) * 1.18)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Raw pressure-ball rates differ only because the situations differ", fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.78)
    fig.savefig(out / "figures" / "raw_sr_trap.png")
    plt.close(fig)

    b.to_parquet(interim(cfg, "ball_value"), index=False)
    print(f"\n  wrote ball_value.parquet ({len(b):,} rows), tables raw_sr_trap.csv, value_face_validity.csv, "
          f"figures/raw_sr_trap.png")
    n_fail = sum(not ok_ for ok_, _ in R)
    print(f"\nStage 7: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 7 checks failed - STOP and review.")

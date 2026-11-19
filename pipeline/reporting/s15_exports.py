"""Stage 15: final answers, evidence cards, report figures, dashboard tables, REPORT_NUMBERS.md.

Final top 10 = Stage 12 consensus top 10 (median rank across variants, tie-break P(top 10), then default RUP).
Pressure types (strongest needs >= 30 balls): batters chase-death (inns 2, overs 16-20), set-death (inns 1,
overs 16-20), collapse, powerplay, knockout; bowlers defend-death (inns 2, overs 16-20), first-innings death,
collapse, powerplay, knockout. P* per type shrinks toward the player's B* with the outcome k_P.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim, load_raw
from pipeline.value.s07_ball_value import display_names
from pipeline.ratings.s08_player_ratings import ball_weights, ROLE_COLS
from pipeline.ratings.s08c_process_ratings import load_balls
from pipeline.reporting.cards import card
from pipeline.common import plotstyle as ps

PTYPES = {
    "bat": [("chase_death", "chase-death", lambda d: (d.innings == 2) & (d.phase == "death")),
            ("set_death", "set-death (1st inns)", lambda d: (d.innings == 1) & (d.phase == "death")),
            ("collapse", "collapse", lambda d: d.flag_collapse),
            ("powerplay", "powerplay", lambda d: d.phase == "powerplay"),
            ("knockout", "knockout", lambda d: d.flag_knockout)],
    "bowl": [("defend_death", "defend-death", lambda d: (d.innings == 2) & (d.phase == "death")),
             ("inn1_death", "death (1st inns)", lambda d: (d.innings == 1) & (d.phase == "death")),
             ("collapse", "collapse", lambda d: d.flag_collapse),
             ("powerplay", "powerplay", lambda d: d.phase == "powerplay"),
             ("knockout", "knockout", lambda d: d.flag_knockout)],
}


# --------------------------------------------------------------------------- per-player evidence
def pressure_types(b, w, role, kP, bstar):
    pid, vcol, _ = ROLE_COLS[role]
    base = b.is_pressure.values & (b.is_faced.values if role == "bat" else True)
    rows = []
    for key, lab, fn in PTYPES[role]:
        m = base & fn(b).values.astype(bool)
        d = pd.DataFrame({"player_id": b[pid].values[m], "w": w[m], "wv": w[m] * b[vcol].values[m]})
        g = d.groupby("player_id").agg(sw=("w", "sum"), sw2=("w", lambda x: (x ** 2).sum()), swv=("wv", "sum"),
                                       balls=("w", "size"))
        g["P_raw"] = 100 * g.swv / g.sw
        n = g.sw ** 2 / g.sw2
        B = g.index.map(bstar).astype(float)
        g["P_star"] = (n * g.P_raw + kP * B) / (n + kP)
        g = g.reset_index()
        g["type_key"], g["type"], g["role"] = key, lab, role
        rows.append(g[["role", "player_id", "type_key", "type", "balls", "P_raw", "P_star"]])
    return pd.concat(rows, ignore_index=True)


def raw_vs_expected(b, role):
    if role == "bat":
        f = b[b.is_faced]
        g = f.groupby(["striker_id", "is_pressure"]).agg(raw=("runs", "mean"), exp=("xR_bat", "mean"))
        g = g * 100
    else:
        g = b.groupby(["bowler_id", "is_pressure"]).agg(c=("bowler_conceded", "sum"), x=("xR_bowl", "sum"),
                                                          l=("is_legal", "sum"))
        g = pd.DataFrame({"raw": 6 * g.c / g.l, "exp": 6 * g.x / g.l})
    g = g.unstack("is_pressure")
    g.columns = [f"{a}_{'pressure' if p else 'normal'}" for a, p in g.columns]
    return g


def top_wpa_balls(b, m, role, pid_val, k=3):
    pid, _, wcol = ROLE_COLS[role]
    d = b[(b[pid] == pid_val) & b.is_pressure & (b.is_faced if role == "bat" else True)]
    d = d.nlargest(k, wcol)
    mm = m.set_index("match_id")
    out = []
    for _, r in d.iterrows():
        mi = mm.loc[r.match_id]
        # over_num is 1-based; cricket notation "18.6" = 6th ball of the 19th over
        sit = f"inns {int(r.innings)}, ball {int(r.over_num) - 1}.{int(r.over_ball_num)}, {int(r.score_before)}/{int(r.wkts_before)}"
        if r.innings == 2 and pd.notna(r.runs_needed):
            sit += f", needing {int(r.runs_needed)} off {int(r.balls_left)}"
        sit += f" (batting side win prob {r.wp_before:.0%})"
        if role == "bat":
            res = f"{int(r.runs)} run{'s' if r.runs != 1 else ''}"
        else:
            res = f"{int(r.bowler_conceded)} conceded" + (f", WICKET ({r.dismissal_type})" if r.bowler_wicket else "")
        lab = f"{r.deliv_fine} / {r.stroke_fine}"
        out.append(f"{mi.comp_name}, {str(mi.match_date)[:10]}, {mi.home_team} v {mi.away_team} - {sit}. "
                   f"Outcome: {res} ({lab}). WPA {100 * r[wcol]:+.1f} pts.")
    return out


# --------------------------------------------------------------------------- figures
def fig_top10_ci(t10, role, path):
    ps.apply()
    d = t10.iloc[::-1]
    fig, ax = ps.plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(d))
    ax.hlines(y, d.RUP_p10, d.RUP_p90, color=ps.BLUE, lw=3, alpha=0.5)
    ax.plot(d.RUP, y, "o", color=ps.PINK, ms=8)
    ax.set_yticks(y, [f"{int(r)}. {n}" for r, n in zip(d["rank"], d.name)], fontsize=9)
    for yi, (hi, p_) in enumerate(zip(d.RUP_p90, d.p_top10)):
        ax.text(hi + 0.05, yi, f"P(top 10) {p_:.0%}", va="center", fontsize=8, color=ps.MUTED)
    ax.set_xlabel("RUP (z-score composite); bar = 80% match-bootstrap interval")
    ax.set_title(f"Top 10 {'batters' if role == 'bat' else 'bowlers'} under pressure")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=d.RUP_p90.max() + 0.9)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_p_vs_b(lb, role, path):
    ps.apply()
    d = lb[lb.eligible]
    fig, ax = ps.plt.subplots(figsize=(8.5, 6.5))
    top = d["final_rank"] <= 10
    ax.scatter(d.R_B[~top], d.D_R[~top], s=22, color=ps.BLUE, alpha=0.45, lw=0, label="eligible pool")
    ax.scatter(d.R_B[top], d.D_R[top], s=60, color=ps.PINK, edgecolor="white", lw=1.5, label="final top 10")
    for _, r in d[top].iterrows():
        ax.annotate(r["name"], (r.R_B, r.D_R), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    xm = d.R_B.median()
    ax.axvline(xm, color=ps.GREY, ls="--", lw=1)
    ax.axhline(0, color=ps.GREY, ls="--", lw=1)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for tx, ty, s, ha in [(xl[1], yl[1], "steady elite + rises", "right"), (xl[0], yl[1], "pressure riser", "left"),
                          (xl[1], yl[0], "good, fades under pressure", "right"), (xl[0], yl[0], "below par", "left")]:
        ax.text(tx, ty, s, ha=ha, va="top" if ty == yl[1] else "bottom", fontsize=9, color=ps.MUTED, style="italic")
    ax.set_xlabel("B: baseline (normal-ball) rating, runs above expected / 100")
    ax.set_ylabel("D = R_P - B: pressure gap, runs / 100")
    ax.set_title(f"{'Batters' if role == 'bat' else 'Bowlers'}: baseline quality vs change under pressure")
    ax.legend(loc="lower center", fontsize=8)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_what_changed(lb, role, path):
    ps.apply()
    d = lb[lb.eligible & lb.final_rank.notna()]
    fig, ax = ps.plt.subplots(figsize=(7, 6.5))
    top = d.final_rank <= 10
    ax.scatter(d.B_rank[~top], d.final_rank[~top], s=20, color=ps.BLUE, alpha=0.45, lw=0, label="eligible")
    ax.scatter(d.B_rank[top], d.final_rank[top], s=55, color=ps.PINK, edgecolor="white", lw=1.2, label="final top 10")
    for _, r in d[top].iterrows():
        ax.annotate(r["name"], (r.B_rank, r.final_rank), xytext=(4, 2), textcoords="offset points", fontsize=7)
    mx = max(d.B_rank.max(), d.final_rank.max())
    ax.plot([1, mx], [1, mx], color=ps.GREY, lw=1)
    ax.axhline(10.5, color=ps.GREY, ls=":", lw=1)
    ax.axvline(10.5, color=ps.GREY, ls=":", lw=1)
    ax.set_xlabel("rank on baseline quality B only")
    ax.set_ylabel("final (consensus) pressure rank")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    tk = [t_ for t_ in (1, 2, 5, 10, 20, 50, 100) if t_ <= mx]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(tk))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_title(f"{'Batters' if role == 'bat' else 'Bowlers'}: what pressure changed")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_league_delivery(lr, path):
    ps.apply()
    d = lr[(lr.family == "deliv_fam") & (lr.n >= 300) & (lr.label != "none")].sort_values("bowler_conceded_pb")
    fig, axes = ps.plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    y = np.arange(len(d))
    axes[0].barh(y, d.bowler_conceded_pb, color=ps.BLUE, height=0.6)
    axes[1].barh(y, d.wicket_pct, color=ps.PINK, height=0.6)
    axes[0].set_yticks(y, [f"{l_} ({n_:,})" for l_, n_ in zip(d.label, d.n)], fontsize=8)
    for ax, col, fmt in [(axes[0], "bowler_conceded_pb", "{:.2f}"), (axes[1], "wicket_pct", "{:.1f}%")]:
        for yi, v_ in enumerate(d[col]):
            ax.text(v_, yi, " " + fmt.format(v_), va="center", fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("runs conceded per ball (incl. wides / no-balls)")
    axes[1].set_xlabel("wicket %")
    fig.suptitle("Death overs (16-20): what each delivery type costs", fontweight="bold")
    fig.text(0.5, -0.02, "All deliveries in overs 16-20, 2022-26; delivery families with >= 300 balls (count in brackets).",
             ha="center", fontsize=8, color=ps.MUTED)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_league_stroke(lr, path):
    ps.apply()
    d = lr[(lr.family == "stroke_fam") & (lr.label != "none")].sort_values("dismissal_pct")
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    y = np.arange(len(d))
    axes[0].barh(y, d.dismissal_pct, color=ps.PINK, height=0.6)
    axes[1].barh(y, d.runs_per_ball, color=ps.BLUE, height=0.6)
    axes[0].set_yticks(y, [f"{l_} ({n_:,})" for l_, n_ in zip(d.label, d.n)], fontsize=8)
    for ax, col, fmt in [(axes[0], "dismissal_pct", "{:.1f}%"), (axes[1], "runs_per_ball", "{:.2f}")]:
        for yi, v_ in enumerate(d[col]):
            ax.text(v_, yi, " " + fmt.format(v_), va="center", fontsize=7)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("dismissal % per ball")
    axes[1].set_xlabel("runs off the bat per ball")
    fig.suptitle("Shot families: risk and reward", fontweight="bold")
    fig.text(0.5, -0.03, "All balls faced 2022-26. Stroke labels are recorded by scorers after the ball (some outcome "
             "information leaks into them).", ha="center", fontsize=8, color=ps.MUTED)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_league_mix(b, path):
    ps.apply()
    pace = b[b.pace_or_spin == "Pace"]
    fams = ["Yorker", "Wide Yorker", "Full Toss", "Bouncer", "pace_off", "seam_swing"]
    mix = pace.groupby("is_pressure").deliv_fam.value_counts(normalize=True).unstack().reindex(columns=fams) * 100
    f = b[b.is_faced]
    aer = f.groupby(["phase", "is_pressure"]).is_aerial.mean().unstack() * 100
    aer = aer.reindex(["powerplay", "middle", "death"])
    fig, axes = ps.plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(fams))
    axes[0].bar(x - 0.2, mix.loc[False], 0.38, color=ps.BLUE, label="normal balls")
    axes[0].bar(x + 0.2, mix.loc[True], 0.38, color=ps.PINK, label="pressure balls")
    for xi, (n_, p_) in enumerate(zip(mix.loc[False], mix.loc[True])):
        axes[0].text(xi - 0.2, n_, f"{n_:.1f}", ha="center", va="bottom", fontsize=7)
        axes[0].text(xi + 0.2, p_, f"{p_:.1f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, ["yorker", "wide yorker", "full toss", "bouncer", "pace-off", "seam/swing"], fontsize=8)
    axes[0].set_ylabel("% of pace deliveries")
    axes[0].set_title("Pace bowlers' delivery mix", fontsize=10)
    axes[0].legend(fontsize=8)
    x = np.arange(3)
    axes[1].bar(x - 0.2, aer[False], 0.38, color=ps.BLUE, label="normal balls")
    axes[1].bar(x + 0.2, aer[True], 0.38, color=ps.PINK, label="pressure balls")
    for xi, (n_, p_) in enumerate(zip(aer[False], aer[True])):
        axes[1].text(xi - 0.2, n_, f"{n_:.1f}", ha="center", va="bottom", fontsize=7)
        axes[1].text(xi + 0.2, p_, f"{p_:.1f}", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks(x, ["powerplay", "middle", "death"])
    axes[1].set_ylabel("aerial shots, % of balls faced")
    axes[1].set_title("Batters' aerial-shot rate", fontsize=10)
    fig.suptitle("Under pressure pace bowlers bowl more yorkers, full tosses and slower balls; batters' aerial "
                 "rate barely changes within a phase", fontweight="bold", fontsize=11)
    fig.savefig(path)
    ps.plt.close(fig)


def fig_chase_death(path, tabs):
    ps.apply()
    d = pd.read_csv(tabs / "li_outcome_rates.csv")
    d = d[(d.innings == 2) & (d.phase == "death")]
    fig, axes = ps.plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(d))
    for ax, (mt, lab) in zip(axes, [("dismissal", "wicket % per ball"), ("boundary", "boundary % per ball"),
                                    ("dot", "dot-ball %")]):
        ax.bar(x - 0.2, 100 * d[f"{mt}_other"], 0.38, color=ps.BLUE, label="other balls")
        ax.bar(x + 0.2, 100 * d[f"{mt}_pressure"], 0.38, color=ps.PINK, label="pressure balls")
        for xi, (o_, p_) in enumerate(zip(d[f"{mt}_other"], d[f"{mt}_pressure"])):
            ax.text(xi - 0.2, 100 * o_, f"{100 * o_:.1f}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + 0.2, 100 * p_, f"{100 * p_:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [f"{w_} wkts down" for w_ in d.wkt_bucket], fontsize=8)
        ax.set_title(lab, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Chases, overs 16-20: pressure balls vs other balls in the same wickets state", fontweight="bold")
    fig.text(0.5, -0.03, "Pressure ball = one of the 20% of deliveries with the highest Leverage Index in its innings. Counts: " + ", ".join(
        f"{w_}: {p_:,} pressure / {o_:,} other" for w_, p_, o_ in zip(d.wkt_bucket, d.n_pressure, d.n_other)),
        ha="center", fontsize=8, color=ps.MUTED)
    fig.savefig(path)
    ps.plt.close(fig)


# --------------------------------------------------------------------------- run
def run(cfg: dict) -> None:
    header("STAGE 15 - EXPORTS, EVIDENCE AND REPORT FIGURES")
    out = Path(cfg["paths"]["out"])
    tabs, figs, fin, dash = out / "tables", out / "figures", out / "final", out / "dashboard"
    for p_ in (fin, dash, figs / "cards"):
        p_.mkdir(parents=True, exist_ok=True)
    R = []
    b = load_balls(cfg)
    m = pd.read_parquet(interim(cfg, "matches"))
    rup = pd.read_parquet(interim(cfg, "rup"))
    cons = pd.read_parquet(interim(cfg, "consensus"))
    rs = pd.read_parquet(interim(cfg, "rank_stability"))
    cstab = pd.read_parquet(interim(cfg, "consistency"))
    proc = pd.read_parquet(interim(cfg, "process_ratings"))
    sp = pd.read_parquet(interim(cfg, "player_splits"))
    kv = json.load(open(tabs / "eb_k_values.json", encoding="utf-8"))
    names = display_names(b)
    w = ball_weights(b, cfg)
    lr = pd.read_csv(tabs / "process_label_rates.csv")

    finals, boards, ptypes_all = {}, {}, []
    for role in ("bat", "bowl"):
        lab = "batters" if role == "bat" else "bowlers"
        r = rup[rup.role == role].copy()
        r = r.merge(rs[rs.role == role].drop(columns="role"), on="player_id", how="left")
        r = r.merge(cons[cons.role == role][["player_id", "consensus_rank", "median_rank_variants", "share_top10",
                                            "share_outside_top20", "fragile"]], on="player_id", how="left")
        r = r.merge(cstab[cstab.role == role][["player_id", "pos_share_raw"]], on="player_id", how="left")
        r = r.merge(raw_vs_expected(b, role), left_on="player_id", right_index=True, how="left")
        bstar = r.set_index("player_id").B_star
        pt = pressure_types(b, w, role, kv[role]["k_P"], bstar)
        ptypes_all.append(pt)
        okp = pt[pt.balls >= 30]
        strongest = okp.loc[okp.groupby("player_id").P_star.idxmax()].set_index("player_id")
        r["strongest_type"] = r.player_id.map(strongest["type"]).fillna("none with >= 30 balls")
        r["strongest_key"] = r.player_id.map(strongest["type_key"])
        r["strongest_P_star"] = r.player_id.map(strongest["P_star"])
        r["final_rank"] = r.consensus_rank
        r["B_rank"] = r.R_B.where(r.eligible).rank(ascending=False, method="first")
        r["evidence"] = [
            f"{pr:+.1f} runs/100 above expected on {pb:.0f} pressure balls vs {br:+.1f} on normal balls; shrunk R_P "
            f"{rp:+.1f}; positive in {ps_:.0%} of pressure matches; top 10 in {st:.0%} of variants and "
            f"{pt_:.0%} of bootstrap resamples; strongest: {sty}."
            for pr, pb, br, rp, ps_, st, pt_, sty in zip(r.P_raw, r.pressure_balls, r.B_raw, r.R_P, r.pos_share_raw,
                                                         r.share_top10.fillna(0), r.p_top10.fillna(0), r.strongest_type)]
        boards[role] = r
        t10 = r[r.final_rank <= 10].sort_values("final_rank").copy()
        t10["rank"] = t10.final_rank.astype(int)
        finals[role] = t10
        header(f"Final top 10 {lab}")
        print(t10[["rank", "name", "RUP", "R_P", "P_star", "proc_P_star", "R_B", "D_R", "C", "P_raw", "pressure_balls",
                   "p_top10", "median_rank_variants", "fragile", "strongest_type"]].round(2).to_string(index=False))
        # ---- checks (stop condition (b))
        el = cfg["eligibility"]
        ok_el = bool(t10.eligible.all() and t10.guard_ok.all() and
                     (t10.last_match_date >= pd.Timestamp(el["active_since"])).all())
        check(len(t10) == 10 and t10.player_id.is_unique, f"{role}: 10 unique players", R)
        check(ok_el, f"{role}: every final top-10 player is eligible (active since {el['active_since']}, minimums, "
              f"P_raw >= 0)", R)
        # ---- CSVs
        cols = {"rank": "rank", "player_id": "player_id", "name": "name", "teams": "teams", "RUP": "RUP", "R_P": "R_P",
                "P_star": "outcome_P_star", "proc_P_star": "process_P_star", "B_star": "outcome_B_star",
                "R_B": "B_blended", "D_R": "D", "C": "C", "P_raw": "P_raw", "pressure_balls": "pressure_balls",
                "pressure_innings": "pressure_innings", "p_top10": "P_top10", "median_rank": "median_rank_bootstrap",
                "rank_p10": "rank_80_lo", "rank_p90": "rank_80_hi", "median_rank_variants": "median_rank_variants",
                "fragile": "fragile", "strongest_type": "strongest_pressure_type", "evidence": "evidence"}
        t10[list(cols)].rename(columns=cols).round(4).to_csv(fin / f"top10_{lab}.csv", index=False)
        lbcols = ["final_rank", "rank", "player_id", "name", "teams", "RUP", "R_P", "R_B", "D_R", "C", "P_raw", "B_raw",
                  "P_star", "B_star", "D_star", "proc_P_raw", "proc_B_raw", "proc_P_star", "proc_B_star", "proc_D_star",
                  "pressure_balls", "pressure_innings", "normal_balls", "units", "failure_rate", "pos_share_raw",
                  "wpa_pressure", "wpa_total", "p_top10", "p_top20", "median_rank", "rank_p10", "rank_p90",
                  "median_rank_variants", "share_top10", "share_outside_top20", "fragile", "B_rank", "strongest_type",
                  "eligible", "guard_ok", "last_match_date", "ipl2026_team", "nationality", "overseas", "is_pakistan",
                  "rr_2026", "availability"]
        full = r[r.eligible].sort_values(["final_rank", "rank_unguarded"])[[c for c in lbcols if c in r.columns]]
        for path_ in (fin / f"full_leaderboard_{role}.csv", dash / f"leaderboard_{'batting' if role == 'bat' else 'bowling'}.csv"):
            try:
                full.rename(columns={"rank": "default_rank"}).round(4).to_csv(path_, index=False)
            except PermissionError:
                print(f"  WARNING: {path_} is open in another program (locked); not overwritten - close it and re-run")

    # ---------------- dropdown check
    sq = load_raw(cfg, "squads")
    nm = pd.concat([sq[["player_id", "player_name"]].rename(columns={"player_name": "n"}),
                    b[["striker_id", "striker_name"]].set_axis(["player_id", "n"], axis=1),
                    b[["bowler_id", "bowler_name"]].set_axis(["player_id", "n"], axis=1)]).dropna().drop_duplicates()
    nm["key"] = nm.n.str.strip().str.lower()
    ids_by_name = nm.groupby("key").player_id.agg(lambda s: sorted(set(s)))
    teams = rup.drop_duplicates("player_id").set_index("player_id").teams
    lines = ["Final lists: names exactly as displayed in the data pack (check each against the Unstop dropdown).", ""]
    for role in ("bat", "bowl"):
        lines.append(f"TOP 10 {'BATTERS' if role == 'bat' else 'BOWLERS'}")
        for _, r in finals[role].iterrows():
            others = [i for i in ids_by_name.get(r["name"].strip().lower(), []) if i != r.player_id]
            alias = sorted(set(nm[nm.player_id == r.player_id].n) - {r["name"]})
            s = f"{int(r['rank']):2d}. {r['name']}  (player_id {r.player_id}; {r.teams})"
            if alias:
                s += f"  [also recorded as: {', '.join(alias)}]"
            if others:
                s += "  WARNING duplicate name: " + "; ".join(f"id {o} ({teams.get(o, 'unknown teams')})" for o in others)
            lines.append(s)
        lines.append("")
    dup = ids_by_name[ids_by_name.map(len) > 1]
    lines.append(f"All names shared by more than one player_id in the data ({len(dup)}): " +
                 "; ".join(f"{k_} -> ids {v_}" for k_, v_ in dup.items()))
    (fin / "dropdown_check.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:26]))

    # ---------------- what pressure changed
    wc_rows = []
    for role in ("bat", "bowl"):
        r = boards[role]
        d = r[r.eligible & ((r.final_rank <= 30) | (r.B_rank <= 30))].copy()
        d["rank_change"] = d.B_rank - d.final_rank
        risers = d[d.final_rank <= 30].nlargest(10, "rank_change")
        fallers = d[d.B_rank <= 30].assign(fr=d.final_rank.fillna(999)).nsmallest(10, "rank_change").drop(columns="fr")
        n_not = int((finals[role].B_rank > 10).sum())
        header(f"What pressure changed ({role}): {n_not} of the final top 10 would NOT be top 10 on B alone")
        show = ["name", "B_rank", "final_rank", "rank_change", "P_raw", "D_R", "C"]
        print("  biggest risers (final top 30):")
        print(risers[show].round(2).to_string(index=False))
        print("  biggest fallers (B-only top 30; final_rank NaN = removed by the P_raw >= 0 guard):")
        print(fallers[show].round(2).to_string(index=False))
        for kind, dd in (("riser", risers), ("faller", fallers)):
            wc_rows.append(dd[show].assign(role=role, kind=kind))
        wc_rows.append(pd.DataFrame([{"role": role, "kind": "summary", "name": "final top 10 not top 10 on B alone",
                                      "rank_change": n_not}]))
        fig_what_changed(r, role, figs / f"what_pressure_changed_{role}.png")
        fig_p_vs_b(r, role, figs / f"p_vs_b_scatter_{role}.png")
        fig_top10_ci(finals[role], role, figs / f"top10_{'batters' if role == 'bat' else 'bowlers'}_ci.png")
    pd.concat(wc_rows, ignore_index=True).to_csv(tabs / "what_pressure_changed.csv", index=False)

    # ---------------- league insight figures
    fig_league_delivery(lr, figs / "league_death_delivery_types.png")
    fig_league_stroke(lr, figs / "league_stroke_families.png")
    fig_league_mix(b, figs / "league_pressure_vs_normal_mix.png")
    fig_chase_death(figs / "league_chase_death_outcomes.png", tabs)

    # ---------------- evidence cards
    pt_all = pd.concat(ptypes_all, ignore_index=True)
    for role in ("bat", "bowl"):
        r = boards[role]
        pool = r[r.eligible]
        for _, row in finals[role].iterrows():
            pid = row.player_id
            if role == "bowl":
                specs = [("yorker %", "yorker_pct"), ("full toss %", "full_toss_pct"), ("wide + no-ball %", "wide_nb_pct"),
                         ("slower ball %", "slower_pct"), ("bouncer %", "bouncer_pct")]
            else:
                specs = [("aerial shot %", "aerial_pct"), ("defensive shot %", "defence_pct"),
                         ("dismissals per aerial shot (%)", "outs_per_aerial"),
                         ("shot-decision value / 100", "decision_value"), ("difficulty faced / 100 (+ = easier)", "difficulty_faced")]
            rows = []
            for lab_, c in specs:
                f_ = 100 if c == "outs_per_aerial" else 1
                rows.append([lab_] + [f"{f_ * v_:.1f}" if pd.notna(v_) else "-" for v_ in
                                      (row[f"{c}_normal"], row[f"{c}_pressure"], pool[f"{c}_normal"].median(),
                                       pool[f"{c}_pressure"].median())])
            s = sp[(sp.player_id == pid) & (sp.role == role) & (sp.balls_type == "pressure")]
            splits = []
            for split, val, lab_ in [("innings", "setting", "setting (1st inns)"), ("innings", "chasing", "chasing (2nd inns)"),
                                     ("phase", "powerplay", "powerplay"), ("phase", "middle", "middle overs"),
                                     ("phase", "death", "death overs")]:
                q = s[(s.split == split) & (s.split_value == val)]
                splits.append((lab_, float(q.v_per100.iloc[0]) if len(q) else 0.0, int(q.balls.iloc[0]) if len(q) else 0))
            ptp = pt_all[(pt_all.player_id == pid) & (pt_all.role == role)].set_index("type_key")
            ptypes = [(k_, l_, float(ptp.loc[k_, "P_star"]) if k_ in ptp.index else 0.0,
                       int(ptp.loc[k_, "balls"]) if k_ in ptp.index else 0) for k_, l_, _ in PTYPES[role]]
            p = {**row.to_dict(), "rank": int(row["rank"]), "process_rows": rows, "splits": splits, "ptypes": ptypes,
                 "raw_normal": row.raw_normal, "raw_pressure": row.raw_pressure, "exp_normal": row.exp_normal,
                 "exp_pressure": row.exp_pressure, "C_pool_median": pool.C.median(),
                 "top_balls": top_wpa_balls(b, m, role, pid), "fragile": bool(row.fragile)}
            card(p, figs / "cards" / f"{role}_{int(row['rank']):02d}_{pid}.png")
    n_cards = len(list((figs / "cards").glob("*.png")))
    check(n_cards >= 20, f"{n_cards} evidence cards written", R)

    # ---------------- dashboard tables
    info_cols = ["player_id", "name", "teams", "nationality", "overseas", "is_pakistan", "ipl2026_team", "rr_2026",
                 "availability", "last_match_date"]
    pl = pd.concat([boards[r_][info_cols].assign(role=r_) for r_ in boards]).groupby("player_id").agg(
        {**{c: "first" for c in info_cols[1:]}, "role": lambda s: ",".join(sorted(set(s)))}).reset_index()
    pl["active"] = pl.last_match_date >= pd.Timestamp(cfg["eligibility"]["active_since"])
    pl["age"] = "unknown (player_meta.csv missing)"
    pl.to_csv(dash / "players.csv", index=False)
    spd = pd.concat([sp, pt_all.rename(columns={"type": "split_value"}).assign(split="pressure_type", balls_type="pressure",
                                                                               v_per100=lambda d: d.P_raw)
                     [["player_id", "split", "split_value", "balls_type", "balls", "v_per100", "P_star", "role"]]],
                    ignore_index=True)
    spd.to_csv(dash / "player_splits.csv", index=False)
    pu = pd.read_parquet(interim(cfg, "pressure_units"))
    pu.assign(name=pu.player_id.map(names)).to_csv(dash / "player_match_pressure.csv", index=False)
    dec = m[(m.result_type == "decisive") & ~m.data_suspect].match_id
    # 30 notable matches: the two report example matches + the most tense decisive matches (highest mean LI)
    tense = b[b.match_id.isin(dec)].groupby("match_id").LI_capped.mean().sort_values(ascending=False).index.tolist()
    sel = list(dict.fromkeys([15187, 28957] + tense))[:30]
    tl = b[b.match_id.isin(sel)][["match_id", "innings", "over_num", "over_ball_num", "score_before", "wkts_before",
                                  "wp_before", "LI", "PI", "is_pressure", "striker_id", "striker_name", "bowler_id",
                                  "bowler_name", "outcome", "runs", "team_runs", "is_wicket", "deliv_fine", "stroke_fine"]]
    tl.to_csv(dash / "match_timeline.csv", index=False)
    mi = m[["match_id", "match_date", "comp_name", "comp_group", "season", "home_team", "away_team", "result",
            "result_type", "is_knockout", "is_final", "data_suspect"]].copy()
    mi["in_timeline"] = mi.match_id.isin(sel)
    mi.to_csv(dash / "matches_index.csv", index=False)
    import shutil
    shutil.copy(tabs / "sensitivity_ranks.csv", dash / "sensitivity_ranks.csv")
    shutil.copy(tabs / "validation_metrics.json", dash / "validation_metrics.json")
    for c_ in ("leaderboard_batting", "leaderboard_bowling", "players", "player_splits", "player_match_pressure",
               "match_timeline", "matches_index", "sensitivity_ranks"):
        pd.read_csv(dash / f"{c_}.csv", low_memory=False).to_parquet(dash / f"{c_}.parquet", index=False)
    check(tl.match_id.nunique() == 30, f"match_timeline has exactly 30 matches ({tl.match_id.nunique()})", R)

    # ---------------- figure inventory (report figures + league insights)
    need = ["li_heatmap_inn1.png", "li_heatmap_inn2.png", "pressure_share_by_phase.png", "wp_calibration.png",
            "raw_sr_trap.png", "reliability_curve.png", "temporal_validity.png", "top10_batters_ci.png",
            "top10_bowlers_ci.png", "p_vs_b_scatter_bat.png", "p_vs_b_scatter_bowl.png", "sensitivity_heatmap_bat.png",
            "sensitivity_heatmap_bowl.png", "what_pressure_changed_bat.png", "what_pressure_changed_bowl.png",
            "league_death_delivery_types.png", "league_stroke_families.png", "league_pressure_vs_normal_mix.png",
            "league_chase_death_outcomes.png"]
    missing = [f_ for f_ in need if not (figs / f_).exists()]
    check(not missing and len(list(figs.glob("example_match_*.png"))) >= 1, f"all report figures present {missing or ''}", R)

    from pipeline.reporting.report_numbers import write_report_numbers
    write_report_numbers(cfg, finals, boards)
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 15: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 15 checks failed - review.")

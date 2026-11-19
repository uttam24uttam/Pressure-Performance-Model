"""Stage 11: validation.

a) Split-half reliability (each player's matches in date order split odd/even; real LI pressure flag; k and C
   prior fixed at full-data values): outcome P_raw, B_raw, D_raw, P*; process P_raw, B_raw, D_raw, P*; C.
   Spearman between halves and Spearman-Brown 2r/(1+r), as a curve over the minimum full-data pressure balls.
b) Temporal test (headline): measures from balls <= 2025-12-31 vs IPL 2026 raw pressure outcome (players with
   >= 40 IPL 2026 and >= 30 train pressure balls): raw pressure SR / economy, raw pressure average, career
   SR / economy, outcome P*, process P*, blended R_P, RUP. Player-bootstrap CIs; paired RUP-minus-measure CIs.
   Supplementary: target IPL 2026 P*; rolling origin (train <= 2024, test IPL 2025 + 2026: more players).
c) Rank stability: Stage 9 match bootstrap (outcome P*, B*, C) + the same resamples for process P*, B*
   (Stage 8c) -> R_P, D, C per rep, z-scores per rep, RUP rank among eligible guard-passing players:
   P(top 10), median rank, 80% rank interval.
d) Model quality: WP metrics and calibration figure (Stage 4); expectation / process calibration tables.

Outputs: tables/validation_split_half.csv, validation_temporal.csv, validation_rank_stability.csv,
         validation_metrics.json; figures/reliability_curve.png, temporal_validity.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.ratings.s08_player_ratings import ball_weights, role_frame, compute_ratings
from pipeline.ratings.s08c_process_ratings import load_balls, PROC_COL
from pipeline.ratings.s10_composite import temporal_frames
from pipeline.ratings.scoring import compose, zscore, rup_from_z
from pipeline.common import plotstyle as ps

THRESHOLDS = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300]
SH_METRICS = ["P_raw", "B_raw", "D_raw", "P_star", "proc_P_raw", "proc_B_raw", "proc_D_raw", "proc_P_star", "C"]
SH_LABEL = {"P_raw": "outcome P raw", "B_raw": "outcome B raw", "D_raw": "outcome D raw (P-B)", "P_star": "outcome P*",
            "proc_P_raw": "process P raw", "proc_B_raw": "process B raw", "proc_D_raw": "process D raw",
            "proc_P_star": "process P*", "C": "consistency C"}


def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[m], y[m])[0] if m.sum() > 3 else np.nan


def _boot_rho(x, y, idx):
    out = np.array([_rho(x[j], y[j]) for j in idx])
    return np.nanpercentile(out, [2.5, 97.5])


# --------------------------------------------------------------------------- a) split-half
def split_half(b, cfg, role, w, k_out, k_proc, prior):
    f = role_frame(b, role, w)
    fp = role_frame(b, role, w, vcol_override=PROC_COL[role])
    md = b.drop_duplicates("match_id").set_index("match_id").match_date
    pm = f[["player_id", "match_id"]].drop_duplicates().assign(d=lambda x: x.match_id.map(md))
    pm["half"] = pm.sort_values(["d", "match_id"]).groupby("player_id").cumcount() % 2
    f = f.merge(pm[["player_id", "match_id", "half"]], on=["player_id", "match_id"])
    fp = fp.merge(pm[["player_id", "match_id", "half"]], on=["player_id", "match_id"])
    mb = cfg["eligibility"][f"match_min_pressure_balls_{role}"]
    halves = []
    for h in (0, 1):
        fh = f[f.half == h].drop(columns="half")
        r, _ = compute_ratings(b, cfg, role, w, frame=fh, k_override=k_out)
        rp, _ = compute_ratings(b, cfg, role, w, frame=fp[fp.half == h].drop(columns="half"), k_override=k_proc)
        r = r.merge(rp[["player_id", "P_raw", "B_raw", "P_star"]].rename(
            columns={"P_raw": "proc_P_raw", "B_raw": "proc_B_raw", "P_star": "proc_P_star"}), on="player_id", how="left")
        u = fh[fh.pressure].groupby(["player_id", "match_id"]).agg(
            v=("v", "sum"), balls=("event", lambda e: (e == "ball").sum()), w=("w", "max")).reset_index()
        u = u[u.balls >= mb].assign(pos=lambda d: (d.v > 0) * d.w)
        cu = u.groupby("player_id").agg(pos=("pos", "sum"), n=("w", "sum"))
        r["C"] = r.player_id.map((cu.pos + prior[0]) / (cu.n + prior[0] + prior[1])).fillna(prior[0] / sum(prior))
        r["D_raw"] = r.P_raw - r.B_raw
        r["proc_D_raw"] = r.proc_P_raw - r.proc_B_raw
        halves.append(r.set_index("player_id")[SH_METRICS])
    return halves


def reliability_table(halves, full_balls, rng, role):
    a, c = halves
    ids = a.index.intersection(c.index)
    rows = []
    sb = lambda q: 2 * q / (1 + q) if np.isfinite(q) and q > -1 else np.nan
    for t in THRESHOLDS:
        sel = [i for i in ids if full_balls.get(i, 0) >= t]
        for m in SH_METRICS:
            x, y = a.loc[sel, m].values.astype(float), c.loc[sel, m].values.astype(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            r = _rho(x, y)
            lo, hi = _boot_rho(x, y, [rng.integers(0, len(x), len(x)) for _ in range(400)]) if len(x) > 10 else (np.nan, np.nan)
            rows.append({"role": role, "metric": m, "min_pressure_balls": t, "n": len(x), "rho_halves": r,
                         "rho_lo": lo, "rho_hi": hi, "spearman_brown": sb(r), "sb_lo": sb(lo), "sb_hi": sb(hi)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- b) temporal
def naive_stats(b, role):
    if role == "bat":
        d = b[b.is_faced]
        p = d[d.is_pressure].groupby("striker_id")
        a = d.groupby("striker_id")
        return pd.DataFrame({"raw_pressure_SR": 100 * p.runs.sum() / p.size(),
                             "raw_pressure_avg": p.runs.sum() / p.batter_out.sum().clip(lower=1),
                             "career_SR": 100 * a.runs.sum() / a.size()})
    p = b[b.is_pressure].groupby("bowler_id")
    a = b.groupby("bowler_id")
    # lower is better for bowlers: negated so a positive Spearman always means "predicts better"
    return pd.DataFrame({"raw_pressure_econ": -6 * p.bowler_conceded.sum() / p.is_legal.sum(),
                         "raw_pressure_avg": -p.bowler_conceded.sum() / p.bowler_wicket.sum().clip(lower=1),
                         "career_econ": -6 * a.bowler_conceded.sum() / a.is_legal.sum()})


MEASURES = {"bat": ["raw_pressure_SR", "raw_pressure_avg", "career_SR", "P_star", "proc_P_star", "R_P", "RUP"],
            "bowl": ["raw_pressure_econ", "raw_pressure_avg", "career_econ", "P_star", "proc_P_star", "R_P", "RUP"]}


def temporal_eval(t, btr, role, a, wts, rng, test_label, targets=("P_raw_test", "P_star_test")):
    t = compose(t, a)
    t[["z_R_P", "z_D_R", "z_C"]] = zscore(t, t.train_pool)
    t["RUP"] = rup_from_z(t, wts)
    t = t.merge(naive_stats(btr, role), left_on="player_id", right_index=True, how="left")
    S = t[t.train_ok & t.test_ok]
    idx = [rng.integers(0, len(S), len(S)) for _ in range(2000)]
    rows = []
    for target in targets:
        y = S[target].values
        base = S.RUP.values
        for m in MEASURES[role]:
            x = S[m].values.astype(float)
            lo, hi = _boot_rho(x, y, idx)
            d = np.array([_rho(base[j], y[j]) - _rho(x[j], y[j]) for j in idx])
            rows.append({"role": role, "test": test_label, "target": target, "measure": m, "n": len(S),
                         "rho": _rho(x, y), "lo": lo, "hi": hi,
                         "rup_minus_this_lo": np.nanpercentile(d, 2.5), "rup_minus_this_hi": np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- c) rank stability
def rank_stability(role, rup, boot, bootp, a, weights):
    r = rup[rup.role == role].set_index("player_id")
    E = r.index[r.eligible]
    g = np.array([bool(r.loc[e, "guard_ok"]) for e in E])
    io = pd.Index(boot[f"{role}_player_id"]).get_indexer(E)
    ip = pd.Index(bootp[f"{role}_player_id"]).get_indexer(E)
    assert (io >= 0).all() and (ip >= 0).all()
    P = (1 - a) * boot[f"{role}_P_star"][io].astype(float) + a * bootp[f"{role}_proc_P_star"][ip].astype(float)
    B = (1 - a) * boot[f"{role}_B_star"][io].astype(float) + a * bootp[f"{role}_proc_B_star"][ip].astype(float)
    arr = {"R_P": P, "D_R": P - B, "C": boot[f"{role}_C"][io].astype(float)}
    z = {n: (v - v.mean(axis=0)) / v.std(axis=0, ddof=1) for n, v in arr.items()}
    rupb = weights["P"] * z["R_P"] + weights["D"] * z["D_R"] + weights["C"] * z["C"]
    rk = np.where(g[:, None], rupb, -np.inf)
    ranks = (-rk).argsort(axis=0).argsort(axis=0) + 1.0
    ranks[~g] = np.nan
    out = pd.DataFrame({"player_id": E, "role": role, "p_top10": (ranks <= 10).mean(axis=1),
                        "p_top20": (ranks <= 20).mean(axis=1), "median_rank": np.median(ranks, axis=1),
                        "rank_p10": np.percentile(ranks, 10, axis=1), "rank_p90": np.percentile(ranks, 90, axis=1),
                        "RUP_p10": np.percentile(rupb, 10, axis=1), "RUP_p90": np.percentile(rupb, 90, axis=1),
                        "R_P_p10": np.percentile(P, 10, axis=1), "R_P_p90": np.percentile(P, 90, axis=1),
                        "D_p10": np.percentile(P - B, 10, axis=1), "D_p90": np.percentile(P - B, 90, axis=1),
                        "C_p10": np.percentile(arr["C"], 10, axis=1), "C_p90": np.percentile(arr["C"], 90, axis=1),
                        "B_p10": np.percentile(B, 10, axis=1), "B_p90": np.percentile(B, 90, axis=1)})
    out.loc[~g, ["p_top10", "p_top20"]] = 0.0
    return out


# --------------------------------------------------------------------------- figures
def fig_reliability(tab, path):
    ps.apply()
    show = ["P_raw", "D_raw", "proc_P_raw", "C"]
    fig, axes = ps.plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, role in zip(axes, ("bat", "bowl")):
        for i, m in enumerate(show):
            d = tab[(tab.role == role) & (tab.metric == m)]
            ax.plot(d.min_pressure_balls, d.spearman_brown, color=ps.CAT[i], marker=ps.MARKERS[i], label=SH_LABEL[m])
            ax.fill_between(d.min_pressure_balls, d.sb_lo, d.sb_hi, color=ps.CAT[i], alpha=0.10, lw=0)
        thr = 100 if role == "bat" else 120
        ax.axvline(thr, color=ps.GREY, ls="--", lw=1)
        ax.text(thr + 4, 0.96, f"eligibility\n{thr} balls", color=ps.MUTED, fontsize=8, va="top")
        ax.axhline(0, color=ps.GREY, lw=0.8)
        ax.set_title("Batters" if role == "bat" else "Bowlers")
        ax.set_xlabel("minimum pressure balls (full data)")
        ax.set_ylim(-1.0, 1.0)
    axes[0].set_ylabel("split-half reliability (Spearman-Brown)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Under pressure, what players DO repeats; what happens to them barely does", fontweight="bold", y=1.02)
    fig.text(0.5, -0.15, "Odd vs even matches per player; shaded = 95% player-bootstrap interval. Process = delivery "
             "execution (bowlers) / shot decision (batters), runs above expected per 100 balls.",
             ha="center", color=ps.MUTED, fontsize=8)
    fig.savefig(path)
    ps.plt.close(fig)


NAMES = {"RUP": "RUP (final score)", "R_P": "R_P (blended pressure rating)", "P_star": "outcome P* (shrunk)",
         "proc_P_star": "process P* (shrunk)", "raw_pressure_SR": "raw pressure strike rate",
         "raw_pressure_avg": "raw pressure average", "career_SR": "career strike rate",
         "raw_pressure_econ": "raw pressure economy", "career_econ": "career economy"}


def fig_temporal(tab, path):
    ps.apply()
    ours = {"RUP", "R_P", "P_star", "proc_P_star"}
    fig, axes = ps.plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for ax, role in zip(axes, ("bat", "bowl")):
        d = tab[(tab.role == role) & (tab.target == "P_raw_test") & (tab.test == "IPL 2026")].iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.rho, color=[ps.PINK if m in ours else ps.BLUE for m in d.measure], height=0.6)
        ax.errorbar(d.rho, y, xerr=[d.rho - d.lo, d.hi - d.rho], fmt="none", ecolor=ps.INK, lw=1, capsize=3)
        ax.set_yticks(y, [NAMES.get(m, m) for m in d.measure])
        ax.axvline(0, color=ps.GREY, lw=1)
        for yi, (r, hi) in enumerate(zip(d.rho, d.hi)):
            ax.text(max(hi, 0) + 0.03, yi, f"{r:+.2f}", va="center", fontsize=8, color=ps.INK)
        ax.set_title(f"{'Batters' if role == 'bat' else 'Bowlers'} (n = {int(d.n.iloc[0])})")
        ax.set_xlabel("Spearman with IPL 2026 raw pressure outcome")
        ax.set_xlim(-1, 1.15)
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ps.PINK, label="our estimate"), Patch(color=ps.BLUE, label="traditional stat")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Predicting IPL 2026 pressure outcomes from data up to 2025", fontweight="bold", y=1.02)
    fig.text(0.5, -0.17, "Bars = Spearman correlation; whiskers = 95% player-bootstrap interval. Bowling stats negated so "
             "higher = better. One test season, ~20 players: every interval crosses zero.", ha="center",
             color=ps.MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    ps.plt.close(fig)


# --------------------------------------------------------------------------- run
def run(cfg: dict) -> None:
    header("STAGE 11 - VALIDATION")
    out = Path(cfg["paths"]["out"])
    tabs, figs = out / "tables", out / "figures"
    R = []
    rng = np.random.default_rng(cfg["seed"])
    b = load_balls(cfg)
    ratings = pd.read_parquet(interim(cfg, "player_ratings"))
    rup = pd.read_parquet(interim(cfg, "rup"))
    temporal = pd.read_parquet(interim(cfg, "temporal"))
    kv = json.load(open(tabs / "eb_k_values.json", encoding="utf-8"))
    kp = json.load(open(tabs / "process_k_values.json", encoding="utf-8"))
    prior = json.load(open(tabs / "consistency_prior.json", encoding="utf-8"))
    wj = json.load(open(tabs / "rup_weights.json", encoding="utf-8"))
    ch = wj["chosen"]
    metrics = {"chosen": ch, "search": wj["search"],
               "k_values": {r: {"outcome": {k_: kv[r][k_] for k_ in ("k_B", "k_P")},
                                "process": {k_: kp[r][k_] for k_ in ("k_B", "k_P")},
                                "clutch_sd_ci_per100": [100 * np.sqrt(x) for x in kv[r]["tau2_D_ci"]],
                                "process_gap_sd_ci_per100": [100 * np.sqrt(x) for x in kp[r]["tau2_D_ci"]]} for r in kv}}

    # d) model quality
    header("d) Model quality (Stages 4, 6, 6b)")
    wpm = json.load(open(tabs / "wp_metrics.json", encoding="utf-8"))
    metrics["wp"] = wpm
    for f_ in ["wp_calibration.png"]:
        check((figs / f_).exists(), f"figure {f_} present", R)
    for f_ in ["expected_calibration.csv", "process_calibration.csv", "process_calibration_season.csv"]:
        check((tabs / f_).exists(), f"table {f_} present", R)
    print("  " + json.dumps({k_: v for k_, v in wpm.items() if not isinstance(v, dict)}, default=str)[:600])

    # a) split-half
    w = ball_weights(b, cfg)
    sh = []
    for role in ("bat", "bowl"):
        header(f"a) Split-half reliability ({role}): odd vs even matches, Spearman-Brown")
        halves = split_half(b, cfg, role, w, kv[role], kp[role], (prior[role]["a"], prior[role]["b"]))
        full = ratings[ratings.role == role].set_index("player_id").pressure_balls.to_dict()
        tab = reliability_table(halves, full, rng, role)
        sh.append(tab)
        piv = tab.pivot(index="min_pressure_balls", columns="metric", values="spearman_brown")[SH_METRICS]
        piv.insert(0, "n", tab[tab.metric == "P_raw"].set_index("min_pressure_balls").n)
        print(piv.round(2).to_string())
    sh = pd.concat(sh, ignore_index=True)
    sh.to_csv(tabs / "validation_split_half.csv", index=False)
    fig_reliability(sh, figs / "reliability_curve.png")
    at = {"bat": 100, "bowl": 120}
    metrics["split_half_at_threshold"] = {
        r: sh[(sh.role == r) & (sh.min_pressure_balls == at[r])].set_index("metric")[
            ["n", "rho_halves", "spearman_brown", "sb_lo", "sb_hi"]].to_dict("index") for r in at}
    for r in at:
        d = sh[(sh.role == r) & (sh.min_pressure_balls == at[r])].set_index("metric")
        check(d.loc["proc_P_raw", "spearman_brown"] > d.loc["P_raw", "spearman_brown"],
              f"{r}: process pressure value more repeatable than outcome pressure value at the threshold "
              f"({d.loc['proc_P_raw', 'spearman_brown']:.2f} vs {d.loc['P_raw', 'spearman_brown']:.2f})", R)

    # b) temporal
    tt = []
    cut = pd.Timestamp(cfg["validation"]["train_until"])
    btr = b[pd.to_datetime(b.match_date) <= cut]
    for role in ("bat", "bowl"):
        header(f"b) Temporal test ({role}): <= {cfg['validation']['train_until']} -> IPL 2026 raw pressure outcome")
        t = temporal[temporal.role == role].copy()
        tab = temporal_eval(t, btr, role, ch[role]["a"], ch[role]["weights"], rng, "IPL 2026")
        tt.append(tab)
        print(tab[tab.target == "P_raw_test"][["measure", "n", "rho", "lo", "hi", "rup_minus_this_lo",
                                                "rup_minus_this_hi"]].round(3).to_string(index=False))
        print("  secondary target (IPL 2026 P*):")
        print(tab[tab.target == "P_star_test"][["measure", "rho", "lo", "hi"]].round(3).to_string(index=False))
        check(np.isfinite(tab.rho).all(), f"{role}: all temporal correlations defined", R)
        # rolling origin: more test players
        t2, _, btr2, bte2 = temporal_frames(b, cfg, role, train_until="2024-12-31", test_prefixes=("IPL 2025", "IPL 2026"))
        tab2 = temporal_eval(t2, btr2, role, ch[role]["a"], ch[role]["weights"], rng, "IPL 2025+2026 (train <= 2024)",
                             targets=("P_raw_test",))
        header(f"b2) Rolling origin ({role}): train <= 2024-12-31 -> IPL 2025 + 2026 raw pressure outcome")
        print(tab2[["measure", "n", "rho", "lo", "hi", "rup_minus_this_lo", "rup_minus_this_hi"]].round(3).to_string(index=False))
        tt.append(tab2)
    tt = pd.concat(tt, ignore_index=True)
    tt.to_csv(tabs / "validation_temporal.csv", index=False)
    fig_temporal(tt, figs / "temporal_validity.png")
    metrics["temporal"] = tt.to_dict("records")

    # c) rank stability
    boot = np.load(Path(cfg["paths"]["interim"]) / "bootstrap.npz")
    bootp = np.load(Path(cfg["paths"]["interim"]) / "bootstrap_process.npz")
    rs = []
    for role in ("bat", "bowl"):
        header(f"c) Rank stability ({role}): {boot[f'{role}_P_star'].shape[1]} match-bootstrap reps")
        s = rank_stability(role, rup, boot, bootp, ch[role]["a"], ch[role]["weights"])
        rs.append(s)
        r = rup[rup.role == role].merge(s, on=["player_id", "role"], how="left")
        top = r[r["rank"] <= 20].sort_values("rank")
        print(top[["rank", "name", "RUP", "p_top10", "median_rank", "rank_p10", "rank_p90"]].round(2).to_string(index=False))
        check(abs(s.p_top10.sum() - 10) < 1e-6, f"{role}: P(top 10) sums to 10", R)
        print(f"  players with P(top10) >= 0.5: {(s.p_top10 >= 0.5).sum()}; >= 0.8: {(s.p_top10 >= 0.8).sum()}")
    rs = pd.concat(rs, ignore_index=True)
    rs.to_csv(tabs / "validation_rank_stability.csv", index=False)
    rs.to_parquet(interim(cfg, "rank_stability"), index=False)
    with open(tabs / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
    print("\n  wrote validation_split_half.csv, validation_temporal.csv, validation_rank_stability.csv, "
          "validation_metrics.json, reliability_curve.png, temporal_validity.png")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 11: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 11 checks failed - review.")

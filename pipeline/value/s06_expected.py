"""Stage 6: context-only expectation models and wicket run-value.

Models (HistGradientBoosting, out-of-fold with the SAME match folds as Stage 4, no player identity):
  xR_bat   balls faced          runs off the bat        Poisson
  xO_bat   balls faced          striker dismissed (excl. retired out; non-striker run-outs excluded)
  xR_bowl  all deliveries       bowler_conceded         Poisson
  xW_bowl  all deliveries       bowler_wicket
  aux: xDot_bat, xBnd_bat (balls faced: 0 off the bat / four or six) for Stage 8 supporting stats
Wicket run-value: R(balls_left, wkts_before, env, tier) = expected runs remaining, fitted on
innings-1 balls only; RV = R(bl, w) - R(bl, w + 1), clipped >= 0 (R(bl, 10) = 0).

Outputs: interim/expected.parquet, interim/expected_folds.json,
         outputs/tables/expected_calibration.csv, rv_table.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_poisson_deviance

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.pressure.s04_wp_models import TIER_CODE

PHASE_CODE = {"powerplay": 0, "middle": 1, "death": 2}
BASE = ["innings", "over_num", "over_ball_c", "phase_code", "wkts_before", "balls_left", "score_before",
        "rrr", "runs_needed", "need_ratio", "wp_before", "env_rpb", "tier_code"]
FEATS = {
    "bat": BASE + ["opp_bowl_strength_rel", "opp_bowl_wpb_rel", "is_spin"],
    "bowl": BASE + ["opp_bat_strength_rel", "opp_bat_wpb_rel", "is_lhb"],
}
FORBIDDEN = {"target", "striker_id", "bowler_id", "non_striker_id", "batting_team_id", "bowling_team_id"}

MODELS = {  # name: (feature set, row filter, target, kind)
    "xR_bat": ("bat", "faced", "runs", "poisson"),
    "xO_bat": ("bat", "faced", "batter_out", "binary"),
    "xDot_bat": ("bat", "faced", "y_dot", "binary"),
    "xBnd_bat": ("bat", "faced", "y_bnd", "binary"),
    "xR_bowl": ("bowl", "all", "bowler_conceded", "poisson"),
    "xW_bowl": ("bowl", "all", "bowler_wicket", "binary"),
}


def prep(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["over_ball_c"] = b.over_ball_num.clip(upper=6)
    b["phase_code"] = b.phase.map(PHASE_CODE)
    b["tier_code"] = b.tier.map(TIER_CODE).astype(float)
    b["is_spin"] = (b.pace_or_spin == "Spin").astype(float).where(b.pace_or_spin.notna())
    b["is_lhb"] = (b.batting_type == "LHB").astype(float).where(b.batting_type.notna())
    b.loc[b.innings == 1, ["rrr", "runs_needed", "need_ratio"]] = np.nan
    b["y_dot"] = (b.runs == 0).astype(int)
    b["y_bnd"] = (b.is_four | b.is_six).astype(int)
    for c in ["batter_out", "bowler_wicket"]:
        b[c] = b[c].astype(int)
    return b


def _model(kind: str, ec: dict, seed: int):
    h = ec[kind]
    common = dict(max_iter=h["max_iter"], learning_rate=h["learning_rate"],
                  min_samples_leaf=h["min_samples_leaf"], max_leaf_nodes=h["max_leaf_nodes"],
                  random_state=seed, early_stopping=False)
    if kind == "poisson":
        return HistGradientBoostingRegressor(loss="poisson", **common)
    return HistGradientBoostingClassifier(**common)


def fit_oof(b, name, folds_k, ec, seed, train_mask, fold_log):
    fset, rows, target, kind = MODELS[name]
    feats = FEATS[fset]
    assert not (set(feats) & FORBIDDEN), f"forbidden feature in {name}"
    rmask = b.is_faced.values if rows == "faced" else np.ones(len(b), bool)
    X = b[feats].values
    y = b[target].values
    pred = np.full(len(b), np.nan)
    fold_log[name] = {}
    for f in range(folds_k):
        tr = rmask & train_mask & (b.fold_id.values != f)
        te = rmask & (b.fold_id.values == f)
        mdl = _model(kind, ec, seed + f).fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te]) if kind == "poisson" else mdl.predict_proba(X[te])[:, 1]
        fold_log[name][f] = sorted(int(x) for x in np.unique(b.fold_id.values[tr]))
    return pred


def _dl_unpack(p):
    k, d = p[0], p[1:11]                       # d[9] = Z0_9; d[j] = Z0_j - Z0_{j+1} >= 0
    return k, np.cumsum(d[::-1])[::-1]


def _dl_Z(u, w, k, z0):
    w = np.asarray(w).astype(int)
    wc = np.minimum(w, 9)
    return np.where(w >= 10, 0.0, z0[wc] * (1 - np.exp(-k * np.asarray(u, float) / z0[wc])))


def fit_rv(b, folds_k, cfg, seed, m):
    """Wicket run-value from a Duckworth-Lewis-form runs-remaining curve (innings 1 only, OOF).

    Z(u, w) = Z0(w) * (1 - exp(-k u / Z0(w))) in env units (runs / env_rpb), u = balls left,
    w = wickets lost; one shared scoring-rate k, Z0 non-increasing in w (so RV >= 0 by
    construction). A per-tier multiplicative factor (from training folds) carries `tier`.
    R = Z * env_rpb * tier_factor;  RV = R(u, w) - R(u, w + 1), R(u, 10) = 0.
    """
    from scipy.optimize import least_squares
    nr = set(m.loc[(m.result_type == "no_result") | m.data_suspect, "match_id"])
    i1 = b[(b.innings == 1) & ~b.match_id.isin(nr)].copy()
    i1["rem_u"] = (i1.inn1_total - i1.score_before) / i1.env_rpb
    fits = {}
    for f in range(folds_k):
        tr = i1[i1.fold_id != f]
        u, w, y = tr.balls_left.values, tr.wkts_before.values.astype(int), tr.rem_u.values

        def resid(p):
            k, z0 = _dl_unpack(p)
            return _dl_Z(u, w, k, z0) - y
        r = least_squares(resid, np.r_[1.5, np.full(10, 15.0)],
                          bounds=(np.r_[0.1, np.full(10, 0.01)], np.r_[5, np.full(10, 300)]))
        k, z0 = _dl_unpack(r.x)
        pred = _dl_Z(u, w, k, z0)
        tf = (pd.DataFrame({"t": tr.tier.values, "y": y, "p": pred}).groupby("t")
              .apply(lambda d: d.y.sum() / d.p.sum(), include_groups=False).to_dict())
        fits[f] = (k, z0, tf)
        if f == 0:
            print(f"  fold 0: k = {k:.3f}, Z0(w) = {np.round(z0, 1).tolist()}")
            print(f"  fold 0 tier factors: { {t: round(v, 3) for t, v in tf.items()} }")
    r_now, r_next = np.zeros(len(b)), np.zeros(len(b))
    w0 = b.wkts_before.values.astype(int)
    for f, (k, z0, tf) in fits.items():
        s = b.fold_id.values == f
        scale = b.env_rpb.values[s] * b.tier[s].map(tf).fillna(1.0).values
        r_now[s] = _dl_Z(b.balls_left.values[s], w0[s], k, z0) * scale
        r_next[s] = _dl_Z(b.balls_left.values[s], w0[s] + 1, k, z0) * scale
    return r_now - r_next, r_now, fits


def _calib(b, pred_col, obs_col, mask, kind, tol, slack):
    d = b[mask]
    g = d.groupby(["innings", "phase", "comp_group"]).agg(n=(obs_col, "size"), obs=(obs_col, "mean"),
                                                         pred=(pred_col, "mean"), var=(obs_col, "var"))
    g["ratio"] = g.pred / g.obs
    g["se"] = np.sqrt(g["var"] / g.n)
    g["within_tol"] = (g.ratio - 1).abs() <= tol
    g["within_se"] = (g.obs - g.pred).abs() <= slack * g.se
    g["pass"] = g.within_tol | g.within_se
    return g.drop(columns="var").reset_index()


def run(cfg: dict) -> None:
    header("STAGE 6 - EXPECTED OUTCOMES AND WICKET RUN-VALUE")
    b = pd.read_parquet(interim(cfg, "leverage"))
    m = pd.read_parquet(interim(cfg, "matches"))
    folds = pd.read_parquet(interim(cfg, "folds"))
    ec, seed, K = cfg["expected"], cfg["seed"], cfg["wp"]["n_folds"]
    tabs = Path(cfg["paths"]["out"]) / "tables"
    R = []

    chk = b[["match_id", "fold_id"]].drop_duplicates().merge(folds, on="match_id", suffixes=("", "_s4"))
    check((chk.fold_id == chk.fold_id_s4).all() and len(chk) == b.match_id.nunique(),
          "using the same match folds as Stage 4", R)
    b = prep(b)
    train_mask = ~b.match_id.isin(m.loc[m.data_suspect, "match_id"]).values
    print(f"features (batter models): {FEATS['bat']}")
    print(f"features (bowler models): {FEATS['bowl']}")
    check(not any(set(v) & FORBIDDEN for v in FEATS.values()),
          "no raw target / player or team identity features", R)

    fold_log = {}
    for name in MODELS:
        print(f"  fitting {name} ...", flush=True)
        b[name] = fit_oof(b, name, K, ec, seed, train_mask, fold_log)
    with open(Path(cfg["paths"]["interim"]) / "expected_folds.json", "w") as f:
        json.dump(fold_log, f, indent=1)
    oof_ok = all(int(f) not in tr for mdl in fold_log.values() for f, tr in mdl.items())
    check(oof_ok, "out-of-fold: no fold's predictions come from a model trained on that fold", R)
    faced = b.is_faced.values
    check(b.loc[faced, ["xR_bat", "xO_bat"]].notna().all().all() and b[["xR_bowl", "xW_bowl"]].notna().all().all(),
          "every eligible ball has OOF expectations", R)

    # ---------------- skill vs naive baselines ----------------
    header("Model skill (OOF) vs a phase x innings x wickets-bucket mean baseline")
    cell = ["innings", "phase", "wkt_bucket"]
    for name, (fset, rows, target, kind) in MODELS.items():
        rm = faced if rows == "faced" else np.ones(len(b), bool)
        d = b[rm & train_mask]
        base = d.groupby(cell)[target].transform("mean").clip(1e-6, None)
        if kind == "poisson":
            s_m, s_b = mean_poisson_deviance(d[target], d[name].clip(1e-6)), mean_poisson_deviance(d[target], base)
            lab = "Poisson deviance"
        else:
            s_m, s_b = log_loss(d[target], d[name].clip(1e-6, 1 - 1e-6)), log_loss(d[target], base.clip(upper=1 - 1e-6))
            lab = "log-loss"
        print(f"  {name:9s} {lab:16s} model {s_m:.4f} vs baseline {s_b:.4f} ({(1 - s_m / s_b):+.2%} better)")

    # ---------------- calibration by phase x innings x comp_group ----------------
    header("Calibration: predicted vs actual by innings x phase x comp_group")
    cal = []
    for pred_col, obs_col, mask, kind, tol in [
            ("xR_bat", "runs", faced, "runs", ec["tol_runs"]),
            ("xR_bowl", "bowler_conceded", np.ones(len(b), bool), "runs", ec["tol_runs"]),
            ("xO_bat", "batter_out", faced, "wkts", ec["tol_wkts"]),
            ("xW_bowl", "bowler_wicket", np.ones(len(b), bool), "wkts", ec["tol_wkts"])]:
        g = _calib(b, pred_col, obs_col, mask & train_mask, kind, tol, ec["se_slack"])
        g.insert(0, "model", pred_col)
        cal.append(g)
        nt, ns, n = g.within_tol.sum(), (g["pass"] & ~g.within_tol).sum(), len(g)
        worst = g.loc[(g.ratio - 1).abs().idxmax()]
        print(f"  {pred_col:8s} within ±{tol:.0%}: {nt}/{n} cells; within {ec['se_slack']:.0f} SE only: {ns}; "
              f"fail: {(~g['pass']).sum()} | overall pred/actual {b.loc[mask, pred_col].mean() / b.loc[mask, obs_col].mean():.4f} "
              f"| worst cell {worst.innings}/{worst.phase}/{worst.comp_group} ratio {worst.ratio:.3f} (n={worst.n:,})")
        check(g["pass"].all(), f"{pred_col}: every cell within ±{tol:.0%} or within {ec['se_slack']:.0f} sampling SE", R)
    cal = pd.concat(cal)
    cal.to_csv(tabs / "expected_calibration.csv", index=False)
    show = cal[cal.model.isin(["xR_bat", "xO_bat"])].copy()
    show["obs"] = np.where(show.model == "xO_bat", show.obs * 100, show.obs)
    show["pred"] = np.where(show.model == "xO_bat", show.pred * 100, show.pred)
    print("\n  batter models by cell (runs per ball / dismissal %):")
    print(show[["model", "innings", "phase", "comp_group", "n", "obs", "pred", "ratio", "pass"]].round(3).to_string(index=False))

    # ---------------- wicket run-value ----------------
    header("Wicket run-value RV = R(balls_left, w) - R(balls_left, w+1)  (innings-1 fit, OOF)")
    b["RV"], b["R_remaining"], dl_fits = fit_rv(b, K, cfg, seed, m)
    check((b.RV >= 0).all(), f"RV non-negative (min {b.RV.min():.3f})", R)
    k0, z00, _ = dl_fits[0]
    U = np.array([114, 90, 60, 30, 12, 6, 1])
    grid = pd.DataFrame({w: (_dl_Z(U, w, k0, z00) - _dl_Z(U, w + 1, k0, z00)) * 9.0 / 6 for w in range(10)}, index=U)
    grid.index.name = "balls_left"
    print("  fold-0 curve: RV in runs at a 9.0 RPO environment, by balls left x wickets lost:")
    print(grid.round(1).to_string())
    rv = b.groupby(["phase", "wkts_before"]).RV.agg(["mean", "size"]).reset_index()
    rv_tab = rv.pivot(index="wkts_before", columns="phase", values="mean").reindex(columns=list(cfg["phases"]))
    rv_n = rv.pivot(index="wkts_before", columns="phase", values="size").reindex(columns=list(cfg["phases"]))
    rv_tab = rv_tab.where(rv_n >= 200)
    print("  mean RV (runs) by wickets lost before the ball x phase (cells with >= 200 balls):")
    print(rv_tab.round(2).to_string())
    by_over = b[b.innings == 1].groupby("over_num").RV.mean()
    print("  mean RV by over (innings 1):", by_over.round(1).to_dict())
    rv.to_csv(tabs / "rv_table.csv", index=False)
    print(f"  RV distribution: p5 {b.RV.quantile(.05):.2f}, p25 {b.RV.quantile(.25):.2f}, median {b.RV.median():.2f}, "
          f"p75 {b.RV.quantile(.75):.2f}, p95 {b.RV.quantile(.95):.2f}, max {b.RV.max():.2f}")
    pp_top = rv_tab.loc[[0, 1], "powerplay"].mean()
    mean_rv = b.loc[b.is_wicket, "RV"].mean()
    print(f"  mean RV on balls where a wicket actually fell: {mean_rv:.2f} runs; on all balls: {b.RV.mean():.2f}")
    check(3 <= mean_rv <= 12 and 8 <= pp_top <= 15,
          f"RV in the plausible 3-12 run range: mean on actual wickets {mean_rv:.1f}; powerplay 0-1 down {pp_top:.1f}", R)
    check(pp_top > rv_tab.loc[:, "death"].max(),
          f"RV largest early with few wickets down (powerplay 0-1 down {pp_top:.1f} > any death cell "
          f"{rv_tab.loc[:, 'death'].max():.1f})", R)
    mono = (by_over.diff().dropna() <= 0.15).mean()
    print(f"  share of overs where mean RV does not rise vs previous over (innings 1): {mono:.0%}")

    # expected runs-remaining sanity vs actual (innings 1, full innings)
    i1 = b[(b.innings == 1) & (b.max_balls == 120) & (b.result_type != "no_result")]
    rr = i1.groupby("over_num").agg(pred=("R_remaining", "mean"), act=("inn1_total", "mean"), sb=("score_before", "mean"))
    rr["act_rem"] = rr.act - rr.sb
    print("  innings-1 runs remaining, predicted vs actual (overs 1, 6, 11, 16, 20):",
          {o: (round(rr.loc[o, "pred"], 1), round(rr.loc[o, "act_rem"], 1)) for o in (1, 6, 11, 16, 20)})

    keep = ["xR_bat", "xO_bat", "xDot_bat", "xBnd_bat", "xR_bowl", "xW_bowl", "RV", "R_remaining",
            "over_ball_c", "phase_code", "tier_code", "is_spin", "is_lhb", "y_dot", "y_bnd"]
    b.to_parquet(interim(cfg, "expected"), index=False)
    print(f"\n  wrote expected.parquet ({len(b):,} rows; new columns: {keep}), tables expected_calibration.csv, rv_table.csv")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 6: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 6 checks failed - STOP and review.")

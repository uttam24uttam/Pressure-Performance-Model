"""Stage 4: win-probability models (WP1 first innings, WP2 chase), out-of-fold by match.

Outputs: interim/wp.parquet (env.parquet + fold_id + wp_before)
         interim/folds.parquet (match_id -> fold_id, reused by Stage 6)
         interim/wp_models.pkl (per-fold models + calibrators, reused by Stage 5 via wp())
         outputs/figures/wp_calibration.png, outputs/tables/wp_metrics.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.pressure.outcomes import add_buckets, classify_outcome, outcome_probs, endgame_dp
from pipeline.common import plotstyle as ps

TIER_CODE = {"T20I_ASSOC": 0, "EMERGING": 1, "T20I_FM": 2, "IPL": 3}

FEATURES = {
    # need_ratio = runs_needed / DLS-style resources (expected runs from this balls/wickets state).
    # Added feature: lets the regularised model reach the extremes (41 off 6: 9% -> 1.6%).
    # Unconstrained `target` dropped: it made chase-start WP non-monotone in the target
    # (24 reversals across folds) and dropping it improved OOF log-loss 0.3949 -> 0.3905.
    2: [("need_ratio", -1), ("runs_needed", -1), ("balls_left", 1), ("wkts_in_hand", 1), ("rrr", -1),
        ("env_rpb", 0), ("tier_code", 0)],
    # Unconstrained `crr` dropped: with balls fixed it duplicates score and let the model
    # overfit sparse early cells (8 runs after 2 overs = +26 WP points; over-1 mean LI 2.93).
    1: [("score_before", 1), ("wkts_in_hand", 1), ("balls_left", 1),
        ("env_par1", 0), ("max_balls", 0), ("tier_code", 0)],
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["wkts_in_hand"] = 10 - df["wkts_before"]
    df["tier_code"] = df["tier"].map(TIER_CODE).astype(float)
    if "runs_needed" in df:
        df["rrr"] = df["runs_needed"] / df["balls_left"].clip(lower=1) * 6
    return df


def assign_folds(match_ids, n_folds: int, seed: int) -> pd.DataFrame:
    ids = np.array(sorted(set(match_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    return pd.DataFrame({"match_id": ids[perm], "fold_id": np.arange(len(ids)) % n_folds})


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return e


def fit_resources(df: pd.DataFrame, seed: int) -> HistGradientBoostingRegressor:
    """Runs remaining in a full first innings (in env_rpb units) ~ balls_left(+), wkts_in_hand(+)."""
    mdl = HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                        min_samples_leaf=200, monotonic_cst=[1, 1], random_state=seed)
    return mdl.fit(df[["balls_left", "wkts_in_hand"]].values, df["rem_units"].values)


def add_need_ratio(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    df = df.copy()
    res = np.full(len(df), np.nan)
    for f in np.unique(df.fold_id.values):
        s = df.fold_id.values == f
        res[s] = models["res"][f].predict(df.loc[s, ["balls_left", "wkts_in_hand"]].values)
    df["res_runs"] = res * df.env_rpb.values
    df["need_ratio"] = df.runs_needed / df.res_runs.clip(lower=0.5)
    return df


def _raw_predict(models, inn, X, fold):
    mdl, iso = models[inn][fold]
    p = mdl.predict_proba(X)[:, 1]
    return iso.predict(p) if iso is not None else p


def wp(state: pd.DataFrame, models: dict) -> np.ndarray:
    """Batting-side win probability for arbitrary (possibly hypothetical/terminal) states.

    Required columns: innings, fold_id, score_before, wkts_before, balls_left, max_balls, tier,
    env_rpb, env_par1; innings 2 also target (runs_needed is derived). Terminal handling:
      inn 2: runs_needed <= 0 -> 1; out of balls/wickets: tie -> 0.5, else 0.
      inn 1: innings over -> 1 - WP2(start of chase, target = score + 1, max balls = inn2 max).
    """
    s = state.copy()
    out = np.full(len(s), np.nan)
    i2 = (s.innings == 2).values
    if i2.any():
        t = add_features(s[i2].assign(runs_needed=lambda d: d.target - d.score_before))
        t = add_need_ratio(t, models)
        done_bat = (t.runs_needed <= 0).values
        done_out = ((t.wkts_before >= 10) | (t.balls_left <= 0)).values & ~done_bat
        p = np.full(len(t), np.nan)
        p[done_bat] = 1.0
        p[done_out] = np.where(t.runs_needed.values[done_out] == 1, 0.5, 0.0)
        live = ~(done_bat | done_out)
        X = t[[f for f, _ in FEATURES[2]]].values
        for f in np.unique(t.fold_id.values[live]):
            sel = live & (t.fold_id.values == f)
            p[sel] = _raw_predict(models, 2, X[sel], f)
        # endgame: blend in the exact DP for the last few balls (weights chosen in run())
        dw = models.get("dp_weights", {})
        if dw:
            for bl_, wgt in dw.items():
                sel = live & (t.balls_left.values == bl_)
                if wgt > 0 and sel.any():
                    dpv = dp_lookup(models, t.fold_id.values[sel], t.balls_left.values[sel],
                                    t.wkts_in_hand.values[sel], t.runs_needed.values[sel])
                    p[sel] = (1 - wgt) * p[sel] + wgt * dpv
        out[i2] = p
    i1 = (s.innings == 1).values
    if i1.any():
        t = add_features(s[i1])
        over = ((t.wkts_before >= 10) | (t.balls_left <= 0)).values
        p = np.full(len(t), np.nan)
        X = t[[f for f, _ in FEATURES[1]]].values
        live = ~over
        for f in np.unique(t.fold_id.values[live]):
            sel = live & (t.fold_id.values == f)
            p[sel] = _raw_predict(models, 1, X[sel], f)
        if over.any():
            p[over] = _chase_start_value(t[over], t.score_before.values[over] + 1, models)
        nb = models.get("inn1_blend_balls", 0)
        if nb:
            # window = min(nb, scheduled balls) so rain-shortened innings blend over their own length
            win = np.minimum(nb, t.max_balls.values)
            bl = live & (t.balls_left.values < win)
            if bl.any():
                term = blend_target(t[bl], models)
                lam = 1 - t.balls_left.values[bl] / win[bl]
                p[bl] = (1 - lam) * p[bl] + lam * term
        out[i1] = p
    return out


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()
CS_TIERS = ["T20I_FM", "T20I_ASSOC", "EMERGING"]        # IPL is the baseline


def _per_fold(models, key, folds, X):
    out = np.zeros(len(folds))
    for f in np.unique(folds):
        s_ = folds == f
        out[s_] = models[key][f].predict(X[s_])
    return out


def _cs_design(models, folds, mb2, env_rpb, tier, target) -> np.ndarray:
    """Chase-start design: log(target / resources of a full chase), tier dummies, IPL interaction."""
    res0 = _per_fold(models, "res", folds, np.c_[mb2, np.full(len(folds), 10)])
    lnr = np.log(np.maximum(np.asarray(target, float), 1.0) / (res0 * env_rpb))
    tier = np.asarray(tier)
    dums = np.stack([(tier == t).astype(float) for t in CS_TIERS], 1)
    return np.c_[lnr, dums, lnr * (tier == "IPL")]


def chase_start_wp(models, folds, mb2, env_rpb, tier, target) -> np.ndarray:
    """P(chasing side wins) at the start of a chase - smooth out-of-fold logistic (per fold)."""
    X = _cs_design(models, folds, mb2, env_rpb, tier, target)
    out = np.zeros(len(folds))
    for f in np.unique(folds):
        s_ = folds == f
        out[s_] = models["cs"][f].predict_proba(X[s_])[:, 1]
    return out


def blend_target(o: pd.DataFrame, models: dict) -> np.ndarray:
    """Projection-integrated value for a live first-innings state s (match in fold f):

        T(s) = E_X[ 1 - CS_f(target = X + 1) ],  X ~ Normal(proj, sd_f(balls_left, wkts_in_hand)),
        proj = score_before + res_f(balls_left, wkts_in_hand) * env_rpb

    CS_f is the chase-start logistic, res_f the resources model, sd_f the out-of-fold spread of the
    final total around proj. All three are fitted on folds != f, so T is out-of-fold.
    Expectation by 15-point Gauss-Hermite quadrature.
    """
    folds = o.fold_id.values
    X = o[["balls_left", "wkts_in_hand"]].values
    proj = o.score_before.values + _per_fold(models, "res", folds, X) * o.env_rpb.values
    sd = np.sqrt(np.clip(_per_fold(models, "sd", folds, X), 0, None)) * o.env_rpb.values
    mb2 = (o["max_balls_inn2"] if "max_balls_inn2" in o else o["max_balls"]).values
    v = np.zeros(len(o))
    for x, w in zip(GH_X, GH_W):
        tgt = np.maximum(proj + x * sd, 0) + 1
        v += w * (1 - chase_start_wp(models, folds, mb2, o.env_rpb.values, o.tier.values, tgt))
    return v


def dp_lookup(models, folds, balls_left, wkts_in_hand, runs_needed) -> np.ndarray:
    out = np.empty(len(folds))
    for f in np.unique(folds):
        s_ = folds == f
        V = models["dp"][f]
        r = np.clip(runs_needed[s_], 0, V.shape[2] - 1).astype(int)
        v = V[balls_left[s_].astype(int), wkts_in_hand[s_].astype(int), r]
        out[s_] = np.where(runs_needed[s_] >= V.shape[2], 0.0, v)
    return out


def _chase_start_value(o: pd.DataFrame, target, models) -> np.ndarray:
    """Innings-1 batting side's WP at the innings break = 1 - CS(target) (same smooth curve as T)."""
    mb2 = (o["max_balls_inn2"] if "max_balls_inn2" in o else o["max_balls"]).values
    return 1 - chase_start_wp(models, o.fold_id.values, mb2, o.env_rpb.values, o.tier.values, target)


def _fit(X, y, mono, cfg, seed, inn):
    w = cfg["wp"][f"wp{inn}"]
    mdl = HistGradientBoostingClassifier(
        max_iter=w["max_iter"], learning_rate=w["learning_rate"], min_samples_leaf=w["min_samples_leaf"],
        max_leaf_nodes=w["max_leaf_nodes"], l2_regularization=w["l2"],
        monotonic_cst=mono, random_state=seed, early_stopping=False)
    return mdl.fit(X, y)


def run(cfg: dict) -> None:
    header("STAGE 4 - WIN-PROBABILITY MODELS")
    ps.apply()
    b = pd.read_parquet(interim(cfg, "env"))
    m = pd.read_parquet(interim(cfg, "matches"))
    seed, K = cfg["seed"], cfg["wp"]["n_folds"]
    folds = assign_folds(b.match_id.unique(), K, seed)
    folds.to_parquet(interim(cfg, "folds"), index=False)
    b = b.merge(folds, on="match_id", how="left")
    b["max_balls_inn2"] = b.match_id.map(m.set_index("match_id").max_balls_inn2)
    b = add_features(b)

    excl = m.loc[(m.result_type != "decisive") | m.data_suspect | m.dls_par_finish, "match_id"]
    train_ok = ~b.match_id.isin(excl) & b.bat_team_won.notna()
    print(f"training matches: {b.loc[train_ok, 'match_id'].nunique()} of {b.match_id.nunique()} "
          f"(excluded: NR/tie/suspect/DLS-par-finish {len(excl)})")

    models = {1: {}, 2: {}, "res": {}}
    # resources model per fold: full (unrevised, completed) first innings of training matches
    nr = m.loc[m.result_type == "no_result", "match_id"]
    i1 = b[(b.innings == 1) & (b.max_balls == 120) & ~b.match_id.isin(nr) & ~b.match_id.isin(
        m.loc[m.data_suspect, "match_id"])].copy()
    i1["rem_units"] = (i1.inn1_total - i1.score_before) / i1.env_rpb
    for f in range(K):
        models["res"][f] = fit_resources(i1[i1.fold_id != f], seed + f)
    # spread of the final total around the projection: residuals use each row's own (OOF) res model
    i1["proj"] = i1.score_before + _per_fold(models, "res", i1.fold_id.values,
                                             i1[["balls_left", "wkts_in_hand"]].values) * i1.env_rpb
    i1["resid2_units"] = ((i1.inn1_total - i1.proj) / i1.env_rpb) ** 2
    models["sd"] = {}
    for f in range(K):
        tr = i1[i1.fold_id != f]
        models["sd"][f] = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, min_samples_leaf=500, monotonic_cst=[1, 0],
            random_state=seed + f).fit(tr[["balls_left", "wkts_in_hand"]].values, tr.resid2_units.values)
    # chase-start logistic: first ball of each training chase
    first = b[(b.innings == 2) & (b.innings_ball_num == 1) & train_ok]
    models["cs"] = {}
    for f in range(K):
        tr = first[first.fold_id != f]
        X = _cs_design(models, tr.fold_id.values * 0 + f, tr.max_balls.values, tr.env_rpb.values,
                       tr.tier.values, tr.target.values)
        models["cs"][f] = LogisticRegression(C=10.0, max_iter=1000).fit(X, tr.bat_team_won.values.astype(int))
    pcs = chase_start_wp(models, first.fold_id.values, first.max_balls.values, first.env_rpb.values,
                         first.tier.values, first.target.values)
    print(f"chase-start logistic (OOF, first chase ball, n={len(first)}): log-loss "
          f"{log_loss(first.bat_team_won.astype(int), pcs):.4f}, AUC {roc_auc_score(first.bat_team_won, pcs):.4f}")
    sd_demo = [float(np.sqrt(models['sd'][0].predict([[bl, w]]))[0] * 9.88 / 6) for bl, w in
               [(120, 10), (60, 8), (24, 6), (6, 5), (1, 5)]]
    print("sd of final first-innings total (IPL 2026 env, runs) at balls_left 120/60/24/6/1: "
          + ", ".join(f"{v:.1f}" for v in sd_demo))
    b = add_need_ratio(b, models)       # NaN in innings 1 (runs_needed NaN) - unused there
    raw_oof = np.full(len(b), np.nan)
    for inn in (1, 2):
        feats = [f for f, _ in FEATURES[inn]]
        mono = [s for _, s in FEATURES[inn]]
        rows_inn = (b.innings == inn).values
        X = b[feats].values
        for f in range(K):
            tr = rows_inn & train_ok.values & (b.fold_id.values != f)
            te = rows_inn & (b.fold_id.values == f)
            mdl = _fit(X[tr], b.bat_team_won.values[tr].astype(int), mono, cfg, seed + f, inn)
            models[inn][f] = (mdl, None)
            raw_oof[te] = mdl.predict_proba(X[te])[:, 1]

    # ---- calibration decision on OOF predictions of training-eligible rows
    R = []
    metrics = {}
    cal_mode = cfg["wp"]["calibration"]
    oof = raw_oof.copy()
    cal_info = {}
    for inn in (1, 2):
        ev = (b.innings == inn).values & train_ok.values
        y, p = b.bat_team_won.values[ev].astype(int), raw_oof[ev]
        e_raw = ece(y, p)
        use_iso = cal_mode == "isotonic" or (cal_mode == "auto" and e_raw >= 0.03)
        cal_info[inn] = (e_raw, use_iso)
        if use_iso:
            feats = [f for f, _ in FEATURES[inn]]
            for f in range(K):
                cal = ev & (b.fold_id.values != f)       # calibrate on other folds' OOF preds
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                    raw_oof[cal], b.bat_team_won.values[cal])
                models[inn][f] = (models[inn][f][0], iso)
                te = (b.innings == inn).values & (b.fold_id.values == f)
                oof[te] = iso.predict(raw_oof[te])
    # ---- endgame DP (chase, last few balls): outcome probabilities per fold from other folds only
    eg = cfg["wp"]["endgame_dp"]
    bo = add_buckets(b)
    bo["outcome"] = classify_outcome(bo)
    alpha = cfg["pressure"]["outcome_smoothing_alpha"]
    models["dp"] = {f: endgame_dp(outcome_probs(bo[bo.fold_id != f], alpha), max_b=eg["max_balls_left"])
                    for f in range(K)}
    header("Endgame DP weight selection (chase, OOF log-loss)")
    models["dp_weights"] = {}
    gbm_only = wp(b[(b.innings == 2)], {**models, "inn1_blend_balls": 0})
    i2rows = b[b.innings == 2]
    for bl_, grid in sorted(eg["grid"].items()):
        sel = (i2rows.balls_left.values == bl_) & train_ok.values[(b.innings == 2).values] & \
              (i2rows.runs_needed.values > 0)
        y = i2rows.bat_team_won.values[sel].astype(int)
        g = gbm_only[sel]
        dpv = dp_lookup(models, i2rows.fold_id.values[sel], i2rows.balls_left.values[sel],
                        (10 - i2rows.wkts_before.values[sel]), i2rows.runs_needed.values[sel])
        res_ = {w_: log_loss(y, np.clip((1 - w_) * g + w_ * dpv, 1e-4, 1 - 1e-4), labels=[0, 1]) for w_ in grid}
        best = min(res_, key=res_.get)
        models["dp_weights"][int(bl_)] = float(best)
        print(f"  {bl_} ball(s) left (n={sel.sum()}): " +
              ", ".join(f"w={w_:.2f}: LL {v:.4f}" for w_, v in res_.items()) + f"  -> chosen w={best}")

    # final stored predictions come from wp() itself (adds the innings-1 terminal blend and the
    # endgame DP), so that stored and hypothetical states are scored by exactly the same function
    models["inn1_blend_balls"] = cfg["wp"]["inn1_blend_balls"]
    oof_unblended = oof.copy()
    oof = wp(b, models)
    nb_rows = (b.innings == 1).values & (b.balls_left.values < models["inn1_blend_balls"])
    ev1 = nb_rows & train_ok.values
    print(f"  innings-1 blend over last {models['inn1_blend_balls']} balls: log-loss on blended rows "
          f"{log_loss(b.bat_team_won.values[ev1].astype(int), oof_unblended[ev1]):.4f} (unblended) -> "
          f"{log_loss(b.bat_team_won.values[ev1].astype(int), oof[ev1]):.4f} (blended), n={ev1.sum():,}")
    for inn in (1, 2):
        ev = (b.innings == inn).values & train_ok.values
        y = b.bat_team_won.values[ev].astype(int)
        p = oof[ev]
        metrics[f"WP{inn}"] = dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                                   logloss=log_loss(y, p), ece=ece(y, p), ece_raw=cal_info[inn][0],
                                   isotonic=bool(cal_info[inn][1]), n_balls=int(ev.sum()),
                                   base_rate=float(y.mean()))
    b["wp_before"] = oof

    header("Acceptance")
    for k_, v in metrics.items():
        print(f"  {k_}: AUC {v['auc']:.4f} | Brier {v['brier']:.4f} | logloss {v['logloss']:.4f} | "
              f"ECE {v['ece']:.4f} (raw {v['ece_raw']:.4f}, isotonic={v['isotonic']}) | "
              f"n={v['n_balls']:,} | base rate {v['base_rate']:.3f}")
    check(metrics["WP2"]["auc"] >= 0.85, f"WP2 OOF AUC >= 0.85 ({metrics['WP2']['auc']:.4f})", R)
    check(metrics["WP2"]["ece"] < 0.03, f"WP2 ECE < 0.03 ({metrics['WP2']['ece']:.4f})", R)
    print(f"  [INFO] WP1 AUC {metrics['WP1']['auc']:.4f} (typical range ~0.70-0.80; first-innings states carry "
          f"little outcome information early - reported as a limitation, not a gate)")
    check(metrics["WP1"]["ece"] < 0.03, f"WP1 ECE < 0.03 ({metrics['WP1']['ece']:.4f})", R)
    check(b.wp_before.notna().all(), "every ball has an out-of-fold wp_before", R)

    # out-of-fold property: a ball's model never trained on its own fold
    check(b.groupby("match_id").fold_id.nunique().max() == 1, "each match lives in exactly one fold", R)

    # per comp_group / phase breakdown
    print("\n  OOF AUC by comp_group (WP2 / WP1):")
    for cg, g in b[train_ok].groupby("comp_group"):
        a = [roc_auc_score(g[g.innings == i].bat_team_won, g[g.innings == i].wp_before) for i in (2, 1)]
        print(f"    {cg:9s} {a[0]:.3f} / {a[1]:.3f}  ({g.match_id.nunique()} matches)")
    print("  WP2 AUC by phase:", {ph: round(roc_auc_score(g.bat_team_won, g.wp_before), 3)
                                 for ph, g in b[train_ok & (b.innings == 2)].groupby("phase")})

    # wp() reproduces OOF predictions & sensible behaviour
    smp = b.sample(5000, random_state=seed)
    rep = wp(smp, models)
    check(np.allclose(rep, smp.wp_before.values, atol=1e-9), "wp() reproduces stored OOF predictions", R)
    first2 = b[(b.innings == 2) & (b.innings_ball_num == 1) & train_ok]
    print(f"\n  mean WP at first chase ball {first2.wp_before.mean():.3f} vs actual chase win rate "
          f"{first2.bat_team_won.mean():.3f}")
    demo = pd.DataFrame({"innings": 2, "fold_id": 0, "score_before": [0, 150, 150, 150, 150, 170, 180, 189],
                         "wkts_before": [0, 5, 5, 8, 5, 5, 5, 9], "balls_left": [120, 24, 12, 12, 6, 6, 6, 1],
                         "max_balls": 120, "tier": "IPL", "env_rpb": 9.7 / 6, "env_par1": 192, "target": 191})
    demo["wp"] = wp(demo, models)
    print("  IPL 2026-environment chase of 191, example states:")
    print(demo[["score_before", "wkts_before", "balls_left", "wp"]].assign(
        need=lambda d: 191 - d.score_before).round(3).to_string(index=False))
    # monotonicity spot-checks on real states
    t = b[(b.innings == 2)].sample(3000, random_state=seed)
    w0 = wp(t, models)
    w_run = wp(t.assign(score_before=t.score_before + 4), models)
    w_wkt = wp(t.assign(wkts_before=(t.wkts_before + 1).clip(upper=10)), models)
    check((w_run >= w0 - 1e-6).all() and (w_wkt <= w0 + 1e-6).all(),
          "WP2 monotone: +4 runs never lowers WP, +1 wicket never raises it (3,000 sampled states)", R)
    t1 = b[(b.innings == 1)].sample(3000, random_state=seed)
    v0 = wp(t1, models)
    v_wkt = wp(t1.assign(wkts_before=(t1.wkts_before + 1).clip(upper=10)), models)
    rise = (v_wkt - v0).max()
    check(rise < 0.01, f"WP1 (incl. terminal blend): +1 wicket never raises WP by >0.01 "
          f"(max {rise:.4f}; share non-increasing {(v_wkt <= v0 + 1e-9).mean():.3f})", R)

    header("WP1 smoothness (3,000 sampled live first-innings states)")
    print("  Blend target for a live first-innings state s in fold f (lambda = 1 - balls_left / "
          "min(120, max_balls)):\n"
          "    T(s) = E_X[1 - CS_f(target = X + 1)],  X ~ Normal(proj, sd_f(balls_left, wkts_in_hand)),\n"
          "    proj = score_before + res_f(balls_left, wkts_in_hand) x env_rpb\n"
          "  CS_f = chase-start logistic, res_f = resources model, sd_f = spread of the final total;\n"
          "  all fitted on folds != f -> T is out-of-fold. Innings-over states use 1 - CS_f(score + 1).")
    v1 = wp(t1.assign(score_before=t1.score_before + 1), models)
    v4 = wp(t1.assign(score_before=t1.score_before + 4), models)
    d1 = np.abs(v1 - v0)
    d4 = np.abs(v4 - v0) / 4
    sm = dict(d1_p50=float(np.median(d1)), d1_p99=float(np.quantile(d1, .99)), d1_max=float(d1.max()),
              d4_p50=float(np.median(d4)), d4_p99=float(np.quantile(d4, .99)), d4_max=float(d4.max()),
              n_jumps_010=int((d1 >= 0.10).sum()))
    print(f"  |WP(s+1) - WP(s)|    : p50 {sm['d1_p50']:.4f}  p99 {sm['d1_p99']:.4f}  max {sm['d1_max']:.4f}")
    print(f"  |WP(s+4) - WP(s)| / 4: p50 {sm['d4_p50']:.4f}  p99 {sm['d4_p99']:.4f}  max {sm['d4_max']:.4f}")
    check(sm["d1_p99"] <= 0.03 and sm["n_jumps_010"] == 0,
          f"no step artifacts: p99 per-run change {sm['d1_p99']:.4f} <= 0.03 and "
          f"{sm['n_jumps_010']} single-run jumps >= 0.10", R)
    g = pd.DataFrame({"innings": 1, "fold_id": 0, "score_before": np.arange(0, 31), "wkts_before": 0,
                      "balls_left": 108.0, "max_balls": 120.0, "tier": "IPL", "env_rpb": 9.88 / 6,
                      "env_par1": 195.8, "max_balls_inn2": 120.0})
    print("  WP1 after 12 balls, 0 down, IPL 2026 env, score 0..30:")
    print("   ", " ".join(f"{s_}:{v:.2f}" for s_, v in zip(g.score_before, wp(g, models))))
    metrics["WP1_smoothness"] = sm

    # ---- figure
    fig, axes = plt_subplots()
    for ax, inn in zip(axes, (1, 2)):
        ev = (b.innings == inn).values & train_ok.values
        y, p = b.bat_team_won.values[ev], b.wp_before.values[ev]
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        mp = [p[idx == i].mean() for i in range(10) if (idx == i).any()]
        my = [y[idx == i].mean() for i in range(10) if (idx == i).any()]
        ax.plot([0, 1], [0, 1], color=ps.GREY, lw=1, ls="--", label="perfect calibration")
        ax.plot(mp, my, color=ps.BLUE if inn == 1 else ps.PINK, marker="o", ms=8, label="model (10 bins)")
        mt = metrics[f"WP{inn}"]
        ax.set_title(f"WP{inn}: {'first innings' if inn == 1 else 'chase'}")
        ax.text(0.03, 0.80, f"AUC {mt['auc']:.3f}\nBrier {mt['brier']:.3f}\nECE {mt['ece']:.3f}",
                transform=ax.transAxes, color=ps.INK, fontsize=10)
        ax.set_xlabel("Predicted win probability (out-of-fold)")
        ax.set_ylabel("Observed win rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
    fig.suptitle("Win-probability model calibration (5-fold, grouped by match)", fontweight="bold")
    out = Path(cfg["paths"]["out"])
    fig.savefig(out / "figures" / "wp_calibration.png")

    (out / "tables").mkdir(exist_ok=True, parents=True)
    with open(out / "tables" / "wp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(Path(cfg["paths"]["interim"]) / "wp_models.pkl", "wb") as f:
        pickle.dump(models, f)
    b.to_parquet(interim(cfg, "wp"), index=False)
    print(f"\n  wrote wp.parquet, folds.parquet, wp_models.pkl, figures/wp_calibration.png, tables/wp_metrics.json")

    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 4: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 4 checks failed - STOP and review.")


def plt_subplots():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    return fig, axes


def load_models(cfg: dict) -> dict:
    with open(Path(cfg["paths"]["interim"]) / "wp_models.pkl", "rb") as f:
        return pickle.load(f)

"""Stage 6b: process (delivery / shot) expectation models - xG-style.

Cross-fitted with the SAME match folds as Stages 4/6; out-of-fold predictions only; no player identity.
Delivery models (context features of Stage 6 + delivery family + on_or_off + over_or_around + bowling_type +
batting_type):
  xR_deliv      all deliveries   bowler_conceded (incl. wides / no-balls)   Poisson
  xW_deliv      all deliveries   bowler_wicket                              binary
  xR_deliv_bat  balls faced      runs off the bat                           Poisson
  xO_deliv_bat  balls faced      striker dismissed                          binary
Shot models (balls faced; delivery-model features + stroke family):
  xR_shot, xO_shot
Type baseline (bowlers): xR_type, xW_type = context + bowling_type + batting_type (no delivery label), so that
bowl_exec measures delivery choice / execution relative to bowlers of the same type.
Every label model is also fitted with FINE labels (suffix _fine) for the Stage 12 variant.

Outputs: interim/process_expected.parquet, interim/process_folds.json,
         tables/process_calibration.csv, process_calibration_season.csv, process_label_rates.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.value.s06_expected import prep, FEATS, FORBIDDEN
from pipeline.common.labels import add_labels

CATS_DELIV = ["on_or_off", "over_or_around", "bowling_type", "batting_type"]
MODELS = {  # name: (context set, rows, target, kind, uses stroke)
    "xR_deliv": ("bowl", "all", "bowler_conceded", "poisson", False),
    "xW_deliv": ("bowl", "all", "bowler_wicket", "binary", False),
    "xR_deliv_bat": ("bat", "faced", "runs", "poisson", False),
    "xO_deliv_bat": ("bat", "faced", "batter_out", "binary", False),
    "xR_shot": ("bat", "faced", "runs", "poisson", True),
    "xO_shot": ("bat", "faced", "batter_out", "binary", True),
    "xR_type": ("bowl", "all", "bowler_conceded", "poisson", "type"),
    "xW_type": ("bowl", "all", "bowler_wicket", "binary", "type"),
}


def _codes(b, cols):
    out = {}
    for c in cols:
        cat = pd.Categorical(b[c].fillna("none").astype(str))
        out[c] = cat.codes.astype(float)
    return pd.DataFrame(out, index=b.index)


def design(b, fset, stroke, level):
    deliv = "deliv_fam" if level == "fam" else "deliv_fine"
    strk = "stroke_fam" if level == "fam" else "stroke_fine"
    if stroke == "type":
        cats = ["bowling_type", "batting_type"]
    else:
        cats = [deliv] + CATS_DELIV + ([strk] if stroke else [])
    num = FEATS[fset]
    assert not (set(num + cats) & FORBIDDEN)
    X = pd.concat([b[num].astype(float), _codes(b, cats)], axis=1)
    catmask = np.r_[np.zeros(len(num), bool), np.ones(len(cats), bool)]
    return X.values, catmask, num + cats


def _model(kind, h, seed, catmask):
    common = dict(max_iter=h["max_iter"], learning_rate=h["learning_rate"], min_samples_leaf=h["min_samples_leaf"],
                  max_leaf_nodes=h["max_leaf_nodes"], random_state=seed, early_stopping=False,
                  categorical_features=catmask)
    if kind == "poisson":
        return HistGradientBoostingRegressor(loss="poisson", **common)
    return HistGradientBoostingClassifier(**common)


def fit_oof(b, name, level, K, ec, seed, train_mask, fold_log):
    fset, rows, target, kind, stroke = MODELS[name]
    X, catmask, cols = design(b, fset, stroke, level)
    rmask = b.is_faced.values if rows == "faced" else np.ones(len(b), bool)
    y = b[target].values
    pred = np.full(len(b), np.nan)
    key = name if level == "fam" else f"{name}_fine"
    fold_log[key] = {}
    for f in range(K):
        tr = rmask & train_mask & (b.fold_id.values != f)
        te = rmask & (b.fold_id.values == f)
        mdl = _model(kind, ec[kind], seed + f, catmask).fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te]) if kind == "poisson" else mdl.predict_proba(X[te])[:, 1]
        fold_log[key][f] = sorted(int(x) for x in np.unique(b.fold_id.values[tr]))
    return pred, cols


def calib(b, pred, obs, mask, keys, tol, slack):
    d = b[mask]
    g = d.groupby(keys).agg(n=(obs, "size"), obs=(obs, "mean"), pred=(pred, "mean"), var=(obs, "var"))
    g["ratio"] = g.pred / g.obs
    g["se"] = np.sqrt(g["var"] / g.n)
    g["pass"] = ((g.ratio - 1).abs() <= tol) | ((g.obs - g.pred).abs() <= slack * g.se)
    g["model"] = pred
    return g.drop(columns="var").reset_index()


def run(cfg: dict) -> None:
    header("STAGE 6b - PROCESS (DELIVERY / SHOT) EXPECTATION MODELS")
    b = pd.read_parquet(interim(cfg, "ball_value"))
    folds = pd.read_parquet(interim(cfg, "folds"))
    m = pd.read_parquet(interim(cfg, "matches"))
    ec, seed, K = cfg["expected"], cfg["seed"], cfg["wp"]["n_folds"]
    tabs = Path(cfg["paths"]["out"]) / "tables"
    R = []
    chk = b[["match_id", "fold_id"]].drop_duplicates().merge(folds, on="match_id", suffixes=("", "_s4"))
    check((chk.fold_id == chk.fold_id_s4).all() and len(chk) == b.match_id.nunique(),
          "using the same match folds as Stages 4 and 6", R)
    keep = b[["match_id", "innings", "innings_ball_num"]].copy()
    b = add_labels(prep(b))
    train_mask = ~b.match_id.isin(m.loc[m.data_suspect, "match_id"]).values
    print("  delivery families:", b.deliv_fam.value_counts().to_dict())
    print("  stroke families (balls faced):", b.loc[b.is_faced, "stroke_fam"].value_counts().to_dict())

    # label face-validity table
    f = b[b.is_faced]
    lr = []
    for col, mask in [("deliv_fam", b.phase.eq("death")), ("stroke_fam", b.is_faced)]:
        d = b[mask]
        g = d.groupby(col).agg(n=("runs", "size"), runs_per_ball=("runs", "mean"),
                               bowler_conceded_pb=("bowler_conceded", "mean"), wicket_pct=("bowler_wicket", "mean"),
                               dismissal_pct=("batter_out", "mean"))
        g["wicket_pct"] *= 100
        g["dismissal_pct"] *= 100
        lr.append(g.reset_index().rename(columns={col: "label"}).assign(family=col,
                  scope="death overs, all deliveries" if col == "deliv_fam" else "all balls faced"))
    lr = pd.concat(lr, ignore_index=True)
    lr.to_csv(tabs / "process_label_rates.csv", index=False)
    print("  death-over delivery families (bowler runs conceded / ball, wicket %):")
    print(lr[lr.family == "deliv_fam"].sort_values("bowler_conceded_pb")[["label", "n", "bowler_conceded_pb", "wicket_pct"]]
          .round(3).to_string(index=False))
    print("  stroke families (runs / ball, dismissal %):")
    print(lr[lr.family == "stroke_fam"].sort_values("dismissal_pct")[["label", "n", "runs_per_ball", "dismissal_pct"]]
          .round(3).to_string(index=False))

    fold_log = {}
    out = keep.copy()
    for level in ("fam", "fine"):
        for name in MODELS:
            if level == "fine" and MODELS[name][4] == "type":
                continue                                  # no labels in the type baseline
            col = name if level == "fam" else f"{name}_fine"
            print(f"  fitting {col} ...", flush=True)
            out[col], cols = fit_oof(b, name, level, K, ec, seed, train_mask, fold_log)
            if level == "fam" and name in ("xR_deliv", "xR_shot"):
                print(f"    features: {cols}")
    with open(Path(cfg["paths"]["interim"]) / "process_folds.json", "w", encoding="utf-8") as fh:
        json.dump(fold_log, fh, indent=1)
    oof_ok = all(int(k_) not in tr for mdl in fold_log.values() for k_, tr in mdl.items())
    check(oof_ok, "out-of-fold: no fold's predictions come from a model trained on that fold", R)
    faced = b.is_faced.values
    check(out.loc[faced, ["xR_deliv_bat", "xO_deliv_bat", "xR_shot", "xO_shot"]].notna().all().all()
          and out[["xR_deliv", "xW_deliv"]].notna().all().all(), "every eligible ball has OOF process expectations", R)

    # calibration
    header("Calibration (family models): innings x phase x comp_group, and by season x comp_group")
    bb = pd.concat([b, out.drop(columns=["match_id", "innings", "innings_ball_num"])], axis=1)
    tol_r, tol_w, slack = cfg["expected"]["tol_runs"], cfg["expected"]["tol_wkts"], cfg["expected"]["se_slack"]
    specs = [("xR_deliv", "bowler_conceded", None, tol_r), ("xW_deliv", "bowler_wicket", None, tol_w),
             ("xR_deliv_bat", "runs", "faced", tol_r), ("xO_deliv_bat", "batter_out", "faced", tol_w),
             ("xR_shot", "runs", "faced", tol_r), ("xO_shot", "batter_out", "faced", tol_w),
             ("xR_type", "bowler_conceded", None, tol_r), ("xW_type", "bowler_wicket", None, tol_w)]
    c1, c2 = [], []
    for p, o, rows, tol in specs:
        mk = (bb.is_faced.values if rows else np.ones(len(bb), bool)) & train_mask
        c1.append(calib(bb, p, o, mk, ["innings", "phase", "comp_group"], tol, slack))
        c2.append(calib(bb, p, o, mk, ["season", "comp_group"], tol, slack))
    c1, c2 = pd.concat(c1, ignore_index=True), pd.concat(c2, ignore_index=True)
    c1.to_csv(tabs / "process_calibration.csv", index=False)
    c2.to_csv(tabs / "process_calibration_season.csv", index=False)
    for nm, c in [("innings x phase x comp", c1), ("season x comp", c2)]:
        s = c.groupby("model").agg(cells=("pass", "size"), passed=("pass", "sum"),
                                   worst_ratio=("ratio", lambda r: r.iloc[(r - 1).abs().argmax()]))
        print(f"  {nm}:")
        print(s.round(3).to_string())
        check(c["pass"].mean() >= 0.9, f"{nm}: >= 90% of cells within tolerance or {slack:.0f} SE "
              f"({c['pass'].mean():.1%})", R)
    for p, o, rows, _ in specs:
        mk = (bb.is_faced.values if rows else np.ones(len(bb), bool)) & train_mask
        r = bb.loc[mk, p].sum() / bb.loc[mk, o].sum()
        check(abs(r - 1) < 0.02, f"{p}: overall predicted / actual {r:.4f}", R)

    out.to_parquet(interim(cfg, "process_expected"), index=False)
    print(f"\n  wrote process_expected.parquet ({len(out):,} rows), process_folds.json, process_calibration*.csv, "
          f"process_label_rates.csv")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 6b: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 6b checks failed - review.")

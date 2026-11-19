"""Stage 9: match-level consistency under pressure + match bootstrap.

Units = a player's PRESSURE balls in one match (batters >= 3, bowlers >= 6; config).
  C = (pos_w + a) / (n_w + a + b): recency/tier-weighted share of units with summed v > 0,
      shrunk by a Beta(a, b) prior fitted to the population (beta-binomial maximum likelihood).
Also: failure_rate (weighted share of units below the role's population 20th percentile of unit v),
      p20_value, sd of unit v, breadth (distinct seasons / competitions with pressure units).
Bootstrap: resample matches with replacement (B reps); P*, B*, D* (k fixed at full-data values) and
C (prior fixed) recomputed per rep from sparse player x match aggregates -> SEs, and saved for
Stage 10/11 rank intervals and P(top 10).

Outputs: interim/consistency.parquet, interim/pressure_units.parquet, interim/bootstrap.npz
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import betabinom

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.ratings.s08_player_ratings import ball_weights, role_frame


def units_frame(b: pd.DataFrame, role: str, w: np.ndarray, min_balls: int) -> pd.DataFrame:
    f = role_frame(b, role, w)
    f = f[f.pressure]
    md = b[["match_id", "match_date", "season", "comp_name", "tier"]].drop_duplicates("match_id")
    u = f.groupby(["player_id", "match_id"]).agg(v=("v", "sum"), wpa=("wpa", lambda x: x.fillna(0).sum()),
                                                 balls=("event", lambda e: (e == "ball").sum()), w=("w", "max"))
    u = u.reset_index().merge(md, on="match_id", how="left")
    u["role"] = role
    u["is_unit"] = u.balls >= min_balls
    return u


def fit_beta_prior(pos: np.ndarray, n: np.ndarray, method: str = "upper95_spread") -> dict:
    """Beta prior for C from a beta-binomial likelihood on (positive units, units) per player.

    ml:              maximum likelihood (a, b).
    upper95_spread:  mean m = pooled share; strength s = a + b at the 95% profile-likelihood LOWER
                     bound, i.e. the largest between-player spread in true consistency that the data
                     cannot rule out (same logic as k_P in Stage 8). If ML strength is finite and
                     smaller, ML is used.
    """
    from scipy.optimize import brentq
    m = pos.sum() / n.sum()
    nll = lambda s: -betabinom.logpmf(pos, n, m * s, (1 - m) * s).sum()
    r = minimize(lambda p: -betabinom.logpmf(pos, n, *np.exp(p)).sum(), np.log([m * 10, (1 - m) * 10]),
                 method="Nelder-Mead")
    a_ml, b_ml = np.exp(r.x)
    s_grid = np.exp(np.linspace(np.log(0.5), np.log(1e7), 400))
    vals = np.array([nll(s) for s in s_grid])
    s_hat = s_grid[vals.argmin()]
    lim = vals.min() + 3.84 / 2
    f = lambda ls: nll(np.exp(ls)) - lim
    s_lo = np.exp(brentq(f, np.log(0.5), np.log(s_hat))) if f(np.log(0.5)) > 0 and s_hat > 0.5 else 0.5
    out = {"a_ml": a_ml, "b_ml": b_ml, "s_ml": a_ml + b_ml, "s_profile_hat": s_hat, "s_lo95": s_lo, "mean": m}
    if method == "ml":
        out["a"], out["b"] = a_ml, b_ml
    else:
        s = min(s_lo, a_ml + b_ml)
        out["a"], out["b"] = m * s, (1 - m) * s
    sd_true = lambda s: np.sqrt(m * (1 - m) / (s + 1))
    out["sd_true_ml"], out["sd_true_used"] = sd_true(a_ml + b_ml), sd_true(out["a"] + out["b"])
    return out


def consistency(u: pd.DataFrame, a: float, b_: float, q20: float) -> pd.DataFrame:
    uu = u[u.is_unit].copy()
    uu["pos"] = (uu.v > 0).astype(float)
    uu["fail"] = (uu.v < q20).astype(float)
    g = uu.groupby("player_id")
    out = pd.DataFrame({
        "units": g.size(), "units_w": g.w.sum(), "pos_w": g.apply(lambda d: (d.w * d.pos).sum(), include_groups=False),
        "pos_share_raw": g.pos.mean(),
        "failure_rate": g.apply(lambda d: (d.w * d.fail).sum() / d.w.sum(), include_groups=False),
        "p20_value": g.v.quantile(0.2), "sd_unit_v": g.v.std(), "mean_unit_v": g.v.mean(),
        "unit_wpa_mean": g.wpa.mean(), "pressure_seasons": g.season.nunique(),
        "pressure_competitions": g.comp_name.nunique()})
    out["C"] = (out.pos_w + a) / (out.units_w + a + b_)
    return out


def _sparse(f: pd.DataFrame, pidx: pd.Index, midx: pd.Index, col: str) -> sparse.csr_matrix:
    r = pidx.get_indexer(f.player_id)
    c = midx.get_indexer(f.match_id)
    return sparse.csr_matrix((f[col].values, (r, c)), shape=(len(pidx), len(midx)))


def bootstrap(b, role, w, k, prior, q20, min_balls, reps, seed, vcol=None):
    """Match bootstrap of P*, B*, D* (x100) and C for every player of the role.

    Same seed + same b -> the identical match resamples (one multinomial draw per rep), so process-track
    bootstraps (vcol) line up rep-by-rep with the Stage 9 outcome bootstrap. C always uses outcome units.
    """
    f = role_frame(b, role, w, vcol_override=vcol)
    f = f.assign(wv=f.w * f.v, w_b=np.where(f.event == "ball", f.w, 0.0),
                 w2_b=np.where(f.event == "ball", f.w ** 2, 0.0))
    pidx = pd.Index(sorted(f.player_id.unique()))
    midx = pd.Index(sorted(b.match_id.unique()))
    mats = {}
    for tag, sub in [("P", f[f.pressure]), ("B", f[~f.pressure])]:
        agg = sub.groupby(["player_id", "match_id"]).agg(sw=("w_b", "sum"), sw2=("w2_b", "sum"), swv=("wv", "sum")).reset_index()
        mats[tag] = {c: _sparse(agg, pidx, midx, c) for c in ("sw", "sw2", "swv")}
    u = units_frame(b, role, w, min_balls)
    uu = u[u.is_unit].assign(pos=lambda d: (d.v > 0).astype(float) * d.w)
    mats["U"] = {"w": _sparse(uu, pidx, midx, "w"), "pos": _sparse(uu, pidx, midx, "pos")}
    rng = np.random.default_rng(seed)
    M = len(midx)

    def stats(cnt):
        out = {}
        for tag in ("P", "B"):
            sw, sw2, swv = (mats[tag][c] @ cnt for c in ("sw", "sw2", "swv"))
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = np.where(sw > 0, swv / sw, 0.0)
                neff = np.where(sw2 > 0, sw ** 2 / sw2, 0.0)
            out[tag] = (mean, neff)
        (Pm, nP), (Bm, nB) = out["P"], out["B"]
        Bs = nB * Bm / (nB + k["k_B"])
        Ps = (nP * Pm + k["k_P"] * Bs) / (nP + k["k_P"])
        C = (mats["U"]["pos"] @ cnt + prior[0]) / (mats["U"]["w"] @ cnt + prior[0] + prior[1])
        return 100 * Ps, 100 * Bs, 100 * (Ps - Bs), C

    point = stats(np.ones(M))
    arr = {n: np.empty((len(pidx), reps), dtype=np.float32) for n in ("P_star", "B_star", "D_star", "C")}
    for r_ in range(reps):
        cnt = rng.multinomial(M, np.full(M, 1 / M)).astype(float)
        for n, v in zip(arr, stats(cnt)):
            arr[n][:, r_] = v
    return pidx, point, arr


def run(cfg: dict) -> None:
    header("STAGE 9 - CONSISTENCY UNDER PRESSURE + BOOTSTRAP")
    b = pd.read_parquet(interim(cfg, "ball_value"))
    b = b[~b.data_suspect].reset_index(drop=True)
    ratings = pd.read_parquet(interim(cfg, "player_ratings"))
    out = Path(cfg["paths"]["out"])
    with open(out / "tables" / "eb_k_values.json") as f:
        kvals = json.load(f)
    el = cfg["eligibility"]
    reps = cfg["validation"]["bootstrap_reps"]
    w = ball_weights(b, cfg)
    R = []
    all_c, all_u, boot, prior_info = [], [], {}, {}

    for role in ("bat", "bowl"):
        mb = el[f"match_min_pressure_balls_{role}"]
        header(f"{'Batting' if role == 'bat' else 'Bowling'}: pressure units (>= {mb} pressure balls in a match)")
        u = units_frame(b, role, w, mb)
        uu = u[u.is_unit]
        q20 = float(uu.v.quantile(0.2))
        print(f"  player-matches with any pressure ball: {len(u):,}; units (>= {mb}): {len(uu):,} "
              f"({uu.player_id.nunique()} players); unit v: mean {uu.v.mean():+.2f}, sd {uu.v.std():.2f}, "
              f"share positive {(uu.v > 0).mean():.1%}, population p20 {q20:+.2f} runs")
        cnt = uu.groupby("player_id").agg(pos=("v", lambda x: int((x > 0).sum())), n=("v", "size"))
        fitset = cnt[cnt.n >= 5]
        pr = fit_beta_prior(fitset.pos.values, fitset.n.values, cfg["consistency"]["prior_method"])
        a, b_ = pr["a"], pr["b"]
        print(f"  Beta prior from {len(fitset)} players with >= 5 units: pooled positive share {pr['mean']:.3f}")
        print(f"    ML strength a+b = {pr['s_ml']:.1f} pseudo-units -> true between-player sd of C {pr['sd_true_ml']:.3f}")
        print(f"    95% lower bound on strength = {pr['s_lo95']:.1f} -> largest sd consistent with data "
              f"{np.sqrt(pr['mean'] * (1 - pr['mean']) / (pr['s_lo95'] + 1)):.3f}")
        print(f"    USED ({cfg['consistency']['prior_method']}): a = {a:.2f}, b = {b_:.2f}, strength {a + b_:.1f}, "
              f"true sd {pr['sd_true_used']:.3f}")
        prior_info[role] = {k_: float(v) for k_, v in pr.items()}
        c = consistency(u, a, b_, q20)
        c["role"] = role
        c["prior_a"], c["prior_b"] = a, b_
        all_c.append(c.reset_index())
        all_u.append(u)

        print(f"  bootstrap: {reps} match resamples ...", flush=True)
        k = kvals[role]
        pidx, point, arr = bootstrap(b, role, w, k, (a, b_), q20, mb, reps, cfg["seed"] + (0 if role == "bat" else 1))
        boot[role] = (pidx, arr)
        rr = ratings[ratings.role == role].set_index("player_id").reindex(pidx)
        inr = rr.P_star.notna().values      # players seen only as run-out non-strikers have no Stage 8 row
        check(np.allclose(point[0][inr], rr.P_star.values[inr], atol=1e-6) and
              np.allclose(point[2][inr], rr.D_star.values[inr], atol=1e-6),
              f"{role}: bootstrap machinery reproduces Stage 8 P* and D* with unit counts "
              f"({(~inr).sum()} non-striker-only ids skipped)", R)
        cc = c.C.reindex(pidx)
        has = cc.notna().values
        check(np.allclose(point[3][has], cc.values[has], atol=1e-9), f"{role}: bootstrap machinery reproduces C", R)

    cons = pd.concat(all_c, ignore_index=True)
    units = pd.concat(all_u, ignore_index=True)

    # standard errors
    se_rows = []
    for role, (pidx, arr) in boot.items():
        se_rows.append(pd.DataFrame({"player_id": pidx, "role": role,
                                     "se_P_star": arr["P_star"].std(axis=1), "se_B_star": arr["B_star"].std(axis=1),
                                     "se_D_star": arr["D_star"].std(axis=1), "se_C": arr["C"].std(axis=1),
                                     "boot_mean_P_star": arr["P_star"].mean(axis=1)}))
    se = pd.concat(se_rows, ignore_index=True)
    cons = cons.merge(se, on=["player_id", "role"], how="outer")
    np.savez_compressed(Path(cfg["paths"]["interim"]) / "bootstrap.npz",
                        **{f"{role}_{n}": v for role, (pidx, arr) in boot.items() for n, v in arr.items()},
                        **{f"{role}_player_id": np.asarray(pidx) for role, (pidx, _) in boot.items()})

    # ---------------- summaries ----------------
    since = pd.Timestamp(el["active_since"])
    for role in ("bat", "bowl"):
        r = ratings[ratings.role == role].merge(cons[cons.role == role], on=["player_id", "role"], how="left")
        mi = el[f"{role}_min_pressure_" + ("innings" if role == "bat" else "spells")]
        q = r[(r.pressure_balls >= el[f"{role}_min_pressure_balls"]) & (r.pressure_innings >= mi) &
              (r.last_match_date >= since)]
        header(f"{role}: consistency among the {len(q)} provisional-pool players")
        print(f"  C: median {q.C.median():.3f}, p10 {q.C.quantile(.1):.3f}, p90 {q.C.quantile(.9):.3f}; units per player "
              f"median {q.units.median():.0f}")
        print(f"  Spearman: C vs P* {q[['C', 'P_star']].corr('spearman').iloc[0, 1]:.2f}, C vs D* "
              f"{q[['C', 'D_star']].corr('spearman').iloc[0, 1]:.2f}, C vs failure_rate "
              f"{q[['C', 'failure_rate']].corr('spearman').iloc[0, 1]:.2f}")
        print(f"  bootstrap SE (median): P* {q.se_P_star.median():.2f}, D* {q.se_D_star.median():.2f}, "
              f"C {q.se_C.median():.3f}; |boot mean - point| P* median {(q.boot_mean_P_star - q.P_star).abs().median():.2f}")
        cols = ["name", "units", "pos_share_raw", "C", "se_C", "failure_rate", "p20_value", "pressure_seasons",
                "pressure_competitions",
                "P_star", "se_P_star", "D_star", "se_D_star"]
        print("  top 15 by C:")
        print(q.nlargest(15, "C")[cols].round(3).to_string(index=False))
        check(q.C.between(0, 1).all(), f"{role}: C in [0, 1] for the pool", R)

    with open(out / "tables" / "consistency_prior.json", "w") as f:
        json.dump(prior_info, f, indent=2)
    cons.to_parquet(interim(cfg, "consistency"), index=False)
    units.to_parquet(interim(cfg, "pressure_units"), index=False)
    print(f"\n  wrote consistency.parquet ({len(cons):,} rows), pressure_units.parquet ({len(units):,} player-matches), "
          f"bootstrap.npz ({reps} reps)")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 9: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 9 checks failed - STOP and review.")

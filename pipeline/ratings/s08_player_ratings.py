"""Stage 8: player ratings under pressure with empirical-Bayes shrinkage.

Each ball weight w = recency x tier weight, recency = 0.5 ** (days before reference / half-life).
Per player (batting = striker on balls faced, + non-striker run-out penalties; bowling = all deliveries):
  P_raw = weighted mean v on pressure balls x 100 (runs above expected per 100 balls)
  B_raw = weighted mean v on normal balls x 100
EB:  B* = n_B B_raw / (n_B + k_B)                 (league mean is 0 by construction)
     P* = (n_P P_raw + k_P B*) / (n_P + k_P)      (shrinks toward the player's own baseline)
     D* = P* - B*
with n = effective sample size (sum w)^2 / sum w^2, k = sigma^2_ball / tau^2.
k_P uses the between-player variance of the pressure-vs-baseline gap (config ratings.k_p_method).

`compute_ratings` is reused by Stages 10-12 (train period, odd/even halves, sensitivity variants).
Outputs: interim/player_ratings.parquet, interim/player_splits.parquet, tables/eb_k_values.json,
         tables/raw_pressure_trap_players.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim, load_raw
from pipeline.value.s07_ball_value import display_names

ROLE_COLS = {"bat": ("striker_id", "v_bat", "wpa_bat"), "bowl": ("bowler_id", "v_bowl", "wpa_bowl")}


# --------------------------------------------------------------------------- weights / units
def ball_weights(b: pd.DataFrame, cfg: dict, half_life=None, tiers=None) -> np.ndarray:
    hl = cfg["recency"]["half_life_days"] if half_life is None else half_life
    tiers = cfg["tiers"] if tiers is None else tiers
    ref = pd.Timestamp(cfg["reference_date"])
    days = (ref - pd.to_datetime(b.match_date)).dt.days.clip(lower=0).values
    rec = np.ones(len(b)) if (hl is None or not np.isfinite(hl)) else 0.5 ** (days / hl)
    return rec * b.tier.map(tiers).fillna(1.0).values


def role_frame(b: pd.DataFrame, role: str, w: np.ndarray, pcol: str = "is_pressure",
               vcol_override: str | None = None) -> pd.DataFrame:
    """One row per value event for the role: player_id, match_id, v, wpa, w, pressure, LI_w.

    vcol_override: use another per-ball value (process track). Batting then uses balls faced only
    (non-striker run-out penalties are outcome events, not process).
    """
    pid, vcol, wcol = ROLE_COLS[role]
    if vcol_override is not None:
        s = b.is_faced.values if role == "bat" else np.ones(len(b), bool)
        return pd.DataFrame({"player_id": b[pid].values[s], "match_id": b.match_id.values[s],
                             "v": b[vcol_override].values[s], "wpa": b[wcol].values[s], "w": w[s],
                             "pressure": b[pcol].values[s], "LI_w": b.LI_w.values[s], "event": "ball"})
    if role == "bat":
        s = b.is_faced.values
        main = pd.DataFrame({"player_id": b[pid].values[s], "match_id": b.match_id.values[s],
                             "v": b[vcol].values[s], "wpa": b[wcol].values[s], "w": w[s],
                             "pressure": b[pcol].values[s], "LI_w": b.LI_w.values[s], "event": "ball"})
        ns = b.nonstriker_out.values
        pen = pd.DataFrame({"player_id": b.non_striker_id.values[ns], "match_id": b.match_id.values[ns],
                            "v": b.ns_runout_penalty.values[ns], "wpa": b.wpa_ns.values[ns], "w": w[ns],
                            "pressure": b[pcol].values[ns], "LI_w": b.LI_w.values[ns], "event": "ns_runout"})
        return pd.concat([main, pen], ignore_index=True)
    return pd.DataFrame({"player_id": b[pid].values, "match_id": b.match_id.values, "v": b[vcol].values,
                         "wpa": b[wcol].values, "w": w, "pressure": b[pcol].values, "LI_w": b.LI_w.values,
                         "event": "ball"})


# --------------------------------------------------------------------------- EB core
def _wmeans(f: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean, n_eff, raw count per player. Non-striker penalties add to the numerator only."""
    f = f.assign(wv=f.w * f.v)
    balls = f[f.event == "ball"]
    g = balls.groupby("player_id").agg(sw=("w", "sum"), sw2=("w", lambda x: (x ** 2).sum()), n=("w", "size"))
    num = f.groupby("player_id").wv.sum()
    g["mean"] = num.reindex(g.index) / g.sw
    g["n_eff"] = g.sw ** 2 / g.sw2
    return g


def _sigma2(f: pd.DataFrame, means: pd.Series) -> float:
    d = f[f.event == "ball"]
    mu = d.player_id.map(means).values
    return float(np.sum(d.w.values * (d.v.values - mu) ** 2) / np.sum(d.w.values))


def _tau2(x: pd.Series, samp_var: pd.Series, floor: float) -> float:
    """Method of moments: Var(player means) - mean(sampling variance)."""
    return max(float(x.var() - samp_var.mean()), floor)


def reml_tau2(x, v, floor: float = 1e-8) -> tuple[float, float, float]:
    """REML between-player variance for x_i ~ N(mu, tau2 + v_i); returns (estimate, lo95, hi95).

    The 95% interval is from the profile likelihood (deviance 3.84 = chi2_1 at 0.95).
    """
    from scipy.optimize import minimize_scalar, brentq
    x, v = np.asarray(x, float), np.asarray(v, float)

    def nll(t):
        W = 1 / (v + t)
        mu = np.sum(W * x) / W.sum()
        return 0.5 * (np.sum(np.log(v + t)) + np.log(W.sum()) + np.sum(W * (x - mu) ** 2))
    hi_b = max(1.0, 10 * np.var(x))
    t = minimize_scalar(nll, bounds=(0, hi_b), method="bounded", options={"xatol": 1e-10}).x
    lim = nll(t) + 3.84 / 2
    g = lambda s: nll(s) - lim
    hi = brentq(g, t, hi_b) if g(hi_b) > 0 else hi_b
    lo = brentq(g, 0, t) if t > 0 and g(0) > 0 else 0.0
    return max(t, floor), lo, hi


def compute_ratings(b: pd.DataFrame, cfg: dict, role: str, w: np.ndarray, pcol: str = "is_pressure",
                    k_override: dict | None = None, verbose: bool = False, frame: pd.DataFrame | None = None,
                    vcol: str | None = None):
    """`frame` (a role_frame subset, e.g. one split-half) overrides building the frame from b.
    `vcol` rates another per-ball value (process track) with the same machinery."""
    rc = cfg["ratings"]
    f = role_frame(b, role, w, pcol, vcol) if frame is None else frame
    P = _wmeans(f[f.pressure])
    B = _wmeans(f[~f.pressure])
    s2P = _sigma2(f[f.pressure], P["mean"])
    s2B = _sigma2(f[~f.pressure], B["mean"])
    r = pd.DataFrame(index=P.index.union(B.index))
    r["P_raw"] = P["mean"]
    r["B_raw"] = B["mean"]
    r["n_P"] = P.n_eff
    r["n_B"] = B.n_eff
    r["pressure_balls"] = P.n
    r["normal_balls"] = B.n
    r[["n_P", "n_B", "pressure_balls", "normal_balls"]] = r[["n_P", "n_B", "pressure_balls", "normal_balls"]].fillna(0)

    if k_override is None:
        mB = r.n_B >= rc["tau_min_neff"]
        both = (r.n_B >= rc["tau_min_neff"]) & (r.n_P >= rc["tau_min_neff"])
        mP = r.n_P >= rc["tau_min_neff"]
        xB, vB = r.loc[mB, "B_raw"], s2B / r.loc[mB, "n_B"]
        xD = r.loc[both, "P_raw"] - r.loc[both, "B_raw"]
        vD = s2P / r.loc[both, "n_P"] + s2B / r.loc[both, "n_B"]
        xP, vP = r.loc[mP, "P_raw"], s2P / r.loc[mP, "n_P"]
        if rc["tau_method"] == "reml":
            tauB, tauB_lo, tauB_hi = reml_tau2(xB, vB, rc["tau_floor"])
            tauD, tauD_lo, tauD_hi = reml_tau2(xD, vD, rc["tau_floor"])
            tauP, _, _ = reml_tau2(xP, vP, rc["tau_floor"])
        else:
            tauB, tauD, tauP = (_tau2(xB, vB, rc["tau_floor"]), _tau2(xD, vD, rc["tau_floor"]),
                                _tau2(xP, vP, rc["tau_floor"]))
            tauB_lo = tauB_hi = tauD_lo = tauD_hi = np.nan
        tau_for_kp = {"clutch_upper95": tauD_hi, "clutch_point": tauD, "spec_literal": tauP}[rc["k_p_method"]]
        k = {"k_B": s2B / tauB, "k_P": s2P / tau_for_kp,
             "k_P_clutch_point": s2P / tauD, "k_P_spec_literal": s2P / tauP,
             "sigma2_P": s2P, "sigma2_B": s2B, "tau2_B": tauB, "tau2_B_ci": [tauB_lo, tauB_hi],
             "tau2_D": tauD, "tau2_D_ci": [tauD_lo, tauD_hi], "tau2_P": tauP,
             "tau_method": rc["tau_method"], "k_p_method": rc["k_p_method"],
             "n_players_tau_B": int(mB.sum()), "n_players_tau_D": int(both.sum())}
    else:
        k = dict(k_override)
    Br = r.B_raw.fillna(0.0)
    r["B_star"] = r.n_B * Br / (r.n_B + k["k_B"])
    Pr = r.P_raw.fillna(0.0)
    r["P_star"] = (r.n_P * Pr + k["k_P"] * r.B_star) / (r.n_P + k["k_P"])
    r["D_star"] = r.P_star - r.B_star
    for c in ["P_raw", "B_raw", "B_star", "P_star", "D_star"]:
        r[c] = r[c] * 100
    r["shrink_P"] = r.n_P / (r.n_P + k["k_P"])          # weight on the player's own pressure record
    r["shrink_B"] = r.n_B / (r.n_B + k["k_B"])

    # WPA and clutch cross-check (all balls of the role)
    fb = f[f.event == "ball"].copy()
    g = fb.assign(wpa0=fb.wpa.fillna(0), wpa_li=(fb.wpa / fb.LI_w).fillna(0)).groupby("player_id")
    r["wpa_total"] = g.wpa0.sum()
    r["wpa_pressure"] = fb[fb.pressure].groupby("player_id").wpa.sum()
    r["mean_LI_faced"] = g.LI_w.mean()
    r["clutch_wpa"] = r.wpa_total / r.mean_LI_faced - g.wpa_li.sum()
    ns = f[f.event == "ns_runout"].groupby("player_id").wpa.sum()
    r["wpa_total"] = r.wpa_total.add(ns, fill_value=0)
    r["role"] = role
    return r.reset_index().rename(columns={"index": "player_id"}), k


# --------------------------------------------------------------------------- supporting stats / splits
def supporting(b: pd.DataFrame, role: str, ok: np.ndarray) -> pd.DataFrame:
    if role == "bat":
        d = b[ok & b.is_faced.values]
        out = []
        for flag, sub in [("pressure", d[d.is_pressure]), ("all", d)]:
            g = sub.groupby("striker_id")
            t = pd.DataFrame({
                f"{flag}_raw_SR": 100 * g.runs.mean(),
                f"{flag}_SR_minus_xSR": 100 * (g.runs.mean() - g.xR_bat.mean()),
                f"{flag}_dismissal_minus_x_pct": 100 * (g.batter_out.mean() - g.xO_bat.mean()),
                f"{flag}_dot_minus_x_pct": 100 * (g.y_dot.mean() - g.xDot_bat.mean()),
                f"{flag}_boundary_minus_x_pct": 100 * (g.y_bnd.mean() - g.xBnd_bat.mean()),
                f"{flag}_raw_avg": g.runs.sum() / g.batter_out.sum().replace(0, np.nan),
                f"{flag}_runs": g.runs.sum(), f"{flag}_outs": g.batter_out.sum()})
            out.append(t)
        return pd.concat(out, axis=1).rename_axis("player_id").reset_index()
    d = b[ok]
    out = []
    for flag, sub in [("pressure", d[d.is_pressure]), ("all", d)]:
        g = sub.groupby("bowler_id")
        legal = g.is_legal.sum()
        lg = sub[sub.is_legal].groupby("bowler_id")
        t = pd.DataFrame({
            f"{flag}_raw_econ": 6 * g.bowler_conceded.sum() / legal,
            f"{flag}_econ_minus_x": 6 * (g.bowler_conceded.sum() - g.xR_bowl.sum()) / legal,
            f"{flag}_wkt_minus_x_pct": 100 * (g.bowler_wicket.mean() - g.xW_bowl.mean()),
            f"{flag}_dot_pct": 100 * lg.apply(lambda x: ((x.bowler_conceded == 0) & ~x.is_wicket).mean(), include_groups=False),
            f"{flag}_boundary_pct": 100 * lg.apply(lambda x: (x.is_four | x.is_six).mean(), include_groups=False),
            f"{flag}_wkts": g.bowler_wicket.sum()})
        out.append(t)
    return pd.concat(out, axis=1).rename_axis("player_id").reset_index()


def splits(b: pd.DataFrame, role: str, w: np.ndarray, ok: np.ndarray) -> pd.DataFrame:
    pid, vcol, wcol = ROLE_COLS[role]
    s = ok & (b.is_faced.values if role == "bat" else True)
    d = b[s].assign(w=w[s])
    defs = {"phase": d.phase, "innings": d.innings.map({1: "setting", 2: "chasing"}),
            "comp_group": d.comp_group,
            "flag_collapse": d.flag_collapse, "flag_steep_chase": d.flag_steep_chase,
            "flag_new_batter": d.flag_new_batter, "flag_death_defence": d.flag_death_defence,
            "flag_knockout": d.flag_knockout,
            "pace_or_spin": d.pace_or_spin if role == "bat" else d.pace_or_spin}
    rows = []
    for name, key in defs.items():
        tmp = d.assign(key=key.astype(str).values, wv=d.w * d[vcol])
        for pres_name, sub in [("pressure", tmp[tmp.is_pressure]), ("all", tmp)]:
            g = sub.groupby([pid, "key"]).agg(balls=(vcol, "size"), sw=("w", "sum"), swv=("wv", "sum"),
                                              wpa=(wcol, "sum"))
            g["v_per100"] = 100 * g.swv / g.sw
            g = g.reset_index().rename(columns={pid: "player_id", "key": "split_value"})
            g["split"] = name
            g["balls_type"] = pres_name
            rows.append(g[["player_id", "split", "split_value", "balls_type", "balls", "v_per100", "wpa"]])
    out = pd.concat(rows, ignore_index=True)
    out["role"] = role
    return out


# --------------------------------------------------------------------------- run
def run(cfg: dict) -> None:
    header("STAGE 8 - PLAYER RATINGS")
    b = pd.read_parquet(interim(cfg, "ball_value"))
    ok = ~b.data_suspect.values
    b = b[ok].reset_index(drop=True)
    ok = np.ones(len(b), bool)
    out = Path(cfg["paths"]["out"])
    R = []
    w = ball_weights(b, cfg)
    print(f"ball weights: recency half-life {cfg['recency']['half_life_days']} days to {cfg['reference_date']}, "
          f"tiers {cfg['tiers']}; mean weight {w.mean():.3f}, IPL 2026 ball {w[b.comp_name.str.startswith('IPL 2026').values].mean():.3f}, "
          f"IPL 2022 ball {w[b.comp_name.str.startswith('IPL 2022').values].mean():.3f}")
    names = display_names(b)
    xi = load_raw(cfg, "xi")
    mm = pd.read_parquet(interim(cfg, "matches"))[["match_id", "match_date", "comp_name", "comp_group"]]
    xi = xi[xi.is_played.astype(str).str.lower().eq("true")].merge(mm, on="match_id")
    act = xi.groupby("player_id").agg(last_match_date=("match_date", "max"), matches=("match_id", "nunique"),
                                      competitions=("comp_name", "nunique"),
                                      comp_groups=("comp_group", lambda s: ",".join(sorted(set(s)))))

    all_r, all_k, all_split = [], {}, []
    for role in ("bat", "bowl"):
        header(f"{'Batting' if role == 'bat' else 'Bowling'}: empirical-Bayes shrinkage")
        r, k = compute_ratings(b, cfg, role, w)
        all_k[role] = k
        pid = ROLE_COLS[role][0]
        rf = b[b.is_faced] if role == "bat" else b
        pin = rf[rf.is_pressure].groupby(pid).match_id.nunique()
        r["pressure_innings"] = r.player_id.map(pin).fillna(0).astype(int)
        r["role_matches"] = r.player_id.map(rf.groupby(pid).match_id.nunique()).fillna(0).astype(int)
        r = r.merge(act, left_on="player_id", right_index=True, how="left")
        r["name"] = r.player_id.map(names)
        r = r.merge(supporting(b, role, ok), on="player_id", how="left")
        all_r.append(r)
        all_split.append(splits(b, role, w, ok))
        print(f"  sigma^2 per ball: pressure {k['sigma2_P']:.3f}, normal {k['sigma2_B']:.3f} (runs^2)")
        sd = lambda t: 100 * np.sqrt(t)
        print(f"  [{k['tau_method'].upper()}] true baseline spread sd(B) = {sd(k['tau2_B']):.2f} per 100 balls "
              f"(95% CI {sd(k['tau2_B_ci'][0]):.2f}-{sd(k['tau2_B_ci'][1]):.2f}; {k['n_players_tau_B']} players)")
        print(f"  [{k['tau_method'].upper()}] true clutch spread sd(P-B) = {sd(k['tau2_D']):.2f} per 100 balls "
              f"(95% CI {sd(k['tau2_D_ci'][0]):.2f}-{sd(k['tau2_D_ci'][1]):.2f}; {k['n_players_tau_D']} players)"
              f" -> using the upper bound {sd(k['tau2_D_ci'][1]):.2f}" if k["k_p_method"] == "clutch_upper95" else "")
        print(f"  k_B = {k['k_B']:.1f} effective balls | k_P = {k['k_P']:.1f} effective pressure balls "
              f"(clutch point estimate would give {k['k_P_clutch_point']:.0f}; spec-literal {k['k_P_spec_literal']:.0f})")
        who = "batter" if role == "bat" else "bowler"
        print(f"  -> a {who} needs ~{k['k_B']:.0f} normal balls before his baseline counts as much as the league "
              f"prior, and ~{k['k_P']:.0f} pressure balls before his pressure record counts as much as his own baseline.")
        check(np.isfinite([k["k_B"], k["k_P"]]).all() and k["k_B"] > 0 and k["k_P"] > 0, f"{role}: finite positive k values", R)
        q = r[r.pressure_balls >= cfg["eligibility"][f"{role}_min_pressure_balls"]]
        print(f"  players with >= {cfg['eligibility'][f'{role}_min_pressure_balls']} pressure balls: {len(q)}; "
              f"shrink weight on own pressure record: median {q.shrink_P.median():.2f}, max {q.shrink_P.max():.2f}")
        print(f"  spread among them (per 100 balls): sd P* {q.P_star.std():.2f}, sd B* {q.B_star.std():.2f}, "
              f"sd D* {q.D_star.std():.2f}, sd raw P {q.P_raw.std():.2f}")
        print(f"  Spearman(P*, B*) {q[['P_star', 'B_star']].corr('spearman').iloc[0, 1]:.2f}; "
              f"Spearman(P*, clutch_wpa) {q[['P_star', 'clutch_wpa']].corr('spearman').iloc[0, 1]:.2f}; "
              f"Spearman(D*, clutch_wpa) {q[['D_star', 'clutch_wpa']].corr('spearman').iloc[0, 1]:.2f}")

    ratings = pd.concat(all_r, ignore_index=True)
    sp = pd.concat(all_split, ignore_index=True)

    # ---------------- sanity ----------------
    header("Sanity")
    for role in ("bat", "bowl"):
        r = ratings[ratings.role == role]
        check(r.player_id.is_unique, f"{role}: one row per player", R)
        check(r[["P_star", "B_star", "D_star"]].notna().all().all(), f"{role}: P*, B*, D* defined for every player", R)
    # reconciliation: weighted P_raw recomputed directly for one player
    pr = ratings[(ratings.role == "bat")].nlargest(1, "pressure_balls").iloc[0]
    d = b[b.is_faced & b.is_pressure & (b.striker_id == pr.player_id)]
    wd = w[(b.is_faced & b.is_pressure & (b.striker_id == pr.player_id)).values]
    pen = b[b.nonstriker_out & b.is_pressure & (b.non_striker_id == pr.player_id)]
    wpen = w[(b.nonstriker_out & b.is_pressure & (b.non_striker_id == pr.player_id)).values]
    direct = 100 * (np.sum(wd * d.v_bat) + np.sum(wpen * pen.ns_runout_penalty)) / wd.sum()
    check(abs(direct - pr.P_raw) < 1e-6, f"P_raw reconciles for {pr['name']} ({direct:.3f} vs {pr.P_raw:.3f})", R)

    # ---------------- preview: top 20 by P* (provisional filter; eligibility is Stage 10) ----------------
    el = cfg["eligibility"]
    since = pd.Timestamp(el["active_since"])
    for role in ("bat", "bowl"):
        r = ratings[ratings.role == role]
        mb, mi = el[f"{role}_min_pressure_balls"], el[f"{role}_min_pressure_" + ("innings" if role == "bat" else "spells")]
        q = r[(r.pressure_balls >= mb) & (r.pressure_innings >= mi) & (r.last_match_date >= since)]
        header(f"Preview ({role}): top 20 by P* among {len(q)} players with >= {mb} pressure balls, >= {mi} "
               f"pressure {'innings' if role == 'bat' else 'spells'}, a match since {el['active_since']}")
        cols = ["name", "pressure_balls", "pressure_innings", "P_raw", "P_star", "B_star", "D_star", "shrink_P",
                "shrink_B", "clutch_wpa"]
        cols += (["pressure_raw_SR", "pressure_SR_minus_xSR", "pressure_dismissal_minus_x_pct"] if role == "bat"
                 else ["pressure_raw_econ", "pressure_econ_minus_x", "pressure_wkt_minus_x_pct"])
        print(q.nlargest(20, "P_star")[cols].round(2).to_string(index=False))

    # ---------------- per-player raw trap ----------------
    header("Per-player raw-stats trap (batters, provisional pool)")
    r = ratings[(ratings.role == "bat")]
    q = r[(r.pressure_balls >= el["bat_min_pressure_balls"]) & (r.pressure_innings >= el["bat_min_pressure_innings"])].copy()
    q["rank_raw_pressure_SR"] = q.pressure_raw_SR.rank(ascending=False)
    q["rank_P_star"] = q.P_star.rank(ascending=False)
    rho = q[["pressure_raw_SR", "P_star"]].corr("spearman").iloc[0, 1]
    q["rank_gap"] = q.rank_raw_pressure_SR - q.rank_P_star
    print(f"  {len(q)} batters; Spearman(raw pressure SR, P*) = {rho:.2f}")
    show = ["name", "pressure_balls", "pressure_raw_SR", "pressure_SR_minus_xSR", "P_star", "rank_raw_pressure_SR", "rank_P_star"]
    print("  raw pressure SR flatters them most (high raw rank, low P* rank):")
    print(q.nsmallest(5, "rank_gap")[show].round(1).to_string(index=False))
    print("  raw pressure SR undersells them most:")
    print(q.nlargest(5, "rank_gap")[show].round(1).to_string(index=False))
    q.sort_values("rank_gap")[show + ["rank_gap"]].to_csv(out / "tables" / "raw_pressure_trap_players.csv", index=False)

    ratings.to_parquet(interim(cfg, "player_ratings"), index=False)
    sp.to_parquet(interim(cfg, "player_splits"), index=False)
    with open(out / "tables" / "eb_k_values.json", "w") as f:
        json.dump(all_k, f, indent=2)
    print(f"\n  wrote player_ratings.parquet ({len(ratings):,} player-role rows), player_splits.parquet "
          f"({len(sp):,} rows), tables eb_k_values.json, raw_pressure_trap_players.csv")
    n_fail = sum(not ok_ for ok_, _ in R)
    print(f"\nStage 8: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 8 checks failed - STOP and review.")

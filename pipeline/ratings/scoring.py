"""Shared machinery for the final rating (Stages 10-12).

build_role : outcome ratings (Stage 8) + process ratings (Stage 8c) + consistency C (Stage 9) for any ball
             subset / weighting / pressure definition. REML k and the Beta prior are re-fitted unless given.
compose    : R_P = a Process_P* + (1-a) Outcome_P*, R_B likewise on normal balls, D = R_P - R_B.
eligible   : final-list eligibility = last match >= active_since AND min pressure balls AND min pressure
             innings/spells. The P_raw >= 0 guard is applied in `score` (ranking only).
score      : z-score R_P, D, C within the eligible pool -> RUP; guard; rank.
player_info: informational columns only (IPL 2026 team, nationality, overseas, Pakistan flag, RR, availability).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import load_raw, interim
from pipeline.ratings.s08_player_ratings import compute_ratings, ROLE_COLS
from pipeline.ratings.s09_consistency import units_frame, fit_beta_prior, consistency

SPELL = {"bat": "innings", "bowl": "spells"}
PROC_COL = {"bat": "bat_decision", "bowl": "bowl_exec"}


def build_role(b: pd.DataFrame, cfg: dict, role: str, w: np.ndarray, k_override: dict | None = None,
               prior_override=None, out_vcol: str | None = None, proc_vcol: str | None = None):
    """Returns (ratings frame, {"out": k, "proc": k, "prior": [a, b]}). Pressure flag = b.is_pressure."""
    ko = k_override or {}
    r, k_out = compute_ratings(b, cfg, role, w, "is_pressure", ko.get("out"), vcol=out_vcol)
    pr, k_proc = compute_ratings(b, cfg, role, w, "is_pressure", ko.get("proc"), vcol=proc_vcol or PROC_COL[role])
    keep = ["P_raw", "B_raw", "P_star", "B_star", "D_star", "shrink_P"]
    r = r.merge(pr[["player_id"] + keep].rename(columns={c: f"proc_{c}" for c in keep}), on="player_id", how="left")
    pid = ROLE_COLS[role][0]
    rf = b[b.is_faced] if role == "bat" else b
    r["pressure_innings"] = r.player_id.map(rf[rf.is_pressure].groupby(pid).match_id.nunique()).fillna(0).astype(int)
    mb = cfg["eligibility"][f"match_min_pressure_balls_{role}"]
    u = units_frame(b, role, w, mb)
    uu = u[u.is_unit]
    q20 = float(uu.v.quantile(0.2))
    if prior_override is None:
        cnt = uu.groupby("player_id").agg(pos=("v", lambda x: int((x > 0).sum())), n=("v", "size"))
        cnt = cnt[cnt.n >= 5]
        p = fit_beta_prior(cnt.pos.values, cnt.n.values, cfg["consistency"]["prior_method"])
        a, b_ = p["a"], p["b"]
    else:
        a, b_ = prior_override
    c = consistency(u, a, b_, q20)
    r = r.merge(c[["C", "units", "failure_rate"]], left_on="player_id", right_index=True, how="left")
    r["C"] = r.C.fillna(a / (a + b_))
    return r, {"out": k_out, "proc": k_proc, "prior": [float(a), float(b_)]}


def compose(r: pd.DataFrame, a: float) -> pd.DataFrame:
    r = r.copy()
    pp = r.proc_P_star.fillna(r.proc_B_star).fillna(0.0)
    pb = r.proc_B_star.fillna(0.0)
    r["R_P"] = a * pp + (1 - a) * r.P_star
    r["R_B"] = a * pb + (1 - a) * r.B_star
    r["D_R"] = r.R_P - r.R_B
    return r


def eligible(r: pd.DataFrame, cfg: dict, role: str, scale: float = 1.0, active=None) -> pd.Series:
    el = cfg["eligibility"]
    ok = (r.pressure_balls >= el[f"{role}_min_pressure_balls"] * scale) & \
         (r.pressure_innings >= el[f"{role}_min_pressure_{SPELL[role]}"] * scale)
    if active is not None:
        ok &= r.player_id.isin(active[active].index)
    return ok


def zscore(r: pd.DataFrame, pool: pd.Series, cols=("R_P", "D_R", "C")) -> pd.DataFrame:
    out = pd.DataFrame(index=r.index)
    for c in cols:
        mu, sd = r.loc[pool, c].mean(), r.loc[pool, c].std()
        out[f"z_{c}"] = (r[c] - mu) / sd
    return out


def rup_from_z(z: pd.DataFrame, wts: dict) -> pd.Series:
    return wts["P"] * z.z_R_P + wts["D"] * z.z_D_R + wts["C"] * z.z_C


def score(r: pd.DataFrame, pool: pd.Series, weights: dict, guard: float = 0.0) -> pd.DataFrame:
    """z-scores, RUP, guard and rank (rank only among eligible players passing the guard)."""
    r = r.copy()
    z = zscore(r, pool)
    r[z.columns] = z
    r["RUP"] = rup_from_z(r, weights)
    r["eligible"] = pool
    r["guard_ok"] = r.P_raw.fillna(-np.inf) >= guard
    r["rank_unguarded"] = r.RUP.where(pool).rank(ascending=False, method="first")
    r["rank"] = r.RUP.where(pool & r.guard_ok).rank(ascending=False, method="first")
    return r


def active_map(cfg: dict) -> pd.Series:
    """player_id -> played a match on/after active_since (user rule of 18 Nov: no squad clause)."""
    xi = load_raw(cfg, "xi")
    m = pd.read_parquet(interim(cfg, "matches"))[["match_id", "match_date"]]
    xi = xi[xi.is_played.astype(str).str.lower().eq("true")].merge(m, on="match_id")
    last = xi.groupby("player_id").match_date.max()
    return last >= pd.Timestamp(cfg["eligibility"]["active_since"])


def player_info(cfg: dict) -> pd.DataFrame:
    """Informational columns. Never used for eligibility or ranking."""
    sq = load_raw(cfg, "squads")
    test = cfg["validation"]["test_comp_prefix"]
    s26 = sq[sq.comp_name == test]
    team26 = s26.groupby("player_id").team_name.agg(lambda s: ", ".join(sorted(set(s))))
    ipl = sq[sq.comp_name.str.startswith("IPL")].copy()
    ipl["yr"] = ipl.comp_name.str[-4:].astype(int)
    ipl_teams = ipl.sort_values("yr", ascending=False).groupby("player_id").team_name.agg(
        lambda s: ", ".join(dict.fromkeys(s)))
    xi = load_raw(cfg, "xi")
    xi = xi[xi.is_played.astype(str).str.lower().eq("true")]
    teams = load_raw(cfg, "teams").drop_duplicates("team_id").set_index("team_id").team_name
    m = pd.read_parquet(interim(cfg, "matches"))[["match_id", "comp_group", "match_date"]]
    nat = xi.merge(m, on="match_id")
    nat = nat[nat.comp_group != "IPL"].assign(team=lambda d: d.team_id.map(teams).str.replace(r" A$", "", regex=True))
    nat = nat.groupby("player_id").team.agg(lambda s: s.value_counts().index[0])
    pids = pd.Index(sorted(set(sq.player_id) | set(xi.player_id)))
    info = pd.DataFrame(index=pids)
    info.index.name = "player_id"
    info["ipl2026_team"] = team26.reindex(pids)
    info["in_ipl2026_squad"] = info.ipl2026_team.notna()
    info["rr_2026"] = info.ipl2026_team.fillna("").str.contains("Rajasthan Royals")
    info["ipl_teams"] = ipl_teams.reindex(pids)
    info["nationality"] = nat.reindex(pids)
    info["overseas"] = np.where(info.nationality.isna(), "unknown (no national-team match in data)",
                                np.where(info.nationality == "India", "no", "yes"))
    info["is_pakistan"] = info.nationality.eq("Pakistan")
    ret = Path(cfg["paths"]["external"]) / "retentions_2027.csv"
    info["availability"] = np.where(info.rr_2026, "RR", np.where(~info.in_ipl2026_squad, "FREE", "UNKNOWN"))
    info["availability_source"] = "retentions file" if ret.exists() else "retentions_2027.csv missing -> UNKNOWN"
    info["teams"] = [", ".join([x for x in (n, it) if isinstance(x, str)]) for it, n in zip(info.ipl_teams, info.nationality)]
    return info.reset_index()


def weight_grid(role: str, cfg: dict, step: float = 0.05) -> list[dict]:
    mw = cfg["composite"]["min_weight"]
    fixed_c = cfg["composite"].get("bowl_fixed_C") if role == "bowl" else None
    n = int(round(1 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            wP, wD = round(i * step, 2), round(j * step, 2)
            wC = round(1 - wP - wD, 2)
            if min(wP, wD, wC) < mw - 1e-9:
                continue
            if fixed_c is not None and abs(wC - fixed_c) > 1e-9:
                continue
            out.append({"P": wP, "D": wD, "C": wC})
    return out


def default_weights(role: str, cfg: dict) -> dict:
    key = "default_weights_bowl" if role == "bowl" and "default_weights_bowl" in cfg["composite"] else "default_weights"
    return dict(cfg["composite"][key])

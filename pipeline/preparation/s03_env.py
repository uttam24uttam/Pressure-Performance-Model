"""Stage 3: scoring environment and opposition strength.

Output: interim/env.parquet (state.parquet + env columns)
        outputs/tables/env_comp_season.csv, outputs/tables/team_strength.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim

CS = ["comp_group", "season"]


def _shrunk(num, den, prior_rate, k):
    return (num + k * prior_rate) / (den + k)


def run(cfg: dict) -> None:
    header("STAGE 3 - ENVIRONMENT AND OPPOSITION")
    b = pd.read_parquet(interim(cfg, "state"))
    m = pd.read_parquet(interim(cfg, "matches"))
    k = cfg["env"]["team_strength_k"]
    lmo = cfg["env"]["leave_match_out"]
    tables = Path(cfg["paths"]["out"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    R = []

    # ---------------- scoring environment ----------------
    env = b.groupby(CS).agg(runs=("team_runs", "sum"), legal=("is_legal", "sum"),
                            wkts=("is_wicket", "sum"), matches=("match_id", "nunique"))
    env["env_rpb"] = env.runs / env.legal
    env["env_wpb"] = env.wkts / env.legal
    full1 = m[(m.max_balls_inn1 == 120) & ((m.inn1_legal == 120) | (m.inn1_wkts == 10))]
    par = full1.groupby(CS).agg(env_par1=("inn1_total", "mean"), n_full_inn1=("inn1_total", "size"))
    env = env.join(par)
    env["env_rpo"] = env.env_rpb * 6
    print("Scoring environment by comp_group x season:")
    print(env[["matches", "env_rpo", "env_par1", "n_full_inn1", "env_wpb"]].round(3).to_string())

    ipl = env.loc["IPL"]
    pre, post = ipl.loc[[2022, 2023], "env_rpb"].mean(), ipl.loc[ipl.index >= 2024, "env_rpb"].mean()
    check(post > pre * 1.05, f"IPL 2024+ inflation visible: {pre * 6:.2f} -> {post * 6:.2f} RPO "
          f"(+{(post / pre - 1):.1%}); par1 {ipl.loc[[2022, 2023], 'env_par1'].mean():.0f} -> "
          f"{ipl.loc[ipl.index >= 2024, 'env_par1'].mean():.0f}", R)
    check(env.env_par1.notna().all(), "env_par1 defined for every comp_group x season", R)
    env.reset_index().to_csv(tables / "env_comp_season.csv", index=False)
    b = b.merge(env[["env_rpb", "env_par1"]].reset_index(), on=CS, how="left")

    # ---------------- team strength (EB-shrunk, leave-match-out) ----------------
    def team_side(team_col, prefix):
        per = b.groupby(["match_id", team_col] + CS).agg(
            runs=("team_runs", "sum"), legal=("is_legal", "sum"), wkts=("is_wicket", "sum")).reset_index()
        per = per.rename(columns={team_col: "team_id"})
        season_tot = per.groupby(["team_id"] + CS)[["runs", "legal", "wkts"]].transform("sum")
        if lmo:
            num = season_tot - per[["runs", "legal", "wkts"]]
        else:
            num = season_tot
        per = per.join(env[["env_rpb", "env_wpb"]], on=CS)
        per[f"{prefix}_rpb"] = _shrunk(num.runs, num.legal, per.env_rpb, k)
        per[f"{prefix}_wpb"] = _shrunk(num.wkts, num.legal, per.env_wpb, k)
        per[f"{prefix}_rpb_rel"] = per[f"{prefix}_rpb"] / per.env_rpb
        per[f"{prefix}_wpb_rel"] = per[f"{prefix}_wpb"] / per.env_wpb
        per[f"{prefix}_balls_used"] = num.legal
        return per[["match_id", "team_id", f"{prefix}_rpb", f"{prefix}_wpb", f"{prefix}_rpb_rel",
                    f"{prefix}_wpb_rel", f"{prefix}_balls_used"]]

    bat = team_side("batting_team_id", "bat")     # how a team scores when batting
    bowl = team_side("bowling_team_id", "bowl")   # how much a team concedes when bowling

    # opposition for the batter = the bowling side's bowling strength (higher rpb = weaker attack)
    ob = bowl.rename(columns={"team_id": "bowling_team_id", "bowl_rpb": "opp_bowl_strength",
                              "bowl_wpb": "opp_bowl_wpb", "bowl_rpb_rel": "opp_bowl_strength_rel",
                              "bowl_wpb_rel": "opp_bowl_wpb_rel"}).drop(columns="bowl_balls_used")
    # opposition for the bowler = the batting side's batting strength
    oa = bat.rename(columns={"team_id": "batting_team_id", "bat_rpb": "opp_bat_strength",
                             "bat_wpb": "opp_bat_wpb", "bat_rpb_rel": "opp_bat_strength_rel",
                             "bat_wpb_rel": "opp_bat_wpb_rel"}).drop(columns="bat_balls_used")
    n0 = len(b)
    b = b.merge(ob, on=["match_id", "bowling_team_id"], how="left").merge(oa, on=["match_id", "batting_team_id"], how="left")
    check(len(b) == n0, "row count unchanged after strength joins", R)
    check(b[["opp_bowl_strength", "opp_bat_strength"]].notna().all().all(), "opposition strength on every ball", R)

    # season-level table (no leave-out) for reporting
    names = (b[["batting_team_id"]].drop_duplicates()
             .merge(m[["home_team_id", "home_team"]].drop_duplicates("home_team_id")
                    .rename(columns={"home_team_id": "batting_team_id", "home_team": "team"}),
                    on="batting_team_id", how="left"))
    st = b.groupby(["batting_team_id"] + CS).agg(bat_runs=("team_runs", "sum"), bat_legal=("is_legal", "sum")).reset_index()
    sw = b.groupby(["bowling_team_id"] + CS).agg(bowl_runs=("team_runs", "sum"), bowl_legal=("is_legal", "sum")).reset_index()
    ts = st.merge(sw.rename(columns={"bowling_team_id": "batting_team_id"}), on=["batting_team_id"] + CS, how="outer")
    ts = ts.join(env[["env_rpb"]], on=CS).merge(names, on="batting_team_id", how="left").rename(columns={"batting_team_id": "team_id"})
    ts["bat_rpo_shrunk"] = 6 * _shrunk(ts.bat_runs, ts.bat_legal, ts.env_rpb, k)
    ts["bowl_rpo_shrunk"] = 6 * _shrunk(ts.bowl_runs, ts.bowl_legal, ts.env_rpb, k)
    ts["net_rpo"] = ts.bat_rpo_shrunk - ts.bowl_rpo_shrunk
    ts.to_csv(tables / "team_strength.csv", index=False)

    header("Team strength checks")
    i26 = ts[(ts.comp_group == "IPL") & (ts.season == 2026)].sort_values("net_rpo", ascending=False)
    print("IPL 2026 team strength (season, shrunk, runs per over):")
    print(i26[["team", "bat_rpo_shrunk", "bowl_rpo_shrunk", "net_rpo", "bat_legal"]].round(2).to_string(index=False))
    t = ts[(ts.comp_group == "T20I")].groupby("team").agg(bat=("bat_rpo_shrunk", "mean"),
                                                          bowl=("bowl_rpo_shrunk", "mean"),
                                                          balls=("bat_legal", "sum"))
    t["net"] = t.bat - t.bowl
    print("\nT20I net strength (mean over seasons), top 6 / bottom 6:")
    tt = t.sort_values("net", ascending=False)
    print(pd.concat([tt.head(6), tt.tail(6)]).round(2).to_string())

    fm = set(cfg["teams"]["full_members"])
    assoc = set(cfg["teams"]["associates"])
    fm_net = t.loc[t.index.isin(fm - {"Zimbabwe", "Ireland"}), "net"].mean()
    as_net = t.loc[t.index.isin(assoc), "net"].mean()
    check(fm_net > as_net, f"major full members net stronger than associates ({fm_net:+.2f} vs {as_net:+.2f} RPO)", R)
    print(f"\n  opp_bowl_strength_rel quantiles: {b.opp_bowl_strength_rel.quantile([.05, .5, .95]).round(3).to_dict()}")
    print(f"  opp_bat_strength_rel  quantiles: {b.opp_bat_strength_rel.quantile([.05, .5, .95]).round(3).to_dict()}")
    print(f"  env_rpb on balls: {b.env_rpb.min() * 6:.2f} - {b.env_rpb.max() * 6:.2f} RPO")

    b.to_parquet(interim(cfg, "env"), index=False)
    print(f"\n  wrote {interim(cfg, 'env').name} ({len(b):,} rows, {b.shape[1]} cols); "
          f"tables/env_comp_season.csv, tables/team_strength.csv")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 3: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 3 checks failed - STOP and review.")

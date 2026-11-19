"""Stage 1: clean balls, derive per-ball accounting columns, reconstruct and validate innings totals.

Outputs: interim/balls.parquet, interim/matches.parquet
"""
import re

import numpy as np
import pandas as pd

from pipeline.common.utils import (load_raw, comp_group, header, check, CheckFailed, interim,
                    ILLEGAL_EXTRAS, BOWLER_EXTRAS, BOWLER_WICKETS)

KO_STAGES = {"Final", "Qualifier", "Eliminator", "Semi Final"}
RR_HOME_VENUES = {6, 1241}
RR_TEAM_ID = 3


def _tier(row, fm: set, asc: set) -> str:
    if row.comp_group == "IPL":
        return "IPL"
    if row.comp_group == "EMERGING":
        return "EMERGING"
    teams = {row.home_team, row.away_team}
    if any(str(t).endswith(" A") for t in teams):
        return "EMERGING"
    if teams <= fm:
        return "T20I_FM"
    return "T20I_ASSOC"          # any associate (or unknown) side involved


def _parse_result(text: str) -> tuple[str | None, float]:
    """Return (margin_kind, margin_value) from the result text: kind in {runs, wickets, None}."""
    if not isinstance(text, str):
        return None, np.nan
    mm = re.search(r"by\s+(\d+)\s+(run|wicket)", text, flags=re.I)
    if not mm:
        return None, np.nan
    return ("runs" if mm.group(2).lower() == "run" else "wickets"), float(mm.group(1))


def run(cfg: dict) -> None:
    header("STAGE 1 - CLEAN")
    b = load_raw(cfg, "balls")
    m = load_raw(cfg, "matches")
    c = load_raw(cfg, "competitions")
    tol = cfg["clean"]["margin_tolerance_runs"]
    fm, asc = set(cfg["teams"]["full_members"]), set(cfg["teams"]["associates"])

    # ---------------- matches ----------------
    n0 = len(m)
    m = m[m.match_id.isin(b.match_id.unique())].copy()
    print(f"matches: {n0} -> {len(m)} after dropping matches without ball data")
    m["match_date"] = pd.to_datetime(m["match_date"])
    m = m.merge(c[["comp_id", "comp_name", "match_format"]], on="comp_id", how="left")
    m["comp_group"] = m["comp_name"].map(comp_group)
    ipl_year = m["comp_name"].str.extract(r"IPL (\d{4})")[0].astype(float)
    m["season"] = np.where(m.comp_group == "IPL", ipl_year, m.match_date.dt.year).astype(int)
    m["tier"] = m.apply(_tier, axis=1, fm=fm, asc=asc)
    m["is_knockout"] = (m.knock_out == 1) | m.stage.isin(KO_STAGES)
    m["is_final"] = m.stage == "Final"
    m["rr_home_inferred"] = m.venue_id.isin(RR_HOME_VENUES) & (m.home_team_id == RR_TEAM_ID)

    # ---------------- balls ----------------
    if cfg["clean"]["drop_super_overs"]:
        nb = len(b)
        b = b[b.innings.isin([1, 2])].copy()
        print(f"balls: {nb:,} -> {len(b):,} after dropping super-over innings 3/4")
    b = b[b.match_id.isin(m.match_id)]
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    for col in ["is_wicket", "is_four", "is_six"]:
        b[col] = b[col].astype(str).str.lower().eq("true")
    ext = b.extras_type
    extra_runs = b.extras1_runs + b.extras_2_runs
    b["team_runs"] = b.runs + extra_runs
    b["is_legal"] = ~ext.isin(ILLEGAL_EXTRAS)
    b["is_faced"] = ext.ne("wide")
    b["bowler_conceded"] = b.runs + np.where(ext.isin(BOWLER_EXTRAS), extra_runs, 0)
    dt = b.dismissal_type
    b["bowler_wicket"] = b.is_wicket & dt.isin(BOWLER_WICKETS)
    not_retired = dt.ne("Retire Out")
    b["batter_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id) & not_retired
    b["nonstriker_out"] = b.is_wicket & (b.dismissed_player_id == b.non_striker_id) & not_retired
    b = b.drop(columns=["bowl_speed"])          # unreliable: zero for 39% of balls, impossible values
    b = b.merge(m[["match_id", "comp_id", "comp_name", "comp_group", "season", "tier", "match_date",
                   "is_knockout", "is_final"]], on="match_id", how="left")

    # ---------------- innings totals ----------------
    inn = b.groupby(["match_id", "innings"]).agg(
        total=("team_runs", "sum"), wkts=("is_wicket", "sum"), legal=("is_legal", "sum"),
        bat_team=("batting_team_id", "first"), bowl_team=("bowling_team_id", "first"),
        n_bat_teams=("batting_team_id", "nunique")).reset_index()
    wide = inn.pivot(index="match_id", columns="innings")
    wide.columns = [f"inn{i}_{k}" for k, i in wide.columns]
    m = m.merge(wide.reset_index(), on="match_id", how="left")
    m["bat_first_team_id"] = m.inn1_bat_team
    m["chase_team_id"] = m.inn1_bowl_team
    m["has_inn2"] = m.inn2_total.notna()

    m["result_type"] = np.where(m.win_team_id.isna(), "no_result", "decisive")
    tie_text = m.result.str.contains("Tie|Super Over", case=False, na=False)
    m.loc[tie_text & m.has_inn2 & (m.inn1_total == m.inn2_total), "result_type"] = "tie"
    m["target"] = np.where(m.revised_target.notna(), m.revised_target, m.inn1_total + 1)
    m["is_dls"] = m.revised_target.notna() | m.result.str.contains("D/L|DLS", case=False, na=False)
    m["max_balls_inn1"] = m.first_inn_revised_overs.fillna(120)
    m["max_balls_inn2"] = m.second_inn_revised_overs.fillna(m.first_inn_revised_overs).fillna(120)
    m["bat_first_won"] = np.where(m.result_type == "decisive",
                                  (m.win_team_id == m.bat_first_team_id).astype(float), np.nan)
    pr = m.result.map(_parse_result)
    m["margin_kind"] = pr.str[0]
    m["margin_value"] = pr.str[1]

    # ---------------- validation against official result ----------------
    header("Margin validation")
    R = []
    dec = m.result_type == "decisive"
    winner_known = m.win_team_id.isin(m.bat_first_team_id) | m.win_team_id.isin(m.chase_team_id)
    bad_team = dec & ~((m.win_team_id == m.bat_first_team_id) | (m.win_team_id == m.chase_team_id))
    print(f"  decisive matches: {dec.sum()}, ties: {(m.result_type == 'tie').sum()}, "
          f"no-result: {(m.result_type == 'no_result').sum()}")
    check(bad_team.sum() == 0, f"win_team_id is one of the two batting sides: {bad_team.sum()} violations", R)

    # Rain stopped the chase and the result came from DLS par, but the revised fields are missing.
    # Real results (not data errors) whose margin cannot be checked: keep the balls for player
    # ratings, exclude from WP training (the label is a mid-chase par calculation, not a finished chase).
    m["dls_par_finish"] = (dec & m.is_dls & m.revised_target.isna() & m.has_inn2
                           & (m.inn2_legal < m.max_balls_inn2) & (m.inn2_wkts < 10)
                           & (m.inn2_total < m.target))
    print(f"  DLS par finishes (chase stopped by rain, revised fields missing): "
          f"{m.dls_par_finish.sum()} -> {m.loc[m.dls_par_finish, 'match_id'].tolist()}")
    dec = dec & ~m.dls_par_finish

    m["margin_ok"] = np.nan
    # (a) bat-first win by N runs: (target - 1) - inn2 == N  (target-1 == inn1 total when no DLS)
    a = dec & (m.margin_kind == "runs") & m.has_inn2
    m.loc[a, "margin_ok"] = ((m.target - 1 - m.inn2_total - m.margin_value).abs() <= tol)[a].astype(float)
    # (b) chasing win: inn2 >= target
    bmask = dec & (m.win_team_id == m.chase_team_id) & m.has_inn2
    m.loc[bmask, "margin_ok"] = (m.inn2_total >= m.target)[bmask].astype(float)
    # (b') wickets margin: wickets fallen == 10 - N   (informational)
    wk = dec & (m.margin_kind == "wickets") & m.has_inn2
    wk_ok = ((10 - m.inn2_wkts) == m.margin_value)[wk]
    # consistency of margin type with winner side
    side_bad = dec & (((m.margin_kind == "runs") & (m.win_team_id != m.bat_first_team_id)) |
                      ((m.margin_kind == "wickets") & (m.win_team_id != m.chase_team_id)))
    # (c) bat-first win with no parsable margin: chase must fall short
    cmask = dec & (m.win_team_id == m.bat_first_team_id) & m.margin_kind.isna() & m.has_inn2
    m.loc[cmask, "margin_ok"] = (m.inn2_total < m.target)[cmask].astype(float)

    chk = m.margin_ok.notna()
    rate = m.loc[chk, "margin_ok"].mean()
    rr = m.loc[a, "margin_ok"]
    print(f"  bat-first 'by N runs' checks: {int(rr.sum())}/{len(rr)} pass")
    print(f"  chasing-win checks (inn2 >= target): {int(m.loc[bmask, 'margin_ok'].sum())}/{bmask.sum()} pass")
    print(f"  wickets-margin checks (10 - wkts fallen == N, informational): {int(wk_ok.sum())}/{len(wk_ok)}")
    print(f"  margin type inconsistent with winner side: {side_bad.sum()}")
    print(f"  decisive matches without 2nd innings data: {(dec & ~m.has_inn2).sum()}")
    check(rate >= 0.99, f"margin-check pass rate {rate:.4f} over {chk.sum()} checkable matches (gate >= 0.99)", R)

    if (po := b.groupby(["match_id", "innings", "overid"]).is_legal.sum()).gt(6).any():
        print(f"  overs with 7+ legal balls (umpire miscount, kept): {po[po > 6].index.tolist()}")
    m["data_suspect"] = (m.margin_ok == 0) | side_bad | bad_team
    # innings with two batting teams would break everything downstream
    m.loc[(m.inn1_n_bat_teams > 1) | (m.inn2_n_bat_teams > 1), "data_suspect"] = True
    sus = m[m.data_suspect]
    print(f"\n  data_suspect matches: {len(sus)}")
    if len(sus):
        cols = ["match_id", "match_date", "comp_name", "inn1_total", "inn2_total", "target",
                "inn2_wkts", "revised_target", "result"]
        print(sus[cols].to_string(index=False))
    if not wk_ok.all():
        print("\n  wickets-margin mismatches (not flagged; often retired/absent batters):")
        ids = wk_ok[~wk_ok].index
        print(m.loc[ids, ["match_id", "inn2_wkts", "margin_value", "result"]].to_string(index=False))

    # ---------------- per-over legality ----------------
    header("Over legality")
    po = b.groupby(["match_id", "innings", "overid"]).is_legal.sum()
    last = b.groupby(["match_id", "innings"]).overid.max()
    po_df = po.reset_index().merge(last.rename("last_over").reset_index(), on=["match_id", "innings"])
    completed = po_df[po_df.overid < po_df.last_over]
    pct6 = (completed.is_legal == 6).mean()
    print(f"  completed overs with 6 legal balls: {pct6:.4%} ({len(completed):,} overs); all overs: {(po == 6).mean():.4%}")
    check(pct6 >= 0.99, "99%+ completed overs have 6 legal balls", R)
    over7 = po_df[po_df.is_legal > 6]
    if len(over7):
        print(f"  overs with >6 legal balls: {len(over7)}")

    # ---------------- per-ball sanity ----------------
    header("Per-ball columns")
    print(f"  balls {len(b):,} | legal {b.is_legal.mean():.4f} | faced {b.is_faced.mean():.4f}")
    print(f"  wickets {b.is_wicket.sum():,} | bowler wickets {b.bowler_wicket.sum():,} | "
          f"striker outs {b.batter_out.sum():,} | non-striker outs {b.nonstriker_out.sum():,} | "
          f"retired out {(b.dismissal_type == 'Retire Out').sum()}")
    print(f"  team runs {b.team_runs.sum():,} | off bat {b.runs.sum():,} | bowler conceded {b.bowler_conceded.sum():,}")
    print("  tiers (matches):", dict(m.tier.value_counts()))
    print("  knockouts:", int(m.is_knockout.sum()), " finals:", int(m.is_final.sum()),
          " RR home (inferred):", int(m.rr_home_inferred.sum()))
    print("  T20I_ASSOC example fixtures:",
          m[m.tier == "T20I_ASSOC"][["home_team", "away_team"]].drop_duplicates().head(6).values.tolist())
    print("  EMERGING sides:", sorted(set(m[m.tier == "EMERGING"].home_team) | set(m[m.tier == "EMERGING"].away_team)))

    b = b.merge(m[["match_id", "data_suspect", "result_type"]], on="match_id", how="left")
    b.to_parquet(interim(cfg, "balls"), index=False)
    m.to_parquet(interim(cfg, "matches"), index=False)
    print(f"\n  wrote {interim(cfg, 'balls').name} ({len(b):,} rows), {interim(cfg, 'matches').name} ({len(m)} rows)")

    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 1: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 1 checks failed - STOP and review.")

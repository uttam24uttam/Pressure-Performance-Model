"""Stage 0: load raw data, describe it, and assert the documented data-pack facts still hold."""
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.common.utils import load_raw, comp_group, header, check, CheckFailed, RAW_FILES, ILLEGAL_EXTRAS


def run(cfg: dict) -> None:
    header("STAGE 0 - DATA AUDIT")
    raw = Path(cfg["paths"]["raw"])
    for f in RAW_FILES.values():
        if not (raw / f).exists():
            raise CheckFailed(f"missing raw file {f}")

    b = load_raw(cfg, "balls"); m = load_raw(cfg, "matches"); c = load_raw(cfg, "competitions")
    t = load_raw(cfg, "teams"); xi = load_raw(cfg, "xi"); sq = load_raw(cfg, "squads")
    print("Row counts:")
    for n, d in [("ball_by_ball", b), ("matches", m), ("competitions", c), ("teams", t),
                 ("playing_eleven", xi), ("comp_team_player_mapping", sq)]:
        print(f"  {n:28s} {len(d):>8,}")
    m["match_date"] = pd.to_datetime(m["match_date"])
    print(f"Match date range: {m.match_date.min().date()} -> {m.match_date.max().date()}")

    m = m.merge(c[["comp_id", "comp_name", "match_format"]], on="comp_id", how="left")
    m["comp_group"] = m["comp_name"].map(comp_group)
    m["season"] = m["match_date"].dt.year
    m["has_balls"] = m.match_id.isin(b.match_id.unique())
    print("\nMatches with ball data by comp_group x season:")
    print(m[m.has_balls].pivot_table(index="comp_group", columns="season", values="match_id",
                                     aggfunc="count", fill_value=0, margins=True))
    print("\nEmerging competitions:", sorted(m.loc[m.comp_group == "EMERGING", "comp_name"].unique()))

    header("Data-pack fact checks")
    R = []
    check(len(b) == 200_999, f"ball_by_ball rows = 200,999 (got {len(b):,})", R)
    check(b.match_id.nunique() == 867, f"867 matches with balls (got {b.match_id.nunique()})", R)
    check(len(m) == 883, f"matches.csv rows = 883 (got {len(m)})", R)
    check(len(c) == 141, f"competitions rows = 141 (got {len(c)})", R)
    # 6,414 is the file's line count incl. header; 6,413 data rows.
    check(len(sq) == 6413, f"mapping data rows = 6,413 (6,414 lines include the header) (got {len(sq)})", R)
    check((~m.has_balls).sum() == 16, f"16 matches without balls (got {(~m.has_balls).sum()})", R)
    n_ipl = m[m.has_balls & (m.comp_group == "IPL")].match_id.nunique()
    check(n_ipl == 367, f"367 IPL matches with balls (got {n_ipl})", R)
    n_em = m[m.has_balls & (m.comp_group == "EMERGING")].match_id.nunique()
    check(n_em == 30, f"30 emerging-tournament matches with balls (got {n_em})", R)
    check(set(b.innings.unique()) == {1, 2, 3, 4},
          f"innings values {{1,2,3,4}} (super overs present): {sorted(b.innings.unique())}", R)
    check(m.comp_name.str.contains("World Cup", case=False).sum() == 0, "no ICC T20 World Cup matches", R)

    ext = b.extras_type
    print("  extras_type values:", {str(k): int(v) for k, v in ext.value_counts(dropna=False).items()})
    known = {"wide", "Overstep", "other", "byes", "legbyes", "Penalty"}
    check(set(ext.dropna().unique()) <= known, "extras_type in known set ('None' string read as NaN)", R)
    share = (b.loc[ext.isna(), "extras1_runs"] == 0).mean()
    check(share > 0.999, f"extras1_runs == 0 on balls with no extras: {share:.5f}  (broken 10-Nov pack fails this)", R)
    check((b.loc[ext == "wide", "extras1_runs"] >= 1).mean() > 0.99, "extras1_runs >= 1 on wides", R)
    check((b.loc[ext == "Overstep", "extras1_runs"] >= 1).mean() > 0.95, "extras1_runs >= 1 on Overstep no-balls", R)
    check((b.loc[ext == "wide", "runs"] == 0).mean() > 0.999, "runs off bat = 0 on wides", R)
    check((b.loc[ext.isin(["byes", "legbyes"]), "runs"] == 0).mean() > 0.99, "runs off bat = 0 on byes/legbyes", R)

    s = b.sort_values(["match_id", "innings", "innings_ball_num"])
    d = s.groupby(["match_id", "innings"])["innings_ball_num"].diff()
    first = s.groupby(["match_id", "innings"])["innings_ball_num"].transform("min")
    check(((d == 1) | d.isna()).mean() > 0.999 and (first == 1).mean() > 0.99,
          f"innings_ball_num sequential from 1 per innings ({((d == 1) | d.isna()).mean():.5f})", R)
    check(b.overid.between(1, 20).all(), "overid in 1..20", R)

    bb = b[b.innings.isin([1, 2])].copy()
    bb["is_legal"] = ~bb.extras_type.isin(ILLEGAL_EXTRAS)
    per_over = bb.groupby(["match_id", "innings", "overid"])["is_legal"].sum()
    last = bb.groupby(["match_id", "innings"])["overid"].transform("max")
    non_last = bb[bb.overid < last].groupby(["match_id", "innings", "overid"])["is_legal"].sum()
    print(f"  overs with 6 legal balls: all overs {(per_over == 6).mean():.4f}, "
          f"excluding innings-ending overs {(non_last == 6).mean():.4f}")
    check((non_last == 6).mean() >= 0.99, "99%+ of completed overs have exactly 6 legal balls", R)

    dt = set(b.dismissal_type.dropna().unique())
    exp_dt = {"Caught", "Bowled", "LBW", "RunOut", "Stump", "Hit Wicket", "Retire Out",
              "Obstracting the field", "Obstructing the field"}
    check(dt <= exp_dt, f"dismissal types known: {sorted(dt)}", R)
    w = b[b.is_wicket.astype(bool)]
    ns = (w.dismissed_player_id == w.non_striker_id).mean()
    check(0.01 < ns < 0.05, f"non-striker dismissals ~2.5% of wickets: {ns:.4f}", R)
    check(set(b.pace_or_spin.dropna().unique()) <= {"Pace", "Spin"},
          f"pace_or_spin values {sorted(b.pace_or_spin.dropna().unique())}", R)
    bs = b.bowl_speed
    print(f"  bowl_speed (unused junk): zeros {(bs == 0).mean():.1%}, negatives {(bs < 0).sum()}, >1000 {(bs > 1000).sum()}")

    nr = m.win_team_id.isna().sum()
    check(35 <= nr <= 45, f"~40 no-result matches (win_team_id NaN): {nr}", R)
    # Regulation ties: "Tie" in text OR decided by super over. win_team_id holds the SUPER-OVER
    # winner for these, so Stage 1/2 must set bat_team_won = NaN for all of them.
    tie_mask = m.result.str.contains("Tie|Super Over", case=False, na=False)
    so_matches = set(b.loc[b.innings > 2, "match_id"])
    print(f"  regulation-tied matches: {tie_mask.sum()} "
          f"({m[tie_mask].win_team_id.notna().sum()} carry a super-over winner in win_team_id)")
    check(set(m.loc[tie_mask, "match_id"]) >= so_matches, "every super-over match is flagged as tied", R)
    bb2 = b[b.innings.isin([1, 2])]
    tot = (bb2.runs + bb2.extras1_runs + bb2.extras_2_runs).groupby([bb2.match_id, bb2.innings]).sum().unstack()
    tied_ids = m.loc[tie_mask, "match_id"]
    eq = (tot.loc[tot.index.intersection(tied_ids), 1] == tot.loc[tot.index.intersection(tied_ids), 2])
    print(f"  tie-flagged matches: {tie_mask.sum()}, with equal reconstructed totals: {eq.sum()}; "
          f"unequal (ambiguous, win_team_id NaN): {list(eq[~eq].index)}")
    check(eq.sum() == 8, f"8 genuine ties incl. super-over games (including super-over games): {eq.sum()}", R)
    rev = m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
    rev_dec = rev & m.win_team_id.notna()
    print(f"  rain-revised matches: {rev.sum()} total, {rev_dec.sum()} decisive, {(rev & m.win_team_id.isna()).sum()} no-result")
    check(20 <= rev_dec.sum() <= 30, f"~25 decisive rain-revised matches: {rev_dec.sum()}", R)
    check(m[["first_inn_revised_overs", "second_inn_revised_overs"]].max().max() <= 120,
          "revised overs stored in balls (max <= 120)", R)
    print("  stage values:", dict(m.stage.value_counts()))
    rr = t[t.team_name == "Rajasthan Royals"].team_id.unique()
    hv = m[m.home_team_id.isin(rr)].venue_id.value_counts().head(3)
    print(f"  RR team_id(s): {list(rr)}; top RR home venue_ids: {dict(hv)}")
    check({6, 1241} <= set(hv.index), "RR home venues include 6 and 1241", R)

    names = pd.concat([b[["striker_id", "striker_name"]].set_axis(["pid", "name"], axis=1),
                       b[["non_striker_id", "non_striker_name"]].set_axis(["pid", "name"], axis=1),
                       b[["bowler_id", "bowler_name"]].set_axis(["pid", "name"], axis=1)]).drop_duplicates()
    dup = names.groupby("name").pid.nunique()
    dup = dup[dup > 1]
    print(f"  names mapping to >1 player_id: {dict(dup)}")
    check({"Abdul Samad", "Haider Ali", "Basil Hameed"} <= set(dup.index),
          "known duplicate names present (join on id only)", R)
    jb = names[names.name == "Jos Buttler"].pid.unique()
    check(list(jb) == [813], f"Jos Buttler = 813: {list(jb)}", R)

    tm = set(t.team_name.unique())
    fm, asc = set(cfg["teams"]["full_members"]), set(cfg["teams"]["associates"])
    ipl_teams = set(sq[sq.comp_name.str.startswith("IPL")].team_name)
    uncl = sorted(tm - fm - asc - ipl_teams - {x for x in tm if x.endswith(" A")})
    print(f"  team names outside the Section 3 taxonomy: {uncl}")

    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 0: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} audit checks failed - STOP and review.")

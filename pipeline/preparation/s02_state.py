"""Stage 2: match state before every delivery (only information known before the ball).

Output: interim/state.parquet  (balls.parquet columns + state columns)
"""
import numpy as np
import pandas as pd

from pipeline.common.utils import header, check, CheckFailed, interim, load_raw

KEYS = ["match_id", "innings"]


def phase_of(over: pd.Series, phases: dict) -> pd.Series:
    out = pd.Series(index=over.index, dtype=object)
    for name, (lo, hi) in phases.items():
        out[(over >= lo) & (over <= hi)] = name
    return out


def _series_context(cfg: dict, m: pd.DataFrame) -> pd.DataFrame:
    """series_decider / dead_rubber for bilateral T20I series (2 teams in the comp).

    Uses raw matches.csv (incl. matches without ball data) so earlier results are complete.
    The number of matches listed for the comp is taken as the scheduled length; abandoned
    fixtures are listed as no-result, so this is a reasonable proxy. Anything else -> NaN.
    """
    raw = load_raw(cfg, "matches")
    raw["match_date"] = pd.to_datetime(raw["match_date"])
    raw = raw[raw.comp_id.isin(m.loc[m.comp_group == "T20I", "comp_id"])]
    rows = []
    for cid, g in raw.groupby("comp_id"):
        teams = set(g.home_team_id) | set(g.away_team_id)
        g = g.sort_values(["match_date", "match_number", "match_id"])
        n = len(g)
        if len(teams) != 2 or n < 2:
            for mid in g.match_id:
                rows.append((mid, np.nan, np.nan))
            continue
        t1, t2 = sorted(teams)
        w1 = w2 = 0
        for i, r in enumerate(g.itertuples()):
            remaining = n - i                      # matches left incl. this one
            # dead rubber: trailing side cannot even draw the series by winning every remaining match
            decided = (w1 > w2 + remaining) or (w2 > w1 + remaining)
            decider = (i == n - 1) and (w1 == w2)
            rows.append((r.match_id, float(decider), float(decided)))
            if r.win_team_id == t1:
                w1 += 1
            elif r.win_team_id == t2:
                w2 += 1
    return pd.DataFrame(rows, columns=["match_id", "series_decider", "dead_rubber"])


def run(cfg: dict) -> None:
    header("STAGE 2 - MATCH STATE")
    b = pd.read_parquet(interim(cfg, "balls"))
    m = pd.read_parquet(interim(cfg, "matches"))
    b = b.sort_values(KEYS + ["innings_ball_num"]).reset_index(drop=True)
    g = b.groupby(KEYS, sort=False)

    # --- cumulative state before the ball
    b["score_before"] = g.team_runs.cumsum() - b.team_runs
    b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
    b["legal_before"] = g.is_legal.cumsum() - b.is_legal.astype(int)
    mx = m.set_index("match_id")
    b["max_balls"] = np.where(b.innings == 1, b.match_id.map(mx.max_balls_inn1),
                              b.match_id.map(mx.max_balls_inn2)).astype(float)
    b["balls_left"] = b.max_balls - b.legal_before
    b["over_num"] = b.overid
    b["phase"] = phase_of(b.over_num, cfg["phases"])
    b["crr"] = b.score_before * 6 / b.legal_before.clip(lower=1)
    b.loc[b.legal_before == 0, "crr"] = np.nan

    # --- chase state
    inn2 = b.innings == 2
    b["inn1_total"] = b.match_id.map(mx.inn1_total)
    b["target"] = np.where(inn2, b.match_id.map(mx.target), np.nan)
    b["runs_needed"] = b.target - b.score_before
    b["rrr"] = b.runs_needed / b.balls_left.clip(lower=1) * 6

    # --- partnership / collapse
    gp = b.groupby(KEYS + ["wkts_before"], sort=False)
    b["partnership_runs"] = b.score_before - gp.score_before.transform("min")
    b["partnership_balls"] = b.legal_before - gp.legal_before.transform("min")
    b["balls_since_wicket"] = b.partnership_balls
    w18 = np.empty(len(b))
    for idx in g.indices.values():
        lb = b.legal_before.values[idx]
        wb = b.wkts_before.values[idx]
        j = np.searchsorted(lb, lb - 18, side="left")       # first ball with legal_before >= L-18
        w18[idx] = wb - wb[j]
    b["wkts_last_18_balls"] = w18.astype(int)

    # --- batter familiarity
    gs = b.groupby(KEYS + ["striker_id"], sort=False)
    b["striker_balls_so_far"] = gs.is_faced.cumsum() - b.is_faced.astype(int)
    ev = b[KEYS + ["innings_ball_num", "striker_id"]].copy()
    ev["faced_after"] = gs.is_faced.cumsum()
    ev = ev.rename(columns={"striker_id": "non_striker_id"}).sort_values("innings_ball_num")
    q = b[KEYS + ["innings_ball_num", "non_striker_id"]].reset_index().sort_values("innings_ball_num")
    q = pd.merge_asof(q, ev, on="innings_ball_num", by=KEYS + ["non_striker_id"],
                      allow_exact_matches=False, direction="backward")
    b["nonstriker_balls_so_far"] = q.set_index("index").faced_after.reindex(b.index).fillna(0).astype(int)

    # --- match context
    sc = _series_context(cfg, m)
    b = b.merge(sc, on="match_id", how="left")
    win = b.match_id.map(mx.win_team_id)
    rtype = b.match_id.map(mx.result_type)
    b["bat_team_won"] = np.where(rtype == "decisive", (win == b.batting_team_id).astype(float), np.nan)
    b["dls_par_finish"] = b.match_id.map(mx.dls_par_finish).astype(bool)
    b["is_dls"] = b.match_id.map(mx.is_dls).astype(bool)

    # ================= sanity =================
    R = []
    header("Sanity checks")
    first = b.groupby(KEYS).head(1)
    check((first.score_before == 0).all() and (first.wkts_before == 0).all() and (first.legal_before == 0).all(),
          "first ball of every innings starts at 0/0, 0 legal balls", R)
    last = b.groupby(KEYS).tail(1).set_index(KEYS)
    tot = m.set_index("match_id")
    fin = last.score_before + last.team_runs
    exp = np.where(last.index.get_level_values(1) == 1,
                   tot.inn1_total.reindex(last.index.get_level_values(0)).values,
                   tot.inn2_total.reindex(last.index.get_level_values(0)).values)
    check(np.allclose(fin.values, exp), "score_before + last-ball runs == innings total (all innings)", R)
    check(b.wkts_before.max() <= 10, f"wkts_before <= 10 (max {b.wkts_before.max()})", R)
    neg = (b.balls_left <= 0).sum()
    print(f"  balls with balls_left <= 0 (revised-overs edge cases): {neg}")
    if neg:
        print(b.loc[b.balls_left <= 0, ["match_id", "innings", "legal_before", "max_balls"]]
              .drop_duplicates(["match_id", "innings"]).to_string(index=False))

    # final-ball chase consistency
    l2 = b[inn2].groupby("match_id").tail(1).set_index("match_id")
    md = m.set_index("match_id")
    ok_ids = md.index[(md.result_type == "decisive") & ~md.is_dls & ~md.data_suspect]
    l2 = l2.loc[l2.index.intersection(ok_ids)]
    need_after = l2.runs_needed - l2.team_runs
    chase_won = l2.bat_team_won == 1
    agree = (need_after <= 0) == chase_won
    rate = agree.mean()
    check(rate >= 0.99, f"inn-2 final-ball runs_needed consistent with result: {agree.sum()}/{len(agree)} "
          f"({rate:.4%}) decisive non-DLS matches", R)
    if not agree.all():
        print(l2.loc[~agree, ["runs_needed", "team_runs", "bat_team_won", "legal_before", "wkts_before"]])

    ties = b[b.match_id.isin(md.index[md.result_type == "tie"]) & inn2].groupby("match_id").tail(1)
    check(((ties.runs_needed - ties.team_runs) == 1).all(), f"ties end 1 run short of target ({len(ties)} matches)", R)

    header("State summaries")
    print(f"  bat_team_won: {b.bat_team_won.notna().mean():.2%} of balls labelled; "
          f"NaN on {b.loc[b.bat_team_won.isna(), 'match_id'].nunique()} matches (NR / ties)")
    print("  phase share of balls:", (b.phase.value_counts(normalize=True).round(3)).to_dict())
    print("  wkts_last_18_balls distribution:", b.wkts_last_18_balls.value_counts().sort_index().to_dict())
    print(f"  new batter (<10 balls faced) share: {(b.striker_balls_so_far < 10).mean():.2%}")
    print(f"  inn-2 rrr quantiles: {b.loc[inn2, 'rrr'].quantile([.05, .25, .5, .75, .95]).round(2).to_dict()}")
    print(f"  max partnership: {b.partnership_runs.max()} runs, {b.partnership_balls.max()} balls")
    sc_m = sc.merge(m[["match_id"]], on="match_id")
    print(f"  T20I series context: deciders {int(sc_m.series_decider.sum())}, dead rubbers "
          f"{int(sc_m.dead_rubber.sum())}, NaN (multi-team / one-off) {int(sc_m.series_decider.isna().sum())} "
          f"of {len(sc_m)} T20I matches")
    print("  knockout balls:", int(b.is_knockout.sum()), " final balls:", int(b.is_final.sum()))

    b.to_parquet(interim(cfg, "state"), index=False)
    print(f"\n  wrote {interim(cfg, 'state').name} ({len(b):,} rows, {b.shape[1]} cols)")
    n_fail = sum(not ok for ok, _ in R)
    print(f"\nStage 2: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 2 checks failed - STOP and review.")

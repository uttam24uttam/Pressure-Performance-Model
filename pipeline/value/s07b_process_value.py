"""Stage 7b: per-ball process values (runs, same wicket run-value RV as Stage 7).

  bowl_exec            = (xR_type - xR_deliv) + RV (xW_deliv - xW_type)                all deliveries
                         (type baseline = context + bowling_type + batting_type)
  bowl_exec_literal    = (xR_ctx_bowl - xR_deliv) + RV (xW_deliv - xW_ctx_bowl)        user formula, Stage 12 variant
  bat_difficulty       = (xR_deliv_bat - xR_ctx_bat) - RV (xO_deliv_bat - xO_ctx_bat)   balls faced
  bat_decision         = (xR_shot - xR_deliv_bat) - RV (xO_shot - xO_deliv_bat)         balls faced
  bat_adjusted_outcome = v_bat - bat_difficulty                                         balls faced
Positive = good for the player in every column (bat_difficulty > 0 means an EASY ball: the delivery was worth
more to the batter than the context alone predicted). Fine-label versions carry the suffix _fine.

Outputs: interim/process_value.parquet, tables/process_model_skill.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_poisson_deviance

from pipeline.common.utils import header, check, CheckFailed, interim
from pipeline.common.labels import add_labels


def values(b: pd.DataFrame, pe: pd.DataFrame, sfx: str = "") -> pd.DataFrame:
    g = lambda c: pe[f"{c}{sfx}"].values
    RV = b.RV.values
    faced = b.is_faced.values
    out = pd.DataFrame(index=b.index)
    out[f"bowl_exec{sfx}"] = (pe.xR_type.values - g("xR_deliv")) + RV * (g("xW_deliv") - pe.xW_type.values)
    out[f"bowl_exec_literal{sfx}"] = (b.xR_bowl.values - g("xR_deliv")) + RV * (g("xW_deliv") - b.xW_bowl.values)
    diff = (g("xR_deliv_bat") - b.xR_bat.values) - RV * (g("xO_deliv_bat") - b.xO_bat.values)
    dec = (g("xR_shot") - g("xR_deliv_bat")) - RV * (g("xO_shot") - g("xO_deliv_bat"))
    out[f"bat_difficulty{sfx}"] = np.where(faced, diff, np.nan)
    out[f"bat_decision{sfx}"] = np.where(faced, dec, np.nan)
    out[f"bat_adjusted_outcome{sfx}"] = np.where(faced, b.v_bat.values - diff, np.nan)
    return out


def run(cfg: dict) -> None:
    header("STAGE 7b - PROCESS VALUES")
    R = []
    b = pd.read_parquet(interim(cfg, "ball_value"))
    pe = pd.read_parquet(interim(cfg, "process_expected"))
    check((pe[["match_id", "innings", "innings_ball_num"]].values == b[["match_id", "innings", "innings_ball_num"]].values).all(),
          "process expectations align row-by-row with ball_value", R)
    b = add_labels(b)
    v = pd.concat([values(b, pe), values(b, pe, "_fine")], axis=1)
    ok = ~b.data_suspect.values
    faced = b.is_faced.values & ok
    header("Means per ball (should be ~0)")
    for c in ["bowl_exec", "bowl_exec_literal", "bat_difficulty", "bat_decision", "bat_adjusted_outcome"]:
        for s in ("", "_fine"):
            mk = faced if c.startswith("bat") else ok
            mu = float(np.nanmean(v[c + s].values[mk]))
            sd = float(np.nanstd(v[c + s].values[mk]))
            check(abs(mu) <= 0.02, f"mean {c + s} = {mu:+.4f} runs/ball (sd {sd:.3f})", R)
    print(f"  for reference: v_bat sd {b.v_bat[faced].std():.3f}, v_bowl sd {b.v_bowl[ok].std():.3f}")

    # model skill: how much the delivery / stroke labels add over context alone
    header("Skill added by process labels (OOF, family models)")
    rows = []
    for nm, (ctx, prc, obs, mk, kind) in {
        "bowler runs: context -> + bowler type": ("xR_bowl", "xR_type", "bowler_conceded", ok, "p"),
        "bowler runs: + type -> + delivery": ("xR_type", "xR_deliv", "bowler_conceded", ok, "p"),
        "bowler runs: context -> + delivery": ("xR_bowl", "xR_deliv", "bowler_conceded", ok, "p"),
        "bowler wicket: context -> + delivery": ("xW_bowl", "xW_deliv", "bowler_wicket", ok, "b"),
        "batter runs: context -> + delivery": ("xR_bat", "xR_deliv_bat", "runs", faced, "p"),
        "batter out: context -> + delivery": ("xO_bat", "xO_deliv_bat", "batter_out", faced, "b"),
        "batter runs: + delivery -> + stroke": ("xR_deliv_bat", "xR_shot", "runs", faced, "p"),
        "batter out: + delivery -> + stroke": ("xO_deliv_bat", "xO_shot", "batter_out", faced, "b"),
    }.items():
        y = b[obs].values[mk].astype(float)
        p0 = (b[ctx] if ctx in b else pe[ctx]).values[mk]
        p1 = (pe[prc] if prc in pe else b[prc]).values[mk]
        if kind == "p":
            s0, s1, lab = mean_poisson_deviance(y, np.clip(p0, 1e-6, None)), mean_poisson_deviance(y, np.clip(p1, 1e-6, None)), "Poisson deviance"
        else:
            s0, s1, lab = log_loss(y, np.clip(p0, 1e-6, 1 - 1e-6)), log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6)), "log-loss"
        rows.append({"comparison": nm, "metric": lab, "before": s0, "after": s1, "improvement": 1 - s1 / s0})
        print(f"  {nm:40s} {lab:16s} {s0:.4f} -> {s1:.4f} ({1 - s1 / s0:+.2%})")
    pd.DataFrame(rows).to_csv(Path(cfg["paths"]["out"]) / "tables" / "process_model_skill.csv", index=False)

    keep = ["match_id", "innings", "innings_ball_num", "deliv_fam", "stroke_fam", "deliv_fine", "stroke_fine",
            "is_yorker", "is_full_toss", "is_wide_nb", "is_pace_off", "is_bouncer", "is_aerial", "is_defence"]
    out = pd.concat([b[keep], v, pe.drop(columns=["match_id", "innings", "innings_ball_num"])], axis=1)
    out.to_parquet(interim(cfg, "process_value"), index=False)
    print(f"\n  wrote process_value.parquet ({len(out):,} rows), tables/process_model_skill.csv")
    n_fail = sum(not ok_ for ok_, _ in R)
    print(f"\nStage 7b: {len(R) - n_fail}/{len(R)} checks passed.")
    if n_fail:
        raise CheckFailed(f"{n_fail} Stage 7b checks failed - review.")

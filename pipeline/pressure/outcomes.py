"""Ball-outcome model shared by Stage 4 (endgame DP) and Stage 5 (leverage).

Outcomes: dot, 1, 2, 3, 4, 6, wicket (0 runs), wide (+1 run, ball not consumed).
p(o) is conditioned on innings x phase x wickets bucket, plus a required-rate band in innings 2,
and each bucket is smoothed toward its parent bucket.
"""
import numpy as np
import pandas as pd

# outcome -> (runs, wicket, consumes a legal ball)
OUTCOMES = {"dot": (0, 0, 1), "1": (1, 0, 1), "2": (2, 0, 1), "3": (3, 0, 1), "4": (4, 0, 1),
            "6": (6, 0, 1), "wicket": (0, 1, 1), "wide": (1, 0, 0)}
WKT_BINS = [-1, 1, 3, 5, 10]
WKT_LABELS = ["0-1", "2-3", "4-5", "6+"]
RRR_BINS = [-np.inf, 8, 10, 12, np.inf]
RRR_LABELS = ["<8", "8-10", "10-12", ">12"]
BUCKET_KEYS = ["innings", "phase", "wkt_bucket", "rrr_band"]


def classify_outcome(b: pd.DataFrame) -> pd.Series:
    r = b.team_runs.clip(upper=6)
    lab = r.map({0: "dot", 1: "1", 2: "2", 3: "3", 4: "4", 5: "4", 6: "6"})
    lab = lab.where(~(b.is_legal & b.is_wicket), "wicket")
    return lab.where(b.is_legal, "wide")


def wkt_bucket(wkts_before) -> np.ndarray:
    return pd.cut(np.asarray(wkts_before), WKT_BINS, labels=WKT_LABELS).astype(str)


def rrr_band(rrr) -> np.ndarray:
    return pd.cut(np.asarray(rrr, dtype=float), RRR_BINS, labels=RRR_LABELS, right=False).astype(str)


def add_buckets(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["wkt_bucket"] = wkt_bucket(b.wkts_before)
    b["rrr_band"] = np.where(b.innings == 2, rrr_band(b.rrr), "all")
    return b


def outcome_probs(b: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Hierarchically smoothed p(o): innings -> x phase -> x wkt bucket -> x rrr band (inn 2).

    Needs columns innings, phase, wkt_bucket, rrr_band, outcome.
    """
    oc = list(OUTCOMES)
    cnt = pd.crosstab([b.innings, b.phase, b.wkt_bucket, b.rrr_band], b.outcome).reindex(columns=oc, fill_value=0)

    def smooth(child, parent_p, parent_keys):
        n = child.sum(axis=1)
        drop = [l for l in child.index.names if l not in parent_keys]
        par = parent_p.reindex(child.index.droplevel(drop)).values
        return (child.values + alpha * par) / (n.values[:, None] + alpha)

    lv0 = cnt.groupby(level="innings").sum()
    p0 = lv0.div(lv0.sum(axis=1), axis=0)
    lv1 = cnt.groupby(level=["innings", "phase"]).sum()
    p1 = pd.DataFrame(smooth(lv1, p0, ["innings"]), index=lv1.index, columns=oc)
    lv2 = cnt.groupby(level=["innings", "phase", "wkt_bucket"]).sum()
    p2 = pd.DataFrame(smooth(lv2, p1, ["innings", "phase"]), index=lv2.index, columns=oc)
    p3 = pd.DataFrame(smooth(cnt, p2, ["innings", "phase", "wkt_bucket"]), index=cnt.index, columns=oc)
    out = p3.reset_index()
    out["n_balls"] = cnt.sum(axis=1).values
    return out


def endgame_dp(probs: pd.DataFrame, max_b: int = 3, max_r: int = 120) -> np.ndarray:
    """Exact chase win probability by backward induction for the last `max_b` balls.

    V[b, w, r] = P(chasing side wins | b legal balls left, w wickets in hand, r runs needed),
    using death-phase innings-2 outcome probabilities by wickets bucket x required-rate band.
    Terminal: r <= 0 -> 1; out of balls or wickets with r == 1 -> 0.5 (tie); otherwise 0.
    """
    dp = probs[(probs.innings == 2) & (probs.phase == "death")].set_index(["wkt_bucket", "rrr_band"])
    oc = list(OUTCOMES)
    V = np.zeros((max_b + 1, 11, max_r + 1))
    V[:, :, 0] = 1.0
    V[0, :, 1] = 0.5
    V[:, 0, 1] = 0.5
    V[0, :, 0] = 1.0
    V[:, 0, 0] = 1.0
    for bb in range(1, max_b + 1):
        for w in range(1, 11):
            wb = wkt_bucket([10 - w])[0]
            for r in range(1, max_r + 1):       # ascending r: a wide (r-1, same bb) is already solved
                band = rrr_band([r / bb * 6])[0]
                p = dp.loc[(wb, band), oc].values.astype(float)
                v = 0.0
                for k, (runs, wk, legal) in enumerate(OUTCOMES.values()):
                    v += p[k] * V[bb - legal, w - wk, max(r - runs, 0)]
                V[bb, w, r] = v
        # band switches in p(o) can create tiny reversals: enforce monotone by construction
        V[bb] = np.minimum.accumulate(V[bb], axis=1)        # non-increasing in runs needed
        V[bb] = np.maximum.accumulate(V[bb], axis=0)        # non-decreasing in wickets in hand
    return V

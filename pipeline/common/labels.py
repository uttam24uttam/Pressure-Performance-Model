"""Delivery and stroke label families (process track). Fine labels are kept for a Stage 12 variant only,
because label coding drifts across seasons (e.g. IPL "Seamed In" 12.9% -> 6.3% -> 10.1%).
"""
import pandas as pd

DELIVERY_FAMILY = {
    # key execution signals kept as individual categories
    "Yorker": "Yorker", "Wide Yorker": "Wide Yorker", "Full Toss": "Full Toss", "Bouncer": "Bouncer",
    "Beamer": "Full Toss",                               # a beamer is a (high) full toss; 129 balls
    # pace off
    "Off Cutter": "pace_off", "Leg Cut": "pace_off", "Knuckle": "pace_off", "Slow Bouncer": "pace_off",
    "Slow Ball": "pace_off",                             # not in the user list; obviously pace-off
    # seam / swing / line
    "Seamed In": "seam_swing", "Seamed Out": "seam_swing", "Inswing": "seam_swing", "Out Swing": "seam_swing",
    "Reverse Swing In": "seam_swing", "Reverse Swing Out": "seam_swing", "Straight": "seam_swing",
    "Angled In": "seam_swing", "Angled Away": "seam_swing",
    "Reverse Swing": "seam_swing", "Quicker Ball": "seam_swing",   # not in the user list (7 / 336 balls)
    # spin types as they are
    "Orthodox": "Orthodox", "Leg Break": "Leg Break", "Off Break": "Off Break", "Googly": "Googly",
    "Carrom": "Carrom", "Arm ball": "Arm ball", "Chinaman": "Chinaman", "Back of Hand": "Back of Hand",
    "Flipper": "Flipper", "Slider": "Slider",            # spin types not in the user list, kept as they are
    "Top Spin": "spin_other", "Doosra": "spin_other", "Flick": "spin_other",   # < 300 balls together
}

STROKE_FAMILY = {
    "Frontfoot defensive": "defence", "Backfoot defensive": "defence", "Defense": "defence",
    "Intentional Padding": "defence",
    "Worked": "rotation", "Push": "rotation", "Steer": "rotation", "Leg Glance": "rotation", "Glance": "rotation",
    "Late Cut": "rotation",
    "Flick": "rotation",       # user rule: moved from aerial because its dismissal rate is 6.1% (< 7%)
    "Cover Drive": "drives", "Off Drive": "drives", "On drive": "drives", "Straight Drive": "drives",
    "Square drive": "drives",
    "Square Cut": "cut_pull", "Pull shot": "cut_pull", "Hook Shot": "cut_pull", "UpperCut": "cut_pull",
    "Sweep Shot": "sweeps", "Paddle Sweep": "sweeps",
    "Slog": "aerial", "Slog Sweep": "aerial", "PickUp": "aerial", "Inside out": "aerial",
    "Paddle Scoop": "unorthodox", "Reverse Paddle Scoop": "unorthodox", "Reverse Sweep": "unorthodox",
    "Switch": "unorthodox",
}


def _clean(s: pd.Series) -> pd.Series:
    return s.where(s.notna() & ~s.astype(str).isin(["None", "", "nan"]), "none").astype(str)


def delivery_family(s: pd.Series) -> pd.Series:
    c = _clean(s)
    return c.map(DELIVERY_FAMILY).fillna(c.where(c == "none", "other"))


def stroke_family(s: pd.Series) -> pd.Series:
    c = _clean(s)
    return c.map(STROKE_FAMILY).fillna(c.where(c == "none", "other"))


def fine(s: pd.Series) -> pd.Series:
    return _clean(s)


def add_labels(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["deliv_fam"] = delivery_family(b.delivery_style)
    b["stroke_fam"] = stroke_family(b.stroke)
    b["deliv_fine"] = fine(b.delivery_style)
    b["stroke_fine"] = fine(b.stroke)
    b["is_yorker"] = b.deliv_fam.isin(["Yorker", "Wide Yorker"])
    b["is_full_toss"] = b.deliv_fam.eq("Full Toss")
    b["is_wide_nb"] = b.extras_type.isin(["wide", "Overstep"])
    b["is_pace_off"] = b.deliv_fam.eq("pace_off")
    b["is_bouncer"] = b.deliv_fam.eq("Bouncer")
    b["is_aerial"] = b.stroke_fam.eq("aerial")
    b["is_defence"] = b.stroke_fam.eq("defence")
    return b

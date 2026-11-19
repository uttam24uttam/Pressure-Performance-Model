"""Shared helpers: raw loading, constants, small utilities."""
from pathlib import Path
import pandas as pd

ILLEGAL_EXTRAS = {"wide", "Overstep", "other"}          # balls not counting toward the over
BOWLER_EXTRAS = {"wide", "Overstep", "other"}           # extras charged to the bowler
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
RAW_FILES = {
    "balls": "ball_by_ball.csv",
    "matches": "matches.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "xi": "playing_eleven.csv",
    "squads": "competition_team_player_mapping.csv",
}


def load_raw(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["paths"]["raw"]) / RAW_FILES[name], low_memory=False)


def comp_group(comp_name: str) -> str:
    if comp_name.startswith("IPL"):
        return "IPL"
    if "Emerging" in comp_name or "Rising Stars" in comp_name:
        return "EMERGING"
    return "T20I"


def interim(cfg: dict, name: str) -> Path:
    return Path(cfg["paths"]["interim"]) / f"{name}.parquet"


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str, results: list | None = None) -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if results is not None:
        results.append((bool(cond), msg))
    return bool(cond)

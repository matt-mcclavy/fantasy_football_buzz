#!/usr/bin/env python3
"""
nfl_explore.py — Explore nflreadpy outcomes and join them to your Yahoo preseason ADP.

What it does:
  1. Loads weekly player stats for 2023-2026 and rolls them up to season PPR totals.
  2. Ranks players within position (WR1, RB12, ...) — the "actual finish".
  3. Reads your Yahoo CSVs from yahoo_adp_output/, ranks preseason ADP within position
     (the "expected finish"), and joins the two by normalized player name.
  4. Prints match rates, unmatched names, and the biggest ADP beats/misses per season —
     i.e., a preview of the project's outcome label.

Setup:
  pip install nflreadpy pandas pyarrow
  Run from the same folder as yahoo_adp_explore.py (it looks for ./yahoo_adp_output/adp_*.csv).

Outputs (in ./nfl_output/):
  season_ppr_<season>.csv           season totals + positional finish
  adp_vs_finish_<season>.csv        joined Yahoo ADP vs actual finish (the label table)
"""

import glob
import re
import sys
import unicodedata
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

SEASONS = [2023, 2024, 2025, 2026]
POSITIONS = ["QB", "RB", "WR", "TE"]
YAHOO_DIR = Path("yahoo_adp_output")
OUT_DIR = Path("nfl_output")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def norm_name(s: str) -> str:
    """'James Cook III' / 'D.J. Moore' / 'Kenneth Walker III' → 'james cook' / 'dj moore' / 'kenneth walker'."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def season_totals(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats([season]).to_pandas()
    df = df[(df["season_type"] == "REG") & (df["position"].isin(POSITIONS))]
    g = (df.groupby(["player_id", "player_display_name", "position"], as_index=False)
           .agg(team=("team", "last"), games=("week", "nunique"),
                last_week=("week", "max"), ppr=("fantasy_points_ppr", "sum")))
    g["ppr_per_game"] = g["ppr"] / g["games"]
    g["finish_pos_rank"] = g.groupby("position")["ppr"].rank(ascending=False, method="min").astype(int)
    g["season"] = season
    g["name_key"] = g["player_display_name"].map(norm_name)
    return g.sort_values("ppr", ascending=False)


def load_yahoo(season: int) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(YAHOO_DIR / f"adp_{season}_*.csv")))
    if not files:
        return None
    y = pd.read_csv(files[-1])                     # most recent snapshot
    y = y[y["position"].isin(POSITIONS)].dropna(subset=["da_preseason_average_pick"]).copy()
    y["adp_pos_rank"] = y.groupby("position")["da_preseason_average_pick"].rank(method="min").astype(int)
    y["name_key"] = y["name"].map(norm_name)
    return y


def explore_season(season: int):
    print(f"\n{'=' * 20} {season} {'=' * 20}")
    st = season_totals(season)
    st.to_csv(OUT_DIR / f"season_ppr_{season}.csv", index=False)
    print(f"nflreadpy: {len(st)} QB/RB/WR/TE with stats, through week {st['last_week'].max()}")
    print("Top 3 PPR finishers per position:")
    top = st[st["finish_pos_rank"] <= 3].sort_values(["position", "finish_pos_rank"])
    print(top[["position", "finish_pos_rank", "player_display_name", "team", "games", "ppr"]]
          .to_string(index=False))

    y = load_yahoo(season)
    if y is None:
        print(f"(no Yahoo CSV for {season} in {YAHOO_DIR}/ — skipping the join)")
        return
    j = y.merge(st, on=["name_key", "position"], how="left", suffixes=("_yahoo", ""))
    matched = j["ppr"].notna()
    print(f"\nYahoo players with preseason ADP: {len(y)} | matched to nflreadpy: "
          f"{matched.sum()} ({matched.mean():.0%})")
    if (~matched).any():
        print("Unmatched (name variants, injured all year, or retired) — first 15:")
        print("  " + ", ".join(j.loc[~matched, "name"].head(15)))

    j = j[matched].copy()
    j["finish_vs_adp"] = j["adp_pos_rank"] - j["finish_pos_rank"]     # + = beat ADP
    cols = ["name", "position", "da_preseason_average_pick", "adp_pos_rank",
            "finish_pos_rank", "finish_vs_adp", "games", "ppr"]
    j[cols].to_csv(OUT_DIR / f"adp_vs_finish_{season}.csv", index=False)

    top_adp = j[j["adp_pos_rank"] <= 36]            # fantasy-relevant tier per position
    print("\nBiggest ADP BEATS (drafted as top-36 at position):")
    print(top_adp.nlargest(8, "finish_vs_adp")[cols].to_string(index=False))
    print("\nBiggest ADP MISSES:")
    print(top_adp.nsmallest(8, "finish_vs_adp")[cols].to_string(index=False))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for s in SEASONS:
        try:
            explore_season(s)
        except Exception as e:                       # e.g., 2026 data not published yet
            print(f"\n{s}: failed — {e}")
    print(f"\nSaved CSVs in {OUT_DIR.resolve()}")
    print("Note: 2026 is partial-season, so its finish ranks are 'so far' — compare per-game too.")


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
yahoo_adp_explore.py — Explore Yahoo Fantasy Football `draft_analysis` (ADP) data.

Read-only. Answers three questions for the text-analytics project:
  1. What fields does draft_analysis actually return? (average_pick, preseason_*, ...)
  2. Do PAST seasons (2023/2024/2025) return real ADP, or blanks?
  3. Does game-level ADP differ from league-level ADP for your own league?

Outputs (in ./yahoo_adp_output/):
  - raw_sample_<game_key>.json        first raw API page, to inspect Yahoo's nested JSON
  - adp_<season>_<YYYY-MM-DD>.csv     flattened player ADP table per season (dated snapshot)
  - summary printed to the console

Setup:
  pip install yahoo_oauth pandas
  Point OAUTH_FILE at the same oauth2.json your Elite Dynasty Tracker uses
  (yahoo_oauth format: {"consumer_key": ..., "consumer_secret": ..., plus tokens}).
  If your pipeline uses a different auth approach, replace get_session() only —
  everything else just needs an object with a .get(url) method that returns a requests.Response.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- config ---
OAUTH_FILE = Path("~/yahoo-adp/oauth2.json").expanduser()
SEASONS = [2023, 2024, 2025, 2026]       # seasons to probe for historical ADP
MAX_PLAYERS = 300                         # per season; Yahoo pages 25 at a time
SORT = "OR"                               # OR = original/preseason rank; AR = actual rank
MY_LEAGUE_KEY_2025 = "461.l.44146"        # your 2025 ELITE DYNASTY league, for comparison
OUT_DIR = Path("yahoo_adp_output")
BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
PAUSE = 0.5                               # seconds between calls, be polite


# ------------------------------------------------------------------ auth ---
def get_session():
    """Return an authenticated session with a .get(url) method."""
    from yahoo_oauth import OAuth2
    oauth = OAuth2(None, None, from_file=str(OAUTH_FILE))
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    return oauth.session


def fetch(session, path):
    url = f"{BASE}/{path}{'&' if '?' in path else '?'}format=json"
    r = session.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}\n{r.text[:300]}")
    time.sleep(PAUSE)
    return r.json()


# --------------------------------------------------------------- parsing ---
def merge_dicts(items):
    """Yahoo returns objects as lists of one-key dicts; merge them into one dict."""
    out = {}
    for it in items if isinstance(items, list) else [items]:
        if isinstance(it, dict):
            out.update(it)
        elif isinstance(it, list):
            out.update(merge_dicts(it))
    return out


def find_players_block(node):
    """Recursively locate the 'players' collection anywhere in a Yahoo response."""
    if isinstance(node, dict):
        if "players" in node and isinstance(node["players"], (dict, list)):
            return node["players"]
        for v in node.values():
            hit = find_players_block(v)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for v in node:
            hit = find_players_block(v)
            if hit is not None:
                return hit
    return None


def parse_players(resp):
    """Flatten a players collection into a list of dict rows."""
    block = find_players_block(resp)
    if not block or isinstance(block, list):     # Yahoo returns [] when empty
        return []
    rows = []
    for k, v in block.items():
        if k == "count":
            continue
        parts = v["player"]                       # [ [meta dicts...], {"draft_analysis": [...]} ]
        meta = merge_dicts(parts[0])
        row = {
            "player_key": meta.get("player_key"),
            "player_id": meta.get("player_id"),
            "name": (meta.get("name") or {}).get("full"),
            "team": meta.get("editorial_team_abbr"),
            "position": meta.get("display_position"),
        }
        for extra in parts[1:]:
            if isinstance(extra, dict) and "draft_analysis" in extra:
                for fk, fv in merge_dicts(extra["draft_analysis"]).items():
                    row[f"da_{fk}"] = fv
        rows.append(row)
    return rows


def to_numeric(df):
    for c in [c for c in df.columns if c.startswith("da_")]:
        df[c] = pd.to_numeric(df[c].replace("-", None), errors="coerce")
    return df


# ------------------------------------------------------------- exploring ---
def resolve_game_keys(session, seasons):
    resp = fetch(session, f"games;game_codes=nfl;seasons={','.join(map(str, seasons))}")
    games = resp["fantasy_content"]["games"]
    keys = {}
    for k, v in games.items():
        if k == "count":
            continue
        g = merge_dicts(v["game"])
        keys[int(g["season"])] = g["game_key"]
    return keys


def pull_adp(session, resource_key, scope="game", save_raw=False):
    rows, start = [], 0
    while start < MAX_PLAYERS:
        path = f"{scope}/{resource_key}/players;sort={SORT};start={start};count=25/draft_analysis"
        resp = fetch(session, path)
        if save_raw and start == 0:
            (OUT_DIR / f"raw_sample_{resource_key}.json").write_text(json.dumps(resp, indent=2))
        page = parse_players(resp)
        if not page:
            break
        rows.extend(page)
        start += 25
    return to_numeric(pd.DataFrame(rows))


def describe(df, label):
    print(f"\n=== {label}: {len(df)} players ===")
    if df.empty:
        print("  (no players returned)")
        return
    da_cols = [c for c in df.columns if c.startswith("da_")]
    print("  draft_analysis fields:", ", ".join(da_cols) or "NONE")
    for c in da_cols:
        print(f"    {c:<32} non-null: {df[c].notna().sum():>4}   "
              f"min {df[c].min():>8.2f}   max {df[c].max():>8.2f}")
    if "da_average_pick" in df:
        top = df.dropna(subset=["da_average_pick"]).sort_values("da_average_pick").head(12)
        show = ["name", "position", "team"] + [c for c in
                ["da_average_pick", "da_preseason_average_pick", "da_percent_drafted"] if c in df]
        print(top[show].to_string(index=False))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    session = get_session()
    today = date.today().isoformat()

    print("Resolving game keys...")
    keys = resolve_game_keys(session, SEASONS)
    print("  ", keys)

    season_dfs = {}
    for season, gk in sorted(keys.items()):
        try:
            df = pull_adp(session, gk, save_raw=True)
        except RuntimeError as e:
            print(f"\n{season} ({gk}) failed: {e}")
            continue
        df.insert(0, "season", season)
        df.insert(1, "pulled_on", today)
        df.to_csv(OUT_DIR / f"adp_{season}_{today}.csv", index=False)
        season_dfs[season] = df
        describe(df, f"Game-level ADP {season} (game {gk})")

    # Q3: is league-level ADP just your league's draft, or the global Yahoo number?
    if MY_LEAGUE_KEY_2025 and 2025 in season_dfs:
        try:
            lg = pull_adp(session, MY_LEAGUE_KEY_2025, scope="league")
            describe(lg, f"League-level ADP {MY_LEAGUE_KEY_2025}")
            if "da_average_pick" in lg:
                m = season_dfs[2025].merge(lg, on="player_key", suffixes=("_game", "_league"))
                same = (m["da_average_pick_game"] == m["da_average_pick_league"]).mean()
                print(f"\n  League vs game average_pick identical for {same:.0%} "
                      f"of {len(m)} overlapping players")
        except RuntimeError as e:
            print(f"\nLeague-level pull failed: {e}")

    print(f"\nDone. Files in {OUT_DIR.resolve()}")
    print("Tip: re-run weekly during an offseason — the dated CSVs become your ADP time series.")


if __name__ == "__main__":
    sys.exit(main())
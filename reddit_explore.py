#!/usr/bin/env python3
"""
reddit_explore.py — Pull a sample of Reddit posts/comments from Arctic Shift and look around.

Arctic Shift is a free, no-login archive of Reddit (https://arctic-shift.photon-reddit.com).
It returns at most 100 items per call, so this script pages through a date window,
saves everything to disk, and prints a quick summary.

Setup:
  pip install requests pandas pyarrow

Examples:
  # Two weeks of r/DynastyFF posts from 2025 training camp (fast, ~1-2 min)
  python reddit_explore.py --kind posts --start 2025-07-20 --end 2025-08-03

  # One day of comments (comments are MUCH higher volume — start small)
  python reddit_explore.py --kind comments --start 2025-08-01 --end 2025-08-02

  # Different subreddit, and count mentions of specific players
  python reddit_explore.py --sub fantasyfootball --kind posts --start 2025-08-01 --end 2025-08-08 \
      --players "Breece Hall" "Tory Horton" "Travis Hunter"

Outputs (in ./reddit_output/):
  <sub>_<kind>_<start>_<end>.parquet   (and .csv for easy eyeballing in Excel)
"""

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://arctic-shift.photon-reddit.com/api"
OUT_DIR = Path("reddit_output")
HEADERS = {"User-Agent": "text-analytics-class-project/0.1"}

FIELDS = {
    "posts": "id,created_utc,author,title,selftext,score,num_comments,link_flair_text,url",
    "comments": "id,created_utc,author,body,score,link_id,parent_id",
}
DEFAULT_PLAYERS = ["Breece Hall", "Travis Hunter", "Ashton Jeanty", "Tetairoa McMillan",
                   "Tory Horton", "RJ Harvey", "Matthew Golden", "Cam Skattebo"]


def to_epoch(d: str) -> int:
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def get_page(kind, sub, after, before, retries=6):
    """One API call, with backoff for Arctic Shift's 'Timeout. Maybe slow down a bit' (HTTP 422/429)."""
    params = {"subreddit": sub, "after": after, "before": before,
              "limit": 100, "sort": "asc", "fields": FIELDS[kind]}
    wait = 2
    for attempt in range(retries):
        r = requests.get(f"{API}/{kind}/search", params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json().get("data") or []
        if r.status_code in (422, 429, 500, 502, 503, 504):
            print(f"    HTTP {r.status_code} ({r.text[:80].strip()}) — waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue
        raise RuntimeError(f"Arctic Shift returned HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Gave up after repeated rate-limit errors; try a smaller window or later.")


def pull(kind, sub, start, end, max_items, pause=1.0):
    """Page forward through time: each call starts just after the last item we received."""
    after, before = to_epoch(start), to_epoch(end)
    seen, rows = set(), []
    while len(rows) < max_items:
        page = get_page(kind, sub, after, before)
        new = [p for p in page if p["id"] not in seen]
        if not new:
            if page and len(page) == 100:      # 100 items share one timestamp — nudge forward
                after += 1
                continue
            break
        for p in new:
            seen.add(p["id"])
        rows.extend(new)
        after = max(p["created_utc"] for p in new)
        last = datetime.fromtimestamp(after, timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  {len(rows):>6} {kind} so far (through {last} UTC)")
        if len(page) < 100:
            break
        time.sleep(pause)
    df = pd.DataFrame(rows[:max_items])
    if not df.empty:
        df["created"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        # Arctic Shift doesn't serve 'permalink'; build a clickable link to verify posts on Reddit
        if kind == "posts":
            df["reddit_link"] = f"https://www.reddit.com/r/{sub}/comments/" + df["id"]
        else:
            df["reddit_link"] = (f"https://www.reddit.com/r/{sub}/comments/"
                                 + df["link_id"].str.replace("t3_", "", regex=False)
                                 + "/_/" + df["id"])
    return df


def mention_counts(df, kind, players):
    text = (df["title"].fillna("") + " " + df["selftext"].fillna("")) if kind == "posts" \
        else df["body"].fillna("")
    out = []
    for name in players:
        last = name.split()[-1]
        full = text.str.contains(rf"\b{re.escape(name)}\b", case=False, regex=True).sum()
        lastonly = text.str.contains(rf"\b{re.escape(last)}\b", case=False, regex=True).sum()
        out.append((name, full, lastonly))
    return pd.DataFrame(out, columns=["player", "full_name_hits", "last_name_hits"])


def summarize(df, kind, players):
    print(f"\n=== {len(df)} {kind} | {df['created'].min():%Y-%m-%d} → {df['created'].max():%Y-%m-%d} ===")
    print("\nVolume per day:")
    print(df.groupby(df["created"].dt.date).size().to_string())
    print(f"\nScore: median {df['score'].median():.0f}, max {df['score'].max()}")
    print(f"Unique authors: {df['author'].nunique()}")

    if kind == "posts":
        print("\nTop 10 posts by score:")
        for _, r in df.nlargest(10, "score").iterrows():
            print(f"  [{r['score']:>4}↑ {r['num_comments']:>3}c] {r['title'][:110]}")
        if "link_flair_text" in df:
            print("\nFlair breakdown:")
            print(df["link_flair_text"].fillna("(none)").value_counts().head(10).to_string())
        lengths = df["selftext"].fillna("").str.len()
        print(f"\nPosts with body text: {(lengths > 0).mean():.0%}  (median length {lengths[lengths>0].median():.0f} chars)")
    else:
        lengths = df["body"].fillna("").str.len()
        print(f"\nComment length: median {lengths.median():.0f} chars, 90th pct {lengths.quantile(.9):.0f}")
        removed = df["body"].isin(["[removed]", "[deleted]"]).mean()
        print(f"Removed/deleted: {removed:.1%}")
        print("\nTop 5 comments by score:")
        for _, r in df.nlargest(5, "score").iterrows():
            print(f"  [{r['score']:>4}↑] {r['body'][:160].replace(chr(10), ' ')}")

    print("\nPlayer mentions (full name vs last name only — the gap shows why entity matching matters):")
    print(mention_counts(df, kind, players).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="DynastyFF")
    ap.add_argument("--kind", choices=["posts", "comments"], default="posts")
    ap.add_argument("--start", default="2025-07-20", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default="2025-08-03", help="YYYY-MM-DD (UTC), exclusive")
    ap.add_argument("--max", type=int, default=20000, help="safety cap on items")
    ap.add_argument("--players", nargs="*", default=DEFAULT_PLAYERS)
    a = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Pulling r/{a.sub} {a.kind} {a.start} → {a.end} from Arctic Shift...")
    df = pull(a.kind, a.sub, a.start, a.end, a.max)
    if df.empty:
        print("No data returned. Check the subreddit name and dates.")
        return 1

    stem = OUT_DIR / f"{a.sub}_{a.kind}_{a.start}_{a.end}"
    df.to_parquet(f"{stem}.parquet", index=False)
    df.to_csv(f"{stem}.csv", index=False)
    summarize(df, a.kind, a.players)
    print(f"\nSaved {stem}.parquet and .csv")


if __name__ == "__main__":
    sys.exit(main())
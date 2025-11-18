"""Aggregate season summaries (points, rebounds, assists) for player_id 2544.

This script parses play_by_play descriptions to extract points ("N PTS"), REBOUND and AST
indicators and groups totals by the game's season (from `game.season_id`).

Run:
  python3 lebron_season_summary.py

"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

PLAYER_ID = '2544'
DB = 'basketball.db'

points_re = re.compile(r"(\d+)\s*PTS", re.IGNORECASE)

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Build a mapping from season_id -> readable season label (e.g. 2005-06)
    season_labels: dict[str, str] = {}
    try:
        cur.execute(
            "SELECT season_id, MIN(substr(game_date,1,4)) as miny, MAX(substr(game_date,1,4)) as maxy FROM game GROUP BY season_id"
        )
        for season_id, miny, maxy in cur.fetchall():
            if season_id is None:
                continue
            if miny is None or maxy is None:
                season_labels[season_id] = season_id
                continue
            try:
                miny_i = int(miny)
                maxy_i = int(maxy)
                if miny_i == maxy_i:
                    season_labels[season_id] = str(miny_i)
                else:
                    season_labels[season_id] = f"{miny_i}-{str(maxy_i % 100).zfill(2)}"
            except Exception:
                season_labels[season_id] = f"{miny}-{maxy}"
    except Exception:
        # If anything fails, fall back to using raw season_id
        season_labels = {}

    q = """
    SELECT p.game_id, g.season_id, p.player1_id, p.player2_id, p.player3_id,
           p.homedescription, p.visitordescription, p.neutraldescription
    FROM play_by_play p
    LEFT JOIN game g ON p.game_id = g.game_id
    WHERE p.player1_id = ? OR p.player2_id = ? OR p.player3_id = ?
    """

    cur.execute(q, (PLAYER_ID, PLAYER_ID, PLAYER_ID))

    season_totals = defaultdict(lambda: {'points': 0, 'rebounds': 0, 'assists': 0})
    row_count = 0

    for row in cur:
        row_count += 1
        game_id, season_id, p1, p2, p3, home_desc, visit_desc, neutral_desc = row
        season = season_id or 'unknown'
        text = ' '.join([str(x) for x in (home_desc, visit_desc, neutral_desc) if x])
        text_up = text.upper()

        # Points: if player acted as player1 (the actor) and description contains "N PTS"
        if p1 == PLAYER_ID:
            m = points_re.search(text)
            if m:
                pts = int(m.group(1))
                season_totals[season]['points'] += pts
        # Rebounds: actor string often contains 'REBOUND'
        if p1 == PLAYER_ID and 'REBOUND' in text_up:
            season_totals[season]['rebounds'] += 1
        # Assists: when player is an assister, they often appear in player2 or player3 fields and description
        # includes 'AST' text. Check player2/player3
        if (p2 == PLAYER_ID or p3 == PLAYER_ID) and 'AST' in text_up:
            season_totals[season]['assists'] += 1
        # Sometimes an assist is recorded as part of scorer description (e.g. "(LeBron 1 AST)") and player1
        # may not be the assister. To count those where LeBron is the assister but appears in text, check name
        if 'LEBRON' in text_up and 'AST' in text_up and not ((p2 == PLAYER_ID) or (p3 == PLAYER_ID)):
            # add assist if assist mention includes his name but he wasn't in player2/3
            season_totals[season]['assists'] += 0  # conservative: don't double count

    conn.close()

    print(f"Processed {row_count} play_by_play rows mentioning player {PLAYER_ID}")
    print("Season summary (season_label / season_id: points, rebounds, assists):")
    for season in sorted(season_totals.keys()):
        t = season_totals[season]
        label = season_labels.get(season, season)
        print(f"{label} / {season}: pts={t['points']}, reb={t['rebounds']}, ast={t['assists']}")


if __name__ == '__main__':
    main()

"""Main script: demonstrates how to query basketball.db and shows available parameters.

Run:
  python3 main.py

This script prints:
- Available tables and their row counts
- Sample player info (LeBron James)
- Sample game data
- Common query patterns you can use
"""

import sqlite3

DB_PATH = 'basketball.db'

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=" * 80)
    print("BASKETBALL DATABASE INFO")
    print("=" * 80)

    print(" AVAILABLE TABLES:")
    print("-" * 80)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cur.fetchall()]
    for table in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cur.fetchone()[0]
        print(f"  {table:25s} - {count:,} rows")

    print("-" * 80)
    cur.execute("SELECT id, full_name, abbreviation FROM team ")
    for row in cur.fetchall():
        print(f"  {row[2]:5s} - {row[1]:30s} (team_id: {row[0]})")



    conn.close()


if __name__ == '__main__':
    main()

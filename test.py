"""Simple inspector: list sqlite tables and print counts + sample rows.

Usage:
  python3 test.py            # uses ./basketball.db
  DB path can be changed by setting the DB_PATH variable below or via CLI arg
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import Iterable


def list_tables(conn: sqlite3.Connection) -> list[str]:
	cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
	return [row[0] for row in cur.fetchall()]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
	cur = conn.execute(f"SELECT COUNT(*) FROM \"{table}\"")
	return cur.fetchone()[0]


def sample_rows(conn: sqlite3.Connection, table: str, limit: int = 10) -> Iterable[tuple]:
	cur = conn.execute(f"SELECT * FROM \"{table}\" LIMIT {limit}")
	cols = [d[0] for d in cur.description] if cur.description else []
	rows = cur.fetchall()
	return cols, rows


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description="Inspect sqlite DB and print tables + sample rows")
	p.add_argument("--db", default="basketball.db", help="Path to sqlite DB (default: basketball.db)")
	p.add_argument("--limit", type=int, default=10, help="Number of sample rows to print per table")
	args = p.parse_args(argv)

	try:
		conn = sqlite3.connect(args.db)
	except Exception as exc:
		print(f"Failed to open DB '{args.db}': {exc}", file=sys.stderr)
		return 2

	try:
		tables = list_tables(conn)
		if not tables:
			print(f"No tables found in '{args.db}'")
			return 0

		print(f"Found {len(tables)} table(s) in '{args.db}':\n")
		for t in tables:
			try:
				cnt = count_rows(conn, t)
			except Exception as exc:
				print(f"- {t}: error counting rows: {exc}")
				continue
			print(f"- {t}: {cnt} rows")
			cols, rows = sample_rows(conn, t, limit=args.limit)
			if not rows:
				print("  (no sample rows)\n")
				continue
			# print header
			header = " | ".join(cols) if cols else "(no columns)"
			print("  ", header)
			for r in rows:
				# print each row in a compact form
				print("  ", " | ".join(str(x) for x in r))
			print()
	finally:
		conn.close()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())



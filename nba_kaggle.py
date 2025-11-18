"""Minimal script: download/import sqlite from Kaggle or import a local sqlite file.

Usage:
  # download and copy any sqlite-like file found in the dataset to basketball.db
  python3 nba_kaggle.py --file nba.sqlite --kaggle-download --db basketball.db

  # import a local sqlite file directly
  python3 nba_kaggle.py --local-file /path/to/nba.sqlite --db basketball.db
"""

from __future__ import annotations
import argparse
import os
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Optional


def is_sqlite_filename(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in (".sqlite", ".db", ".sqlite3", ".db3", ".s3db", ".dl3"))


def run_kaggle_cli_download_and_find_sqlite(dataset: str, filename: Optional[str], dest_dir: str) -> str:
    # Try specific file first
    if filename:
        try:
            cmd = [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-f",
                filename,
                "-p",
                dest_dir,
                "--unzip",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            candidate = os.path.join(dest_dir, filename)
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass

    # Otherwise download full dataset and search for sqlite-like files
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", dest_dir, "--unzip"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.lower().endswith((".sqlite", ".db", ".sqlite3", ".db3", ".s3db", ".dl3")):
                return os.path.join(root, f)
    raise RuntimeError("No sqlite-like files found in downloaded dataset")


def copy_sqlite(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Import sqlite from Kaggle or local file")
    p.add_argument("--dataset", "-d", default="wyattowalsh/basketball")
    p.add_argument("--file", "-f", help="Filename in dataset (optional)")
    p.add_argument("--local-file", help="Path to local sqlite file")
    p.add_argument("--kaggle-download", action="store_true", help="Download from Kaggle using kaggle CLI")
    p.add_argument("--db", default="basketball.db", help="Destination DB path")
    args = p.parse_args(argv)

    if args.local_file:
        if not os.path.exists(args.local_file):
            print("Local file not found:", args.local_file)
            return 1
        copy_sqlite(args.local_file, args.db)
        print("Copied local sqlite to", args.db)
        return 0

    if args.kaggle_download:
        if shutil.which("kaggle") is None:
            print("Kaggle CLI not found. Install with: pip install kaggle")
            return 2
        with tempfile.TemporaryDirectory() as td:
            try:
                found = run_kaggle_cli_download_and_find_sqlite(args.dataset, args.file, td)
            except Exception as exc:
                print("Download failed:", exc)
                return 3
            copy_sqlite(found, args.db)
            print("Copied downloaded sqlite to", args.db)
            return 0

    print("Nothing to do. Use --local-file or --kaggle-download --file <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

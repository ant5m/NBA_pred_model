#!/usr/bin/env python3
"""Cron job: Refresh team stats database."""

import os
import sys
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_team_stats_db

def main():
    """Refresh the team stats database."""
    print(f"[{date.today()}] Refreshing team stats database...")
    
    try:
        # Run the database build script
        build_team_stats_db.main()
        print("✅ Database refresh complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

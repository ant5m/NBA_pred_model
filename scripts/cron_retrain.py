#!/usr/bin/env python3
"""Cron job: Retrain model weekly."""

import os
import sys
from datetime import date
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_model import train_ensemble_model

def main():
    """Retrain the ensemble model and recalibrate."""
    print(f"[{date.today()}] Starting weekly retrain...")
    
    try:
        # Step 1: Retrain model
        print("Step 1/2: Retraining ensemble...")
        train_ensemble_model(n_models=5, epochs=50)
        print("✅ Retrain complete")
        
        # Step 2: Recalibrate
        print("Step 2/2: Recalibrating model...")
        result = subprocess.run(
            ['python3', 'calibrate_model.py', '--ensemble'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if result.returncode == 0:
            print("✅ Calibration complete")
        else:
            print(f"⚠️  Calibration warning: {result.stderr}")
            # Don't fail the job if calibration fails
        
        print("✅ Weekly retrain + calibration complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

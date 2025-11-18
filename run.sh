#!/usr/bin/env bash
# Minimal setup script for the NBA_pred_model project
# Creates a venv at .venv, activates it, and installs requirements

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}

echo "Using python: $(command -v $PYTHON) ($( $PYTHON --version 2>&1 ))"

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in $ROOT_DIR"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtualenv at .venv"
  $PYTHON -m venv .venv
fi

echo "Activating virtualenv"
# shellcheck source=/dev/null
source .venv/bin/activate

echo "Upgrading pip and setuptools"
python -m pip install --upgrade pip setuptools wheel

echo "Installing requirements from requirements.txt"
if python -m pip install -r requirements.txt; then
  echo "All packages installed successfully."
else
  echo "Failed to install some packages from requirements.txt."
  uname_s=$(uname -s)
  uname_m=$(uname -m)
  echo "Detected platform: $uname_s / $uname_m"
  if [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
    echo "On macOS Apple Silicon: TensorFlow may require 'tensorflow-macos' and 'tensorflow-metal'."
    echo "Try running the following inside the activated venv:" 
    echo "  python -m pip install tensorflow-macos tensorflow-metal"
  fi
  echo "You can inspect the pip output above to see which package failed."
  exit 1
fi

echo "Setup complete. To activate the venv later run: source .venv/bin/activate"
echo "Quick test: python test.py"

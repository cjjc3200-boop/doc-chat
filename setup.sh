#!/usr/bin/env bash
# One-command setup + first run. macOS / Linux.
set -euo pipefail

cd "$(dirname "$0")"

PY=python3
command -v $PY >/dev/null || { echo "Python 3 not found. Install it from python.org"; exit 1; }

echo "Creating virtual environment…"
$PY -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "Building index…"
python ingest.py

echo ""
echo "Setup complete. Start chatting with:"
echo "    source .venv/bin/activate && python chat.py"

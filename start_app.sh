#!/usr/bin/env bash
# FieldScreen AI — Launch App (Linux/Mac)
# ========================================
# Usage: bash start_app.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "ERROR: Virtual environment not found."
    echo "       Run 'bash setup.sh' first to create it."
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"
echo "Using Python: $(which python3)"

python3 "$SCRIPT_DIR/app.py"

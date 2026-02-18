#!/usr/bin/env bash
# FieldScreen AI — Environment Setup (Linux/Mac)
# ================================================
# Usage: bash setup.sh

set -e

echo ""
echo "============================================"
echo "  FieldScreen AI — Environment Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi
echo "Found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Done."
else
    echo "Virtual environment already exists."
fi

source venv/bin/activate
pip install --upgrade pip --quiet

# 1. PyTorch with CUDA
echo ""
echo "[1/3] Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Transformers from source (MedASR needs >= 5.0)
echo ""
echo "[2/3] Installing transformers from source..."
pip install "git+https://github.com/huggingface/transformers.git" || \
    pip install "transformers>=4.45"

# 3. Remaining dependencies
echo ""
echo "[3/3] Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  1. Download models:  python download_models.py"
echo "  2. Get llama-server: see README.md"
echo "  3. Launch app:       python app.py"
echo "============================================"

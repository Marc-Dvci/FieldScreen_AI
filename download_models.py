"""
Download all required models for FieldScreen AI.
=================================================

This script downloads the model weights that are too large to include
in the Git repository. Run this once after cloning:

    python download_models.py

Models downloaded (~8 GB total):
  - MedGemma 4B-It GGUF (Q4_K_M)    → Models/MedGemma/
  - Vision Projector (mmproj-BF16)    → Models/MedGemma/
  - HeAR (Google, PyTorch)            → Models/HeAR/
  - MedASR (Google, PyTorch)          → Models/MedASR/
  - TranslateGemma 4B GGUF           → Models/TranslateGemma/

llama-server must be obtained separately from:
  https://github.com/ggerganov/llama.cpp/releases
  Copy llama-server.exe (+ CUDA DLLs) into bin/
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

# ── Model definitions ──
# Each entry: (url, destination_path, description, size_hint)
DIRECT_DOWNLOADS = [
    (
        "https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/main/"
        "medgemma-1.5-4b-it-Q4_K_M.gguf",
        "Models/MedGemma/medgemma-1.5-4b-it-Q4_K_M.gguf",
        "MedGemma 1.5 4B-It (Q4_K_M quantization)",
        "~2.5 GB",
    ),
    (
        "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/"
        "mmproj-BF16.gguf",
        "Models/MedGemma/mmproj-BF16.gguf",
        "Vision projector (BF16)",
        "~850 MB",
    ),
]

# HuggingFace repos to clone with huggingface_hub
HF_REPOS = [
    (
        "google/hear-pytorch",
        "Models/HeAR",
        "HeAR — Health Acoustic Representations (ViT-L)",
        "~1.2 GB",
    ),
    (
        "google/medasr",
        "Models/MedASR",
        "MedASR — Medical Speech Recognition (105M)",
        "~420 MB",
    ),
]

# Optional GGUF for TranslateGemma — large, skip if not needed
OPTIONAL_DOWNLOADS = [
    (
        "https://huggingface.co/google/translategemma-4b-it-GGUF/resolve/main/"
        "translategemma-4b-it.Q4_K_M.gguf",
        "Models/TranslateGemma/translategemma-4b-it.Q4_K_M.gguf",
        "TranslateGemma 4B (Q4_K_M) — multilingual reports",
        "~2.5 GB",
    ),
]


def download_file(url, dest_path, description, size_hint):
    """Download a file with progress indicator. Skips if already exists."""
    dest = PROJECT_DIR / dest_path
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  ✓ {description} — already downloaded")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {description} ({size_hint})...")
    print(f"    URL: {url}")
    print(f"    To:  {dest}")

    try:
        tmp = str(dest) + ".tmp"
        urllib.request.urlretrieve(url, tmp, _progress_hook)
        print()  # newline after progress
        os.rename(tmp, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Done ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _progress_hook(block_num, block_size, total_size):
    """Progress callback for urllib."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        print(f"\r    [{bar}] {pct:5.1f}% ({mb_done:.0f}/{mb_total:.0f} MB)",
              end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r    Downloaded {mb:.0f} MB...", end="", flush=True)


def download_hf_repo(repo_id, dest_path, description, size_hint):
    """Download a HuggingFace repo. Falls back to manual instructions."""
    dest = PROJECT_DIR / dest_path

    # Check if already downloaded (has model files)
    if dest.exists():
        model_files = list(dest.glob("*.safetensors")) + list(dest.glob("*.bin"))
        if model_files:
            print(f"  ✓ {description} — already downloaded")
            return True

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {description} ({size_hint})...")
    print(f"    Repo: {repo_id}")
    print(f"    To:   {dest}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        print(f"  ✓ Done")
        return True
    except ImportError:
        print(f"  ⚠ huggingface_hub not installed.")
        print(f"    Install: pip install huggingface_hub")
        print(f"    Then re-run this script, or manually download from:")
        print(f"    https://huggingface.co/{repo_id}")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"    Manually download from: https://huggingface.co/{repo_id}")
        return False


def check_llama_server():
    """Check if llama-server is available."""
    bin_dir = PROJECT_DIR / "bin"
    exe = bin_dir / "llama-server.exe"
    if exe.exists():
        print(f"  ✓ llama-server.exe found in bin/")
        return True
    else:
        print(f"  ⚠ llama-server.exe not found in bin/")
        print(f"    Download CUDA build from:")
        print(f"    https://github.com/ggerganov/llama.cpp/releases")
        print(f"    Look for: llama-*-bin-win-cuda-cu12*-x64.zip")
        print(f"    Extract llama-server.exe + DLLs into bin/")
        return False


def main():
    print("=" * 60)
    print("  FieldScreen AI — Model Downloader")
    print("=" * 60)
    print()

    all_ok = True

    # Direct downloads (GGUF files)
    print("── GGUF Models ──")
    for url, dest, desc, size in DIRECT_DOWNLOADS:
        if not download_file(url, dest, desc, size):
            all_ok = False
    print()

    # HuggingFace repos
    print("── HuggingFace Models ──")
    for repo, dest, desc, size in HF_REPOS:
        if not download_hf_repo(repo, dest, desc, size):
            all_ok = False
    print()

    # Optional
    print("── Optional Models ──")
    for url, dest, desc, size in OPTIONAL_DOWNLOADS:
        if not download_file(url, dest, desc, size):
            print(f"    (TranslateGemma is optional — app works without it)")
    print()

    # llama-server
    print("── Inference Engine ──")
    check_llama_server()
    print()

    # LoRA adapter check
    print("── LoRA Adapter ──")
    lora = PROJECT_DIR / "Models" / "MedGemma" / "image-lora.gguf"
    if lora.exists():
        print(f"  ✓ LoRA adapter present ({lora.stat().st_size/1024/1024:.0f} MB)")
    else:
        print(f"  ⚠ LoRA adapter not found (should be included in repo via Git LFS)")
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("  All models ready! Run START_APP.bat to launch.")
    else:
        print("  Some downloads failed. Check errors above and retry.")
    print("=" * 60)


if __name__ == "__main__":
    main()

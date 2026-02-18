"""
Prepare Coswara cough audio for HeAR classifier training.
==========================================================

Reads Coswara metadata CSVs, maps covid_status to binary labels,
and copies cough.wav files into tb/ and normal/ directories
expected by train_cough_classifier.py.

Label mapping:
  POSITIVE (respiratory illness): positive_mild, positive_moderate,
                                   resp_illness_not_identified
  NORMAL  (healthy):              healthy, no_resp_illness_exposed
  EXCLUDED:                        recovered_full, positive_asymp

Usage:
    python prepare_coswara_data.py
"""

import csv
import os
import shutil
from pathlib import Path

# ── Paths ──
SCRIPT_DIR = Path(__file__).parent.resolve()
ARCHIVE_DIR = SCRIPT_DIR / "cough_data" / "archive (2)"
CSVS_DIR = ARCHIVE_DIR / "csvs"
AUDIO_DIR = ARCHIVE_DIR / "coswara_data" / "kaggle_data"
OUTPUT_DIR = SCRIPT_DIR / "data"

# ── Label mapping ──
POSITIVE_STATUSES = {
    "positive_mild",
    "positive_moderate",
    "resp_illness_not_identified",
}
NORMAL_STATUSES = {
    "healthy",
    "no_resp_illness_exposed",
}
EXCLUDED_STATUSES = {
    "recovered_full",
    "positive_asymp",
}


def load_metadata():
    """Parse all per-date CSV files and return {user_id: covid_status}."""
    user_status = {}
    csv_files = sorted(CSVS_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} metadata CSV files")

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("id", "").strip()
                status = row.get("covid_status", "").strip()
                if uid and status:
                    user_status[uid] = status

    print(f"Total unique users in metadata: {len(user_status)}")
    return user_status


def prepare_data():
    """Copy audio files into tb/ and normal/ directories."""
    user_status = load_metadata()

    # Count statuses
    status_counts = {}
    for s in user_status.values():
        status_counts[s] = status_counts.get(s, 0) + 1
    print("\nStatus distribution:")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        label = ("POSITIVE" if s in POSITIVE_STATUSES
                 else "NORMAL" if s in NORMAL_STATUSES
                 else "EXCLUDED")
        print(f"  {s}: {c} [{label}]")

    # Create output directories
    tb_dir = OUTPUT_DIR / "tb"
    normal_dir = OUTPUT_DIR / "normal"
    tb_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    n_pos, n_neg, n_skip, n_no_audio = 0, 0, 0, 0

    for uid, status in user_status.items():
        audio_path = AUDIO_DIR / uid / "cough.wav"
        if not audio_path.exists():
            n_no_audio += 1
            continue

        if status in POSITIVE_STATUSES:
            dest = tb_dir / f"{uid}.wav"
            if not dest.exists():
                shutil.copy2(str(audio_path), str(dest))
            n_pos += 1
        elif status in NORMAL_STATUSES:
            dest = normal_dir / f"{uid}.wav"
            if not dest.exists():
                shutil.copy2(str(audio_path), str(dest))
            n_neg += 1
        else:
            n_skip += 1

    print(f"\n{'='*50}")
    print(f"Data prepared in: {OUTPUT_DIR}")
    print(f"  Positive (tb/):    {n_pos} files")
    print(f"  Normal (normal/):  {n_neg} files")
    print(f"  Excluded:          {n_skip}")
    print(f"  No audio found:    {n_no_audio}")
    print(f"{'='*50}")
    print(f"\nNext step: python train_cough_classifier.py --data-dir {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_data()

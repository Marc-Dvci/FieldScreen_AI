"""
Build Independent Evaluation Dataset
=====================================
Creates a balanced, anonymised evaluation set from the Tawsifurrahman
TB Chest Radiography Database — images that were NEVER used for training.

What this script does:
  1. Reads images from Tawsifurrahman Normal/ and Tuberculosis/ folders
  2. Randomly samples N images from each class (balanced)
  3. Copies them to eval_dataset/images/ with neutral sequential names
     (img_0001.png, img_0002.png, ...) so filenames leak nothing
  4. Creates eval_dataset/labels.csv with columns: filename, ground_truth
  5. No age, gender, source URL, or any other metadata is included

Usage:
  python build_eval_dataset.py              # defaults: 100 TB + 100 Normal
  python build_eval_dataset.py --per-class 50   # 50 TB + 50 Normal
  python build_eval_dataset.py --per-class 200  # 200 TB + 200 Normal
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent

# Source dataset (never used for training)
TAWSIFUR_DIR = PROJECT_DIR / "Dataset tawsifurrahman" / "TB_Chest_Radiography_Database"
NORMAL_DIR   = TAWSIFUR_DIR / "Normal"
TB_DIR       = TAWSIFUR_DIR / "Tuberculosis"

# Output
OUTPUT_DIR   = SCRIPT_DIR / "eval_dataset"
IMAGES_DIR   = OUTPUT_DIR / "images"
LABELS_CSV   = OUTPUT_DIR / "labels.csv"

SEED = 123   # different from training seed (42) to avoid any overlap logic


def collect_images(folder, label):
    """Collect all PNG images from a folder with their ground-truth label."""
    images = []
    for p in sorted(folder.glob("*.png")):
        images.append({"source_path": p, "ground_truth": label})
    return images


def build_dataset(per_class):
    """Build the evaluation dataset."""
    print(f"Source: {TAWSIFUR_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples per class: {per_class}")
    print()

    # Collect all available images
    normal_images = collect_images(NORMAL_DIR, "NORMAL")
    tb_images     = collect_images(TB_DIR, "TB")
    print(f"Available: {len(normal_images)} Normal, {len(tb_images)} TB")

    if per_class > len(tb_images):
        print(f"WARNING: Only {len(tb_images)} TB images available, "
              f"using all of them.")
        per_class = min(per_class, len(tb_images))

    if per_class > len(normal_images):
        print(f"WARNING: Only {len(normal_images)} Normal images available, "
              f"using {per_class}.")
        per_class = min(per_class, len(normal_images))

    # Random balanced sample
    rng = random.Random(SEED)
    sampled_normal = rng.sample(normal_images, per_class)
    sampled_tb     = rng.sample(tb_images, per_class)

    # Merge and shuffle (so the order doesn't reveal the class)
    all_samples = sampled_normal + sampled_tb
    rng.shuffle(all_samples)

    total = len(all_samples)
    print(f"Selected: {total} images ({per_class} Normal + {per_class} TB)")

    # Create output directory
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images with neutral filenames and write labels
    labels = []
    for idx, sample in enumerate(all_samples):
        # Neutral filename: img_0001.png, img_0002.png, ...
        neutral_name = f"img_{idx + 1:04d}.png"
        dest = IMAGES_DIR / neutral_name
        shutil.copy2(sample["source_path"], dest)

        labels.append({
            "filename":     neutral_name,
            "ground_truth": sample["ground_truth"],
        })

    # Write labels CSV (only filename + ground_truth — no other metadata)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ground_truth"])
        writer.writeheader()
        writer.writerows(labels)

    n_tb = sum(1 for l in labels if l["ground_truth"] == "TB")
    n_nm = sum(1 for l in labels if l["ground_truth"] == "NORMAL")
    print(f"\nDataset built successfully:")
    print(f"  Images: {IMAGES_DIR}  ({total} files)")
    print(f"  Labels: {LABELS_CSV}")
    print(f"  Balance: {n_tb} TB + {n_nm} Normal")
    print(f"  Filenames: neutral (img_XXXX.png) — no label leakage")
    print(f"  Metadata: none — only filename and ground_truth in CSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build balanced evaluation dataset from Tawsifurrahman")
    parser.add_argument(
        "--per-class", type=int, default=100,
        help="Number of images per class (default: 100 → 200 total)")
    args = parser.parse_args()

    build_dataset(args.per_class)

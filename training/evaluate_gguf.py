"""
MedGemma GGUF Evaluation — Majority-Vote Methodology
=====================================================
Evaluates the GGUF model (with and without LoRA) using llama-server HTTP API.
Uses an independent evaluation dataset built from the Tawsifurrahman
TB Chest Radiography Database — images NEVER used for training.

Methodology:
  - Prompt: "Is this chest X-ray normal or abnormal? Answer with:
    NORMAL or ABNORMAL, then confidence percentage."
  - Sampling: Instruct preset (min_p=0.2, temperature=1.0)
  - Each image is evaluated 5 times
  - Final answer = majority vote across the 5 runs
  - ABNORMAL → TB, NORMAL → NORMAL for ground-truth comparison
  - Images have neutral filenames (img_XXXX.png) — no label leakage
  - No metadata (age, gender, source) is sent to the model

Two evaluation passes:
  1. Base model alone   → accuracy
  2. Base model + LoRA  → accuracy
"""

import csv
import json
import base64
import time
import random
import re
import socket
import logging
import subprocess
import sys
import threading
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests

# ============================================================
# CONFIGURATION — edit these paths for your system
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent

# Model paths (relative to project — edit if your models are elsewhere)
GGUF_MODEL   = str(PROJECT_DIR / "Models" / "MedGemma" / "medgemma-1.5-4b-it-Q4_K_M.gguf")
MMPROJ       = str(PROJECT_DIR / "Models" / "MedGemma" / "mmproj-BF16.gguf")
LORA_DIR     = str(PROJECT_DIR / "Models" / "MedGemma")

LLAMA_SERVER = str(PROJECT_DIR / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server"))


# Primary: Independent evaluation dataset (Tawsifurrahman, never used for training)
EVAL_DIR    = SCRIPT_DIR / "eval_dataset"
EVAL_CSV    = EVAL_DIR / "labels.csv"
EVAL_IMAGES = EVAL_DIR / "images"

# Fallback: Montgomery 80/20 holdout (only if eval_dataset not built yet)
MONTGOMERY_DIR    = SCRIPT_DIR.parent / "Dataset Montgomery"
MONTGOMERY_CSV    = MONTGOMERY_DIR / "montgomery_metadata.csv"
MONTGOMERY_IMAGES = MONTGOMERY_DIR / "images" / "images"

REPORT_FILE = SCRIPT_DIR / "evaluation_report_gguf.txt"

GPU_LAYERS = -1     # offload everything to GPU
CTX_SIZE   = 4096
MAX_TOKENS = 100    # short answer expected (NORMAL/ABNORMAL + %)

# ── Evaluation parameters ──
PROMPT = (
    "Is this chest X-ray normal or abnormal? "
    "Answer with: NORMAL or ABNORMAL, then confidence percentage."
)
RUNS_PER_IMAGE = 5          # number of runs for majority vote
TEMPERATURE    = 1.0        # default — needed for variability across runs
MIN_P          = 0.2        # Instruct preset from text-generation-webui

# ============================================================
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("eval-gguf")


# ---- 1. DATA LOADING ------------------------------------------------

def load_test_set():
    """Load the evaluation dataset.

    Primary: Independent Tawsifurrahman dataset (eval_dataset/labels.csv).
    Fallback: Montgomery 80/20 holdout (if eval_dataset not built yet).

    Returns list of dicts with keys: filename, image_path, ground_truth.
    No metadata (age, gender, source) is included — only image + label.
    """
    # --- Try independent dataset first ---
    if EVAL_CSV.exists() and EVAL_IMAGES.exists():
        return _load_eval_dataset()

    # --- Fallback to Montgomery holdout ---
    logger.warning("eval_dataset/ not found — falling back to Montgomery holdout")
    logger.warning("Run BUILD_EVAL_DATASET.bat first for the independent test set")
    return _load_montgomery_holdout()


def _load_eval_dataset():
    """Load independent evaluation set from eval_dataset/labels.csv."""
    samples = []
    with open(EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            img_path = EVAL_IMAGES / filename
            if not img_path.exists():
                logger.warning(f"Missing image: {img_path}")
                continue
            samples.append({
                "filename":     filename,
                "image_path":   str(img_path),
                "ground_truth": row["ground_truth"].strip(),
            })

    tb = sum(1 for s in samples if s["ground_truth"] == "TB")
    nm = sum(1 for s in samples if s["ground_truth"] == "NORMAL")
    logger.info(f"Eval dataset (Tawsifurrahman): {len(samples)} samples  "
                f"({tb} TB, {nm} Normal)")
    return samples


def _load_montgomery_holdout():
    """Fallback: load Montgomery 20% holdout set (same split as training)."""
    all_samples = []
    with open(MONTGOMERY_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["study_id"].strip()
            img_path = MONTGOMERY_IMAGES / filename
            if not img_path.exists():
                continue

            findings = row.get("findings", "").strip()
            is_tb = findings.lower() != "normal"

            all_samples.append({
                "filename":     filename,
                "image_path":   str(img_path),
                "ground_truth": "TB" if is_tb else "NORMAL",
            })

    # Reproduce the training split (same seed & logic as train_image_lora.py)
    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    test_set = all_samples[split:]

    tb = sum(1 for s in test_set if s["ground_truth"] == "TB")
    nm = sum(1 for s in test_set if s["ground_truth"] == "NORMAL")
    logger.info(f"Montgomery holdout: {len(test_set)} samples  "
                f"({tb} TB, {nm} Normal)")
    return test_set


# ---- 2. LLAMA-SERVER MANAGEMENT -----------------------------------

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _drain_stderr(proc):
    """Read stderr to prevent buffer overflow (runs in background thread)."""
    try:
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
    except Exception:
        pass


def start_server(lora_path=None):
    """Start llama-server and wait until healthy. Returns (process, port)."""
    port = _find_free_port()
    cmd = [
        LLAMA_SERVER,
        "--model",      GGUF_MODEL,
        "--mmproj",     MMPROJ,
        "--ctx-size",   str(CTX_SIZE),
        "--gpu-layers", str(GPU_LAYERS),
        "--port",       str(port),
        "--flash-attn", "on",
        "--no-webui",
    ]

    if lora_path:
        cmd += ["--lora", str(lora_path)]
        logger.info(f"LoRA: {Path(lora_path).name}")

    logger.info(f"Starting llama-server on port {port} ...")
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, bufsize=0)
    threading.Thread(target=_drain_stderr, args=(proc,), daemon=True).start()

    health = f"http://127.0.0.1:{port}/health"
    session = requests.Session()
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.poll()}")
        try:
            r = session.get(health, timeout=2)
            if r.status_code == 200:
                logger.info("Server ready [OK]")
                return proc, port
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.terminate()
    raise TimeoutError("Server did not become healthy within 120 s")


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    logger.info("Server stopped.")


# ---- 3. INFERENCE -------------------------------------------------

def image_to_data_url(img_path):
    img = Image.open(img_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def query_server(port, data_url, session):
    """Send an image + prompt to the server, return the response text.

    Uses Instruct preset parameters (min_p=0.2, temperature=1.0).
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": PROMPT},
                ],
            }
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "min_p":       MIN_P,
        "stream":      False,
    }

    r = session.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---- 4. CLASSIFICATION (simple for the direct prompt) ------------

def classify_response(response_text):
    """Parse the model's response to extract NORMAL or ABNORMAL.

    The prompt asks for "NORMAL or ABNORMAL, then confidence %".
    We look for the first occurrence of either keyword.
    Returns ("NORMAL" | "ABNORMAL" | "UNCLEAR", confidence_or_None).
    """
    text = response_text.strip().upper()

    # Try to find confidence percentage
    conf_match = re.search(r'(\d{1,3})(?:\.\d+)?\s*%', response_text)
    confidence = int(conf_match.group(1)) if conf_match else None

    # Find position of NORMAL and ABNORMAL
    pos_abnormal = text.find("ABNORMAL")
    pos_normal   = text.find("NORMAL")

    # "ABNORMAL" contains "NORMAL", so if ABNORMAL is found,
    # check that any NORMAL match isn't just part of ABNORMAL
    if pos_abnormal >= 0:
        # Verify that the NORMAL we found (if any) is part of ABNORMAL
        if pos_normal >= 0 and pos_normal == pos_abnormal + 2:
            # "NORMAL" inside "ABNORMAL" — the answer is ABNORMAL
            return "ABNORMAL", confidence
        elif pos_normal >= 0 and pos_normal < pos_abnormal:
            # NORMAL appears before ABNORMAL — take the first
            return "NORMAL", confidence
        else:
            return "ABNORMAL", confidence

    if pos_normal >= 0:
        return "NORMAL", confidence

    return "UNCLEAR", confidence


def majority_vote(run_labels):
    """Take the majority label from a list of classification results.

    Returns (final_label, vote_count, total_runs).
    Ignores UNCLEAR responses for the vote.  If all are UNCLEAR,
    returns UNCLEAR.
    """
    valid = [lbl for lbl in run_labels if lbl != "UNCLEAR"]
    if not valid:
        return "UNCLEAR", 0, len(run_labels)
    counts = Counter(valid)
    winner, n = counts.most_common(1)[0]
    return winner, n, len(run_labels)


def map_to_ground_truth(label):
    """Map model output labels to ground-truth labels.

    ABNORMAL → TB, NORMAL → NORMAL.
    """
    if label == "ABNORMAL":
        return "TB"
    if label == "NORMAL":
        return "NORMAL"
    return "UNCLEAR"


# ---- 5. EVALUATION LOOP ------------------------------------------

def evaluate(port, test_set, label):
    """Run all test cases with majority-vote methodology.

    Each image is queried RUNS_PER_IMAGE times; the majority label
    becomes the prediction.
    """
    session = requests.Session()
    results = []
    correct = 0
    total   = len(test_set)

    logger.info(f"\n{'='*60}")
    logger.info(f"  EVALUATING: {label}  ({total} samples, "
                f"{RUNS_PER_IMAGE} runs each)")
    logger.info(f"  Prompt: {PROMPT[:60]}...")
    logger.info(f"  Params: temperature={TEMPERATURE}, min_p={MIN_P}")
    logger.info(f"{'='*60}")

    for i, case in enumerate(test_set):
        # Encode image once, reuse for all 5 runs
        data_url = image_to_data_url(case["image_path"])
        run_labels = []
        run_confidences = []
        run_responses = []
        t0 = time.time()

        for run_idx in range(RUNS_PER_IMAGE):
            try:
                response = query_server(port, data_url, session)
                lbl, conf = classify_response(response)
                run_labels.append(lbl)
                run_confidences.append(conf)
                run_responses.append(response.strip().replace("\n", " ")[:80])
            except Exception as e:
                logger.warning(f"    Run {run_idx+1} error: {e}")
                run_labels.append("UNCLEAR")
                run_confidences.append(None)
                run_responses.append(f"[ERROR: {e}]")

        elapsed = time.time() - t0

        # Majority vote
        vote_label, vote_count, vote_total = majority_vote(run_labels)
        pred = map_to_ground_truth(vote_label)
        ok = pred == case["ground_truth"]
        if ok:
            correct += 1

        # Average confidence (ignoring None)
        valid_confs = [c for c in run_confidences if c is not None]
        avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else None

        mark = "[OK]" if ok else "[FAIL]"
        votes_str = "/".join(run_labels)
        logger.info(
            f"  [{i+1}/{total}] {case['filename']:<25} "
            f"GT={case['ground_truth']:<6}  "
            f"Vote={vote_label:<8} ({vote_count}/{vote_total})  "
            f"Pred={pred:<6}  {mark}  "
            f"{elapsed:.1f}s"
        )
        logger.info(f"           Runs: {votes_str}")

        results.append({
            "filename":       case["filename"],
            "ground_truth":   case["ground_truth"],
            "predicted":      pred,
            "correct":        ok,
            "vote_label":     vote_label,
            "vote_count":     vote_count,
            "vote_total":     vote_total,
            "run_labels":     run_labels,
            "run_confidences": run_confidences,
            "avg_confidence":  round(avg_conf, 1) if avg_conf else None,
            "run_responses":  run_responses,
            "time_s":         round(elapsed, 1),
            "excerpt":        run_responses[0][:120] if run_responses else "",
        })

    accuracy = correct / total * 100 if total else 0
    logger.info(f"\n  >>> {label} accuracy: {correct}/{total} = {accuracy:.1f}%\n")
    return results, accuracy


# ---- 6. REPORT ---------------------------------------------------

def _confusion_counts(results):
    """Returns (TP, FN, FP, TN) treating TB as positive class."""
    tp = sum(1 for r in results
             if r["ground_truth"] == "TB" and r["predicted"] == "TB")
    fn = sum(1 for r in results
             if r["ground_truth"] == "TB" and r["predicted"] != "TB")
    fp = sum(1 for r in results
             if r["ground_truth"] == "NORMAL" and r["predicted"] == "TB")
    tn = sum(1 for r in results
             if r["ground_truth"] == "NORMAL" and r["predicted"] != "TB")
    return tp, fn, fp, tn


def write_report(base_results, base_acc, lora_results, lora_acc):
    lines = [
        "=" * 60,
        "  MedGemma GGUF Evaluation Report",
        "  Majority-Vote Methodology (5 runs per image)",
        "=" * 60,
        "",
        f"Model : {Path(GGUF_MODEL).name}",
        f"MMProj: {Path(MMPROJ).name}",
        f"Prompt: {PROMPT}",
        f"Params: temperature={TEMPERATURE}, min_p={MIN_P}",
        f"Runs per image: {RUNS_PER_IMAGE}",
        f"Test set: {len(base_results)} samples "
        f"({'Tawsifurrahman independent' if EVAL_CSV.exists() else 'Montgomery holdout'})",
        "",
    ]

    def _section(results, acc, label):
        lines.append("-" * 60)
        lines.append(f"  {label}: {acc:.1f}%")
        lines.append("-" * 60)

        tp, fn, fp, tn = _confusion_counts(results)
        total = tp + fn + fp + tn
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        ppv  = tp / (tp + fp) if (tp + fp) else 0
        npv  = tn / (tn + fn) if (tn + fn) else 0
        f1   = 2*ppv*sens / (ppv+sens) if (ppv+sens) else 0

        lines.append(f"  Sensitivity: {sens:.1%}  |  Specificity: {spec:.1%}")
        lines.append(f"  PPV: {ppv:.1%}  |  NPV: {npv:.1%}  |  F1: {f1:.3f}")
        lines.append(f"  Confusion: TP={tp}  FN={fn}  FP={fp}  TN={tn}")
        lines.append("")

        for r in results:
            mark = "[OK]" if r.get("correct") else "[FAIL]"
            votes = "/".join(r.get("run_labels", []))
            conf = r.get("avg_confidence")
            conf_str = f"{conf:.0f}%" if conf is not None else "N/A"
            lines.append(
                f"  {r['filename']:<25} GT={r['ground_truth']:<6} "
                f"Pred={r['predicted']:<6} {mark}  "
                f"Vote={r.get('vote_label','?'):<8} "
                f"({r.get('vote_count','?')}/{r.get('vote_total','?')})  "
                f"Conf={conf_str}  Runs=[{votes}]"
            )
        lines.append("")

    _section(base_results, base_acc, "Base Model")
    if lora_results and lora_acc >= 0:
        _section(lora_results, lora_acc, "Base + LoRA")

    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {REPORT_FILE}")
    print("\n" + report)


# ---- 7. MAIN -----------------------------------------------------

def find_lora_file():
    """Find the LoRA adapter file. Try .gguf first, then .safetensors."""
    lora_dir = Path(LORA_DIR)
    if not lora_dir.exists():
        return None
    for f in lora_dir.glob("*.gguf"):
        return str(f)
    st = lora_dir / "adapter_model.safetensors"
    if st.exists():
        return str(st)
    return None


def main():
    logger.info("=" * 60)
    logger.info("  MedGemma GGUF Evaluation — Majority Vote")
    logger.info(f"  {RUNS_PER_IMAGE} runs per image, Instruct preset")
    logger.info("=" * 60)

    # Validate paths
    for label, p in [("Model", GGUF_MODEL), ("MMProj", MMPROJ),
                     ("Server", LLAMA_SERVER)]:
        if not Path(p).exists():
            logger.error(f"{label} not found: {p}")
            sys.exit(1)

    test_set = load_test_set()

    # --- Run 1: Base model (no LoRA) ---
    proc, port = start_server(lora_path=None)
    try:
        base_results, base_acc = evaluate(port, test_set, "Base Model")
    finally:
        stop_server(proc)

    # --- Run 2: Base + LoRA ---
    lora_file = find_lora_file()
    if lora_file:
        logger.info(f"Found LoRA: {lora_file}")
        try:
            proc, port = start_server(lora_path=lora_file)
            try:
                lora_results, lora_acc = evaluate(
                    port, test_set, "Base + LoRA")
            finally:
                stop_server(proc)
        except Exception as e:
            logger.error(f"LoRA run failed: {e}")
            lora_results = []
            lora_acc = -1
    else:
        logger.warning("No LoRA adapter found — skipping LoRA evaluation")
        lora_results = []
        lora_acc = -1

    # --- Report ---
    write_report(base_results, base_acc, lora_results, lora_acc)

    # Save JSON (compatible with app.py's Evaluation Dashboard)
    json_path = SCRIPT_DIR / "evaluation_results_gguf.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_accuracy":  base_acc,
            "lora_accuracy":  lora_acc,
            "base_results":   base_results,
            "lora_results":   lora_results,
            "methodology":    "majority_vote",
            "dataset":        "tawsifurrahman_independent" if EVAL_CSV.exists() else "montgomery_holdout",
            "runs_per_image": RUNS_PER_IMAGE,
            "prompt":         PROMPT,
            "temperature":    TEMPERATURE,
            "min_p":          MIN_P,
        }, f, indent=2)
    logger.info(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()

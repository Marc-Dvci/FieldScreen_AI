"""
Train a logistic regression classifier on HeAR embeddings for TB cough detection.
================================================================================

This script extracts HeAR embeddings from labeled cough audio files and trains
a scikit-learn LogisticRegression classifier.  The resulting weights are saved
to Models/HeAR/tb_cough_classifier.npz so the FieldScreen AI app can load
them at runtime for real cough-based TB pre-screening.

Usage
-----
    python train_cough_classifier.py --data-dir <path> [--output <path>]

Data directory structure
------------------------
    data-dir/
        tb/           # TB-positive cough audio files (.wav, .mp3, .flac, .ogg)
        normal/       # Healthy / non-TB cough audio files

Each subdirectory should contain audio files of individual cough recordings
(2+ seconds recommended).  The script handles resampling and chunking
automatically.

Requirements
------------
    pip install numpy scikit-learn librosa soundfile torch transformers

If you already have the FieldScreen AI environment set up, all dependencies
should be available.

Output
------
    Models/HeAR/tb_cough_classifier.npz  (default)
        Contains:
            weights  — (512,) float32 array (logistic regression coefficients)
            bias     — scalar float32 (intercept)
            accuracy — scalar float32 (training accuracy for reference)
            n_tb     — int (number of TB samples used)
            n_normal — int (number of normal samples used)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Workaround for PyTorch 2.6+ CVE-2025-32434 ──
# PyTorch 2.6 changed torch.load to default weights_only=True,
# which breaks transformers model loading from local checkpoint files.
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ── Project paths ──
PROJECT_DIR = Path(__file__).parent.parent.resolve()
HEAR_LOCAL_DIR = PROJECT_DIR / "Models" / "HeAR"
DEFAULT_OUTPUT = HEAR_LOCAL_DIR / "tb_cough_classifier.npz"

HEAR_SAMPLE_RATE = 16000
HEAR_CLIP_SAMPLES = 32000   # 2 seconds at 16 kHz
HEAR_EMBEDDING_DIM = 512

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def find_audio_files(directory):
    """Recursively find audio files in a directory."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(files)


def load_hear_model():
    """Load the HeAR model (same logic as app.py HeARManager)."""
    import torch

    if (HEAR_LOCAL_DIR / "config.json").exists():
        from transformers import AutoModel
        print(f"Loading HeAR from {HEAR_LOCAL_DIR} ...")
        model = AutoModel.from_pretrained(
            str(HEAR_LOCAL_DIR),
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        model = model.to("cpu")
        return model, "pytorch"
    else:
        print(f"ERROR: No HeAR model found in {HEAR_LOCAL_DIR}")
        print("Download 'google/hear-pytorch' from HuggingFace first.")
        sys.exit(1)


def hear_mel_pcen(audio_tensor):
    """Convert raw 16 kHz audio to mel-PCEN spectrogram for HeAR.

    Replicates the official HeAR preprocessing pipeline.
    Input:  torch.Tensor of shape (batch, 32000)
    Output: torch.Tensor of shape (batch, 1, 192, 128)
    """
    import torch
    import torch.nn.functional as F
    import librosa as _lr

    mel_fb = torch.tensor(
        _lr.filters.mel(sr=HEAR_SAMPLE_RATE, n_fft=400, n_mels=128,
                        fmin=60.0, fmax=7800.0),
        dtype=torch.float32,
    )
    hann_win = torch.hann_window(400)

    specs = []
    for i in range(audio_tensor.shape[0]):
        stft = torch.stft(
            audio_tensor[i], n_fft=400, hop_length=160, win_length=400,
            window=hann_win, return_complex=True, center=True,
        )
        power = stft.abs() ** 2

        mel = torch.matmul(mel_fb, power)

        # Per-Channel Energy Normalization (PCEN)
        alpha, s, delta, root, eps = 0.8, 0.04, 2.0, 2.0, 1e-6
        smoothed = torch.zeros_like(mel)
        smoothed[:, 0] = mel[:, 0]
        for t in range(1, mel.shape[1]):
            smoothed[:, t] = (1 - s) * smoothed[:, t - 1] + s * mel[:, t]
        gain = (eps + smoothed) ** (-alpha)
        pcen = (mel * gain + delta) ** (1.0 / root) - delta ** (1.0 / root)

        pcen = pcen.T.unsqueeze(0).unsqueeze(0)
        pcen = F.interpolate(pcen, size=(192, 128),
                             mode="bilinear", align_corners=False)
        specs.append(pcen.squeeze(0))

    return torch.stack(specs)


def extract_embedding(model, audio_path):
    """Extract HeAR embedding from one audio file.

    Returns a (512,) numpy array (mean-pooled over 2-second chunks).
    """
    import torch
    import librosa as _lr

    audio, _ = _lr.load(str(audio_path), sr=HEAR_SAMPLE_RATE, mono=True)

    # Chunk into 2-second segments
    chunks = []
    if len(audio) <= HEAR_CLIP_SAMPLES:
        padded = np.zeros(HEAR_CLIP_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        chunks.append(padded)
    else:
        for start in range(0, len(audio) - HEAR_CLIP_SAMPLES + 1,
                           HEAR_CLIP_SAMPLES):
            chunk = audio[start:start + HEAR_CLIP_SAMPLES]
            chunks.append(chunk.astype(np.float32))
        remainder = len(audio) % HEAR_CLIP_SAMPLES
        if remainder > 0:
            padded = np.zeros(HEAR_CLIP_SAMPLES, dtype=np.float32)
            padded[:remainder] = audio[-remainder:]
            chunks.append(padded)

    chunks_np = np.stack(chunks)
    raw_tensor = torch.tensor(chunks_np, dtype=torch.float32)
    spec_batch = hear_mel_pcen(raw_tensor).to("cpu")

    with torch.no_grad():
        output = model(
            pixel_values=spec_batch,
            return_dict=True,
            output_hidden_states=True,
        )
        embeddings = output.pooler_output.cpu().numpy()

    return np.mean(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Train a HeAR cough classifier for TB detection")
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory with tb/ and normal/ subdirectories of cough audio")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output path for classifier weights (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="Fraction of data to hold out for evaluation (default: 0.2)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tb_dir = data_dir / "tb"
    normal_dir = data_dir / "normal"

    if not tb_dir.exists() or not normal_dir.exists():
        print(f"ERROR: Expected subdirectories {tb_dir} and {normal_dir}")
        print("Structure your data as:")
        print(f"  {data_dir}/")
        print(f"    tb/        (TB-positive cough audio)")
        print(f"    normal/    (healthy cough audio)")
        sys.exit(1)

    tb_files = find_audio_files(tb_dir)
    normal_files = find_audio_files(normal_dir)
    print(f"Found {len(tb_files)} TB files, {len(normal_files)} normal files")

    if len(tb_files) < 2 or len(normal_files) < 2:
        print("ERROR: Need at least 2 files per class for training.")
        sys.exit(1)

    # ── Load HeAR model ──
    model, backend = load_hear_model()
    print(f"HeAR loaded ({backend})")

    # ── Extract embeddings ──
    print("Extracting embeddings...")
    embeddings = []
    labels = []
    t0 = time.time()

    for i, f in enumerate(tb_files):
        try:
            emb = extract_embedding(model, f)
            embeddings.append(emb)
            labels.append(1)  # TB = positive
            print(f"  [{i+1}/{len(tb_files)+len(normal_files)}] "
                  f"TB: {f.name}")
        except Exception as e:
            print(f"  SKIP (error): {f.name} — {e}")

    for i, f in enumerate(normal_files):
        try:
            emb = extract_embedding(model, f)
            embeddings.append(emb)
            labels.append(0)  # Normal = negative
            print(f"  [{len(tb_files)+i+1}/{len(tb_files)+len(normal_files)}] "
                  f"Normal: {f.name}")
        except Exception as e:
            print(f"  SKIP (error): {f.name} — {e}")

    elapsed = time.time() - t0
    print(f"Extracted {len(embeddings)} embeddings in {elapsed:.1f}s")

    X = np.array(embeddings, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # ── Train/test split ──
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed,
        stratify=y,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train logistic regression ──
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=args.seed,
        C=1.0,
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTrain accuracy: {train_acc:.1%}")
    print(f"Test accuracy:  {test_acc:.1%}")
    print("\nTest set classification report:")
    print(classification_report(
        y_test, y_pred_test, target_names=["Normal", "TB"]))

    # ── Save weights ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = clf.coef_[0].astype(np.float32)       # (512,)
    bias = np.float32(clf.intercept_[0])              # scalar

    np.savez(
        str(output_path),
        weights=weights,
        bias=bias,
        accuracy=np.float32(test_acc),
        n_tb=np.int32(sum(y == 1)),
        n_normal=np.int32(sum(y == 0)),
    )
    print(f"\nClassifier saved to: {output_path}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Bias: {bias:.6f}")
    print(f"\nThe FieldScreen AI app will automatically load these weights "
          f"on next startup.")


if __name__ == "__main__":
    main()

"""
FieldScreen AI — TB Screening Demo App
=======================================
Multi-model HAI-DEF pipeline for tuberculosis detection in chest X-rays.
Uses MedGemma (vision-language CXR analysis) + MedASR (medical speech-to-text)
+ HeAR (health acoustic cough analysis) + WHO-aligned clinical symptom scoring
+ TranslateGemma (multilingual report translation).
3-tab layout: Clinical Workflow, Reports, About.
"""

import os

# ── Offline-first: prevent any network calls to HuggingFace or Gradio ──
# These MUST be set before importing transformers / gradio.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import re
import base64
import time
import socket
import subprocess
import threading
import atexit
import logging
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import requests

# ============================================================
# CONFIGURATION — auto-detected, override via environment vars
# ============================================================
PROJECT_DIR = Path(__file__).parent.resolve()


def _find_gguf(directory, pattern="*.gguf", exclude=None):
    """Find the first GGUF file in *directory* matching *pattern*."""
    d = Path(directory)
    if not d.exists():
        return None
    for f in sorted(d.glob(pattern)):
        if exclude and exclude in f.name.lower():
            continue
        return str(f)
    return None


def _find_binary(name):
    """Find an executable in project bin/, venv, or system PATH."""
    import shutil
    suffix = ".exe" if os.name == "nt" else ""
    # 1. project bin/
    p = PROJECT_DIR / "bin" / (name + suffix)
    if p.exists():
        return str(p)
    # 2. venv site-packages (llama_cpp_binaries)
    venv_sp = PROJECT_DIR / "venv" / "Lib" / "site-packages"
    if venv_sp.exists():
        for hit in venv_sp.glob(f"llama_cpp_binaries/bin/{name}{suffix}"):
            return str(hit)
    # 3. system PATH
    found = shutil.which(name)
    if found:
        return found
    return None


# ── Paths (project-local; override via FIELDSCREEN_* environment variables) ──
GGUF_MODEL = (
    os.environ.get("FIELDSCREEN_GGUF_MODEL")
    or _find_gguf(PROJECT_DIR / "Models" / "MedGemma", exclude="mmproj")
    or str(PROJECT_DIR / "Models" / "MedGemma" / "medgemma-1.5-4b-it-Q4_K_M.gguf")
)
MMPROJ = (
    os.environ.get("FIELDSCREEN_MMPROJ")
    or _find_gguf(PROJECT_DIR / "Models" / "MedGemma", pattern="mmproj*.gguf")
    or str(PROJECT_DIR / "Models" / "MedGemma" / "mmproj-BF16.gguf")
)
LORA_DIR = PROJECT_DIR / "Models" / "MedGemma"
LLAMA_SERVER = (
    os.environ.get("FIELDSCREEN_LLAMA_SERVER")
    or _find_binary("llama-server")
    or str(PROJECT_DIR / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server"))
)

GPU_LAYERS = -1
CTX_SIZE   = 4096
MAX_TOKENS = 300

# MedASR — voice-to-text for clinical notes (HAI-DEF model #2)
MEDASR_MODEL_ID  = "google/medasr"
MEDASR_LOCAL_DIR = PROJECT_DIR / "Models" / "MedASR"   # offline-first
HF_TOKEN_FILE    = PROJECT_DIR / "hf_token.txt"   # optional HuggingFace token

# HeAR — cough analysis for TB pre-screening (HAI-DEF model #3)
HEAR_MODEL_ID    = "google/hear-pytorch"
HEAR_LOCAL_DIR   = PROJECT_DIR / "Models" / "HeAR"     # offline-first
HEAR_SAMPLE_RATE  = 16000
HEAR_CLIP_SAMPLES = 32000   # 2 seconds at 16 kHz
HEAR_EMBEDDING_DIM = 512

# TranslateGemma — multilingual report translation (Google model #4)
TRANSLATE_GEMMA_DIR  = PROJECT_DIR / "Models" / "TranslateGemma"
TRANSLATE_GEMMA_GGUF = None   # auto-detected from TRANSLATE_GEMMA_DIR
TRANSLATE_GPU_LAYERS = 0      # CPU-only to avoid VRAM conflict with MedGemma
TRANSLATE_CTX_SIZE   = 2048
TRANSLATE_MAX_TOKENS = 600

# Languages supported — major TB-burden regions
TRANSLATE_LANGUAGES = {
    "Hindi":      "hi",
    "French":     "fr",
    "Spanish":    "es",
    "Arabic":     "ar",
    "Swahili":    "sw",
    "Portuguese": "pt",
    "Bengali":    "bn",
    "Urdu":       "ur",
    "Chinese":    "zh",
    "Russian":    "ru",
    "Turkish":    "tr",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Thai":       "th",
    "Amharic":    "am",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("fieldscreen")

# Log resolved paths at startup
logger.info("GGUF_MODEL:   %s  (exists: %s)", GGUF_MODEL, Path(GGUF_MODEL).exists())
logger.info("MMPROJ:       %s  (exists: %s)", MMPROJ, Path(MMPROJ).exists())
logger.info("LLAMA_SERVER: %s  (exists: %s)", LLAMA_SERVER, Path(LLAMA_SERVER).exists())
logger.info("LORA_DIR:     %s  (exists: %s)", LORA_DIR, LORA_DIR.exists())

# ============================================================
# CLASSIFICATION — parse NORMAL/ABNORMAL from evaluation-style responses
# ============================================================
def classify_response(response_text):
    """Parse the model's classification response for NORMAL/ABNORMAL.

    The classification prompt asks for 'NORMAL or ABNORMAL, then confidence %'.
    We look for the first occurrence of either keyword and extract the
    confidence percentage.  Falls back to keyword search if neither
    NORMAL nor ABNORMAL is found.

    Returns (label, confidence_pct_or_None).
      label is 'TB' or 'NORMAL'.
    """
    text = response_text.strip().upper()

    # Extract confidence percentage (e.g. "85%")
    conf_match = re.search(r'(\d{1,3})(?:\.\d+)?\s*%', response_text)
    confidence = int(conf_match.group(1)) if conf_match else None

    # Find positions of NORMAL and ABNORMAL
    pos_abnormal = text.find("ABNORMAL")
    pos_normal   = text.find("NORMAL")

    # "ABNORMAL" contains "NORMAL", so disambiguate
    if pos_abnormal >= 0:
        if pos_normal >= 0 and pos_normal == pos_abnormal + 2:
            # "NORMAL" is a substring of "ABNORMAL"
            return "TB", confidence
        elif pos_normal >= 0 and pos_normal < pos_abnormal:
            return "NORMAL", confidence
        else:
            return "TB", confidence

    if pos_normal >= 0:
        return "NORMAL", confidence

    # Fallback: keyword search if model didn't use NORMAL/ABNORMAL
    lower = response_text.lower()
    tb_kw = any(w in lower for w in (
        "tuberculosis", " tb ", "infiltrat", "opacit", "cavit",
        "consolidat", "suggestive",
    ))
    nrm_kw = any(w in lower for w in (
        "clear", "healthy", "unremarkable", "no findings",
    ))
    if tb_kw and not nrm_kw:
        return "TB", confidence
    if nrm_kw and not tb_kw:
        return "NORMAL", confidence

    # Last resort: conservative — flag as TB (better safe than sorry)
    return "TB", confidence


def find_lora_file():
    """Find the LoRA adapter (.gguf preferred, then .safetensors)."""
    lora_dir = Path(LORA_DIR)
    if not lora_dir.exists():
        return None
    for f in lora_dir.glob("*.gguf"):
        return str(f)
    st = lora_dir / "adapter_model.safetensors"
    if st.exists():
        return str(st)
    return None


# ============================================================
# SERVER MANAGER
# ============================================================
class ServerManager:
    """Manages the llama-server process for inference."""

    def __init__(self):
        self._proc = None
        self._port = None
        self._mode = None        # "base" or "lora"
        self._lock = threading.Lock()

    @staticmethod
    def _find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def _drain_stderr(proc, label="llama-server"):
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                for line in chunk.decode("utf-8", errors="replace").splitlines():
                    if line.strip():
                        logger.debug("[%s] %s", label, line.rstrip())
        except Exception:
            pass

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._port = None
        self._mode = None

    def _start(self, use_lora=False):
        self._stop()
        port = self._find_free_port()
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
        if use_lora:
            lora_file = find_lora_file()
            if lora_file:
                cmd += ["--lora", lora_file]
                logger.info("Starting server WITH LoRA: %s", lora_file)
            else:
                logger.warning("LoRA requested but no file found! "
                               "Running base model only.")
        else:
            logger.info("Starting server WITHOUT LoRA (base model only).")

        # Ensure companion DLLs (ggml.dll, llama.dll, etc.) are findable
        env = os.environ.copy()
        server_dir = str(Path(LLAMA_SERVER).parent)
        env["PATH"] = server_dir + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, bufsize=0,
                                env=env)
        threading.Thread(target=self._drain_stderr, args=(proc,),
                         daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        session = requests.Session()
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {proc.poll()}")
            try:
                r = session.get(health_url, timeout=2)
                if r.status_code == 200:
                    self._proc = proc
                    self._port = port
                    self._mode = "lora" if use_lora else "base"
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)

        proc.terminate()
        raise TimeoutError("Server did not become healthy within 120 s")

    def ensure_ready(self, use_lora=False):
        """Start or reuse the server. Returns the port number."""
        with self._lock:
            desired = "lora" if use_lora else "base"
            if (self._proc and self._proc.poll() is None
                    and self._mode == desired):
                return self._port
            self._start(use_lora)
            return self._port

    def cleanup(self):
        with self._lock:
            self._stop()


server_manager = ServerManager()
atexit.register(server_manager.cleanup)


# ============================================================
# MEDASR MANAGER (HAI-DEF model #2 — medical speech-to-text)
# ============================================================
def _read_hf_token():
    """Try to find a HuggingFace token for gated model access."""
    # 1. Environment variable
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    # 2. Token file in project directory
    if HF_TOKEN_FILE.exists():
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    # 3. Let transformers use its cached login (~/.cache/huggingface/token)
    return None


class MedASRManager:
    """Lazy-loaded MedASR pipeline for medical speech recognition.

    Offline-first: checks MEDASR_LOCAL_DIR for a pre-downloaded model
    before falling back to the HuggingFace Hub.  After the first Hub
    download, transformers caches the model so subsequent loads are
    offline-capable automatically.
    """

    def __init__(self):
        self._pipe = None
        self._error = None
        self._source = None     # "local" or "hub"
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._pipe is not None

    @staticmethod
    def _has_local_model():
        """Check if a local model directory has model files."""
        if not MEDASR_LOCAL_DIR.exists():
            return False
        # A valid local model dir has at least config.json
        return (MEDASR_LOCAL_DIR / "config.json").exists()

    def load(self):
        """Load MedASR model. Returns True on success."""
        with self._lock:
            if self._pipe is not None:
                return True
            try:
                from transformers import pipeline as hf_pipeline
                token = _read_hf_token()

                if self._has_local_model():
                    model_path = str(MEDASR_LOCAL_DIR)
                    logger.info(f"Loading MedASR from local: {model_path}")
                    self._pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model=model_path,
                        device="cpu",
                        trust_remote_code=True,
                    )
                    self._source = "local"
                else:
                    logger.info(
                        "Loading MedASR from HuggingFace Hub "
                        "(first use — may download ~400 MB)...")
                    self._pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model=MEDASR_MODEL_ID,
                        device="cpu",
                        trust_remote_code=True,
                        token=token,
                    )
                    self._source = "hub"

                logger.info("MedASR loaded successfully.")
                return True
            except Exception as e:
                msg = str(e)
                if "lasr_ctc" in msg or "does not recognize this architecture" in msg:
                    msg = (
                        "MedASR requires transformers >= 5.0.  "
                        "Install from source: pip install "
                        "git+https://github.com/huggingface/transformers.git"
                    )
                self._error = msg
                logger.error(f"MedASR load failed: {msg}")
                return False

    @staticmethod
    def _clean_ctc_output(text):
        """Post-process CTC output: remove blank tokens and duplicates.

        The MedASR CTC model uses <epsilon> as the blank/pad token.
        The transformers pipeline may not always collapse these properly,
        producing output like '<epsilon>P<epsilon>atiientent<epsilon>'.
        """
        import re as _re
        # Strip CTC blank tokens and special tokens
        text = text.replace("<epsilon>", "")
        text = _re.sub(r"</?s>", "", text)
        text = _re.sub(r"<extra_id_\d+>", "", text)
        # Collapse duplicate consecutive words (e.g. "re report" → "report")
        text = _re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
        # Clean up whitespace
        text = _re.sub(r"\s+", " ", text).strip()
        return text

    def transcribe(self, audio_path):
        """Transcribe an audio file. Returns text string."""
        if not self.is_loaded:
            if not self.load():
                return f"[MedASR unavailable: {self._error}]"
        try:
            result = self._pipe(
                audio_path,
                chunk_length_s=20,
                stride_length_s=2,
            )
            text = result.get("text", "").strip()
            text = self._clean_ctc_output(text)
            logger.info(f"MedASR transcription: {text[:80]}...")
            return text
        except Exception as e:
            return f"[Transcription error: {e}]"

    @property
    def status_message(self):
        if self._pipe is not None:
            src = "local" if self._source == "local" else "hub/cache"
            return f"MedASR loaded and ready ({src})"
        if self._error:
            return f"MedASR error: {self._error}"
        if self._has_local_model():
            return "MedASR ready (local model found, loads on first use)"
        return "MedASR not loaded yet (loads on first use from Hub)"


medasr_manager = MedASRManager()


def transcribe_audio(audio_path):
    """Gradio callback: transcribe audio from microphone or file upload."""
    if audio_path is None:
        return gr.update()
    return medasr_manager.transcribe(audio_path)


# ============================================================
# HEAR MANAGER (HAI-DEF model #3 — health acoustic cough analysis)
# ============================================================
def _hear_mel_pcen(audio_tensor):
    """Convert raw 16 kHz audio to mel-PCEN spectrogram for HeAR.

    Replicates the official HeAR preprocessing pipeline:
      raw audio → STFT → power spectrogram → mel filterbank → PCEN → resize.

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
        # STFT → power spectrogram
        stft = torch.stft(
            audio_tensor[i], n_fft=400, hop_length=160, win_length=400,
            window=hann_win, return_complex=True, center=True,
        )
        power = stft.abs() ** 2                       # (201, T)

        # Mel filterbank
        mel = torch.matmul(mel_fb, power)             # (128, T)

        # Per-Channel Energy Normalization (PCEN)
        alpha, s, delta, root, eps = 0.8, 0.04, 2.0, 2.0, 1e-6
        smoothed = torch.zeros_like(mel)
        smoothed[:, 0] = mel[:, 0]
        for t in range(1, mel.shape[1]):
            smoothed[:, t] = (1 - s) * smoothed[:, t - 1] + s * mel[:, t]
        gain = (eps + smoothed) ** (-alpha)
        pcen = (mel * gain + delta) ** (1.0 / root) - delta ** (1.0 / root)

        # Resize to (1, 192, 128): time × mel
        pcen = pcen.T.unsqueeze(0).unsqueeze(0)       # (1, 1, T, 128)
        pcen = F.interpolate(pcen, size=(192, 128),
                             mode="bilinear", align_corners=False)
        specs.append(pcen.squeeze(0))                  # (1, 192, 128)

    return torch.stack(specs)                          # (B, 1, 192, 128)


class HeARManager:
    """Lazy-loaded HeAR model for health acoustic cough analysis.

    Supports two model formats:
    - TensorFlow SavedModel (saved_model.pb + variables/) — loaded with tf.saved_model.load
    - HuggingFace PyTorch (config.json + model.safetensors) — loaded with AutoModel
    """

    def __init__(self):
        self._model = None
        self._infer_fn = None   # callable for TF models
        self._backend = None    # "tf" or "pytorch"
        self._error = None
        self._source = None     # "local", "cache", "hub"
        self._lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._model is not None

    @staticmethod
    def _has_local_hf_model():
        """Check for HuggingFace PyTorch format (config.json)."""
        if not HEAR_LOCAL_DIR.exists():
            return False
        return (HEAR_LOCAL_DIR / "config.json").exists()

    @staticmethod
    def _has_local_tf_model():
        """Check for TensorFlow SavedModel format (saved_model.pb)."""
        if not HEAR_LOCAL_DIR.exists():
            return False
        return (HEAR_LOCAL_DIR / "saved_model.pb").exists()

    def load(self):
        """Load HeAR model. Returns True on success."""
        with self._lock:
            if self._model is not None:
                return True
            try:
                if self._has_local_hf_model():
                    # --- HuggingFace PyTorch format ---
                    import torch as _torch
                    from transformers import AutoModel
                    model_path = str(HEAR_LOCAL_DIR)
                    logger.info(f"Loading HeAR from local (PyTorch): {model_path}")
                    self._model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    self._model.eval()
                    self._model = self._model.to("cpu")
                    self._backend = "pytorch"
                    self._source = "local"

                elif self._has_local_tf_model():
                    # --- TensorFlow SavedModel format ---
                    try:
                        import tensorflow as tf
                    except ImportError:
                        raise ImportError(
                            "HeAR model is in TensorFlow SavedModel format "
                            f"({HEAR_LOCAL_DIR / 'saved_model.pb'}) but "
                            "TensorFlow is not installed.  Either:\n"
                            "  1. pip install tensorflow\n"
                            "  2. Replace with PyTorch format: download "
                            "'google/hear-pytorch' from HuggingFace "
                            "(config.json + model.safetensors)"
                        )
                    model_path = str(HEAR_LOCAL_DIR)
                    logger.info(f"Loading HeAR from local (TensorFlow): {model_path}")
                    self._model = tf.saved_model.load(model_path)
                    # Resolve the inference callable
                    if hasattr(self._model, 'signatures'):
                        sig_keys = list(self._model.signatures.keys())
                        logger.info(f"HeAR TF signatures: {sig_keys}")
                        if 'serving_default' in self._model.signatures:
                            self._infer_fn = self._model.signatures['serving_default']
                        elif sig_keys:
                            self._infer_fn = self._model.signatures[sig_keys[0]]
                    if self._infer_fn is None and callable(self._model):
                        self._infer_fn = self._model
                    self._backend = "tf"
                    self._source = "local"
                else:
                    raise FileNotFoundError(
                        f"No HeAR model found in {HEAR_LOCAL_DIR}. "
                        "Download 'google/hear-pytorch' from HuggingFace "
                        "(needs config.json + model.safetensors) or "
                        "the TF SavedModel (needs saved_model.pb + variables/)."
                    )

                logger.info(f"HeAR loaded ({self._backend}, {self._source}).")
                return True
            except Exception as e:
                self._error = str(e)
                logger.error(f"HeAR load failed: {e}")
                return False

    def _infer_tf(self, chunks_np):
        """Run inference with TensorFlow SavedModel.

        Tries raw audio input first, then mel-PCEN spectrogram input.
        Returns numpy embeddings of shape (n_chunks, embedding_dim).
        """
        import tensorflow as tf

        # --- Attempt 1: feed raw audio waveforms ---
        try:
            audio_input = tf.constant(chunks_np, dtype=tf.float32)
            result = self._infer_fn(audio_input)
            # Extract embedding tensor from result dict or direct tensor
            if isinstance(result, dict):
                # Common keys: 'embedding', 'embeddings', 'output_0', 'pooler_output'
                for key in ('embedding', 'embeddings', 'pooler_output',
                            'output_0', 'last_hidden_state'):
                    if key in result:
                        emb = result[key].numpy()
                        if emb.ndim == 2 and emb.shape[-1] == HEAR_EMBEDDING_DIM:
                            return emb
                # Fall back to first tensor that looks like embeddings
                for key, val in result.items():
                    arr = val.numpy()
                    if arr.ndim == 2 and arr.shape[-1] == HEAR_EMBEDDING_DIM:
                        return arr
                # Last resort: take the first output
                first_key = list(result.keys())[0]
                return result[first_key].numpy()
            else:
                return result.numpy()
        except Exception as e:
            logger.debug(f"HeAR TF raw-audio attempt failed: {e}")

        # --- Attempt 2: feed mel-PCEN spectrograms ---
        import torch as _torch
        raw_tensor = _torch.tensor(chunks_np, dtype=_torch.float32)
        spec_batch = _hear_mel_pcen(raw_tensor).numpy()
        spec_input = tf.constant(spec_batch, dtype=tf.float32)
        result = self._infer_fn(spec_input)
        if isinstance(result, dict):
            for key in ('embedding', 'embeddings', 'pooler_output',
                        'output_0', 'last_hidden_state'):
                if key in result:
                    return result[key].numpy()
            first_key = list(result.keys())[0]
            return result[first_key].numpy()
        return result.numpy()

    def extract_embeddings(self, audio_path):
        """Extract HeAR embeddings from an audio file.

        Returns (embeddings_np, n_chunks, duration_s).
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError(f"HeAR unavailable: {self._error}")

        import librosa as _lr

        # Load audio — librosa handles wav, mp3, flac, ogg, etc.
        audio, _ = _lr.load(audio_path, sr=HEAR_SAMPLE_RATE, mono=True)
        duration_s = len(audio) / HEAR_SAMPLE_RATE

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

        if self._backend == "tf":
            embeddings = self._infer_tf(chunks_np)
        else:
            # PyTorch path (original)
            import torch as _torch
            raw_tensor = _torch.tensor(chunks_np, dtype=_torch.float32)
            spec_batch = _hear_mel_pcen(raw_tensor).to("cpu")
            with _torch.no_grad():
                output = self._model(
                    pixel_values=spec_batch,
                    return_dict=True,
                    output_hidden_states=True,
                )
                embeddings = output.pooler_output.cpu().numpy()

        return embeddings, len(chunks), duration_s

    def analyze_cough(self, audio_path):
        """Full cough analysis pipeline.

        Returns a dict with risk_score, risk_level, embedding stats, etc.
        Requires a trained classifier (Models/HeAR/tb_cough_classifier.npz).
        If the classifier weights are missing, risk_score is None and
        risk_level is "UNKNOWN".
        """
        t0 = time.time()
        embeddings, n_chunks, duration = self.extract_embeddings(audio_path)
        elapsed = time.time() - t0

        # Aggregate embeddings (mean pooling across chunks)
        agg = np.mean(embeddings, axis=0)   # (512,)

        # Load trained logistic-regression classifier weights
        classifier_path = HEAR_LOCAL_DIR / "tb_cough_classifier.npz"
        if classifier_path.exists():
            data = np.load(str(classifier_path))
            w = data["weights"].astype(np.float32)   # (512,)
            b = float(data["bias"])                    # scalar
            logit = float(np.dot(agg, w) + b)
            risk_score = 1.0 / (1.0 + np.exp(-logit))
            classifier_status = "trained"
        else:
            risk_score = None
            classifier_status = "untrained"

        if classifier_status == "untrained":
            risk_level = "UNKNOWN"
        elif risk_score > 0.65:
            risk_level = "HIGH"
        elif risk_score > 0.40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "classifier_status": classifier_status,
            "n_chunks": n_chunks,
            "duration_s": duration,
            "elapsed_s": elapsed,
            "embedding_norm": float(np.linalg.norm(agg)),
            "embedding_mean": float(np.mean(agg)),
            "embedding_std": float(np.std(agg)),
        }

    @property
    def _classifier_status(self):
        if (HEAR_LOCAL_DIR / "tb_cough_classifier.npz").exists():
            return "classifier trained"
        return "classifier not trained"

    @property
    def status_message(self):
        if self._model is not None:
            return (f"HeAR loaded ({self._backend}, {self._source}, "
                    f"{self._classifier_status})")
        if self._error:
            return f"HeAR error: {self._error}"
        if self._has_local_hf_model() or self._has_local_tf_model():
            return (f"HeAR ready (local model found, "
                    f"{self._classifier_status})")
        return "HeAR not loaded yet (no local model found)"


hear_manager = HeARManager()


def analyze_cough_audio(audio_path):
    """Gradio callback: analyse cough audio with HeAR."""
    if audio_path is None:
        return (
            '<div style="color:#856404;padding:12px;text-align:center;">'
            'Record or upload a cough audio sample.</div>',
            "",
            "",
        )
    try:
        result = hear_manager.analyze_cough(audio_path)
    except Exception as e:
        return (
            f'<div style="background:#f8d7da;color:#721c24;padding:12px;'
            f'border-radius:8px;"><b>HeAR error:</b> {e}</div>',
            "",
            "",
        )

    details = (
        f"Embedding: {HEAR_EMBEDDING_DIM}-dim | "
        f"Norm: {result['embedding_norm']:.2f} | "
        f"Mean: {result['embedding_mean']:.4f} | "
        f"Std: {result['embedding_std']:.4f}\n"
        f"Audio: {result['duration_s']:.1f}s in "
        f"{result['n_chunks']} chunk(s)"
    )

    if result.get("classifier_status") == "untrained":
        badge_html = (
            '<div style="background:#17a2b8;color:white;padding:14px;'
            'border-radius:10px;text-align:center;font-size:1.1em;">'
            'HeAR embeddings extracted successfully<br>'
            '<span style="font-size:0.8em;opacity:0.9;">'
            'Cough classifier not yet trained &mdash; run '
            'train_cough_classifier.py with labeled data</span></div>'
            f'<div style="margin-top:8px;font-size:0.85em;color:#6c757d;'
            f'text-align:center;">'
            f'HeAR embedding extraction &bull; '
            f'{result["elapsed_s"]:.1f}s</div>'
        )
        summary = ("HeAR cough analysis: embeddings extracted "
                    "(classifier not trained)")
        return badge_html, details, summary

    risk  = result["risk_level"]
    score = result["risk_score"]
    if risk == "HIGH":
        bg, fg = "#dc3545", "white"
    elif risk == "MODERATE":
        bg, fg = "#fff3cd", "#856404"
    else:
        bg, fg = "#28a745", "white"

    badge_html = (
        f'<div style="background:{bg};color:{fg};padding:14px;'
        f'border-radius:10px;text-align:center;font-size:1.3em;'
        f'font-weight:bold;letter-spacing:1px;">'
        f'Cough Risk: {risk} ({score:.0%})</div>'
        f'<div style="margin-top:8px;font-size:0.85em;color:#6c757d;'
        f'text-align:center;">'
        f'HeAR embedding analysis &bull; {result["elapsed_s"]:.1f}s</div>'
    )
    summary = (
        f"HeAR cough pre-screening: {risk} RISK "
        f"(score: {score:.2f})"
    )
    return badge_html, details, summary


# ============================================================
# TRANSLATEGEMMA MANAGER (multilingual report translation)
# ============================================================
def _find_translate_gemma_gguf():
    """Auto-detect the TranslateGemma GGUF file in the model directory."""
    if TRANSLATE_GEMMA_GGUF:
        return TRANSLATE_GEMMA_GGUF
    if not TRANSLATE_GEMMA_DIR.exists():
        return None
    for f in sorted(TRANSLATE_GEMMA_DIR.glob("*.gguf")):
        return str(f)
    return None


class TranslateGemmaManager:
    """Manages a separate llama-server instance for TranslateGemma.

    Runs on CPU by default (TRANSLATE_GPU_LAYERS=0) so it can coexist
    with the MedGemma server that holds the GPU.  The server is started
    on first translation request and kept alive for subsequent ones.
    """

    def __init__(self):
        self._proc = None
        self._port = None
        self._lock = threading.Lock()
        self._model_path = None

    @property
    def is_available(self):
        return _find_translate_gemma_gguf() is not None

    def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._port = None

    def _start(self):
        self._stop()
        model_path = _find_translate_gemma_gguf()
        if not model_path:
            raise FileNotFoundError(
                f"No .gguf file found in {TRANSLATE_GEMMA_DIR}. "
                "Place the TranslateGemma GGUF model there."
            )
        self._model_path = model_path

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        cmd = [
            LLAMA_SERVER,
            "--model",      model_path,
            "--ctx-size",   str(TRANSLATE_CTX_SIZE),
            "--gpu-layers", str(TRANSLATE_GPU_LAYERS),
            "--port",       str(port),
            "--no-webui",
        ]

        logger.info(f"Starting TranslateGemma server on port {port} "
                     f"(GPU layers: {TRANSLATE_GPU_LAYERS})...")

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=ServerManager._drain_stderr,
                         args=(proc, "TranslateGemma"),
                         daemon=True).start()

        health_url = f"http://127.0.0.1:{port}/health"
        session = requests.Session()
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"TranslateGemma server exited with code {proc.poll()}")
            try:
                r = session.get(health_url, timeout=2)
                if r.status_code == 200:
                    self._proc = proc
                    self._port = port
                    logger.info("TranslateGemma server ready.")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)

        proc.terminate()
        raise TimeoutError(
            "TranslateGemma server did not become healthy within 120 s")

    def ensure_ready(self):
        """Start or reuse the TranslateGemma server.  Returns port."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return self._port
            self._start()
            return self._port

    def translate(self, text, target_lang_code, target_lang_name):
        """Translate English text to the target language.

        Uses the OpenAI-compatible chat completions endpoint.
        """
        if not text or not text.strip():
            return ""
        port = self.ensure_ready()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        prompt = (
            f"Translate the following medical screening report from "
            f"English to {target_lang_name}. "
            f"Preserve all medical terminology, numbers, and formatting. "
            f"Do not add any commentary.\n\n{text.strip()}"
        )

        payload = {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": TRANSLATE_MAX_TOKENS,
            "temperature": 0.1,
            "stream": False,
        }

        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[Translation error: {e}]"

    def cleanup(self):
        with self._lock:
            self._stop()

    @property
    def status_message(self):
        if self._proc is not None and self._proc.poll() is None:
            model_name = Path(self._model_path).stem if self._model_path else "?"
            return f"TranslateGemma running ({model_name}, port {self._port})"
        gguf = _find_translate_gemma_gguf()
        if gguf:
            return f"TranslateGemma ready ({Path(gguf).name})"
        return (
            f"TranslateGemma not available — place a .gguf file in "
            f"{TRANSLATE_GEMMA_DIR}"
        )


translate_manager = TranslateGemmaManager()
atexit.register(translate_manager.cleanup)


def translate_report(report_text, language_choice):
    """Gradio callback: translate the screening report."""
    if not report_text or not report_text.strip():
        return (
            '<div style="color:#856404;padding:12px;text-align:center;">'
            'Run an X-ray analysis first to generate a report.</div>',
            translate_manager.status_message,
        )

    if not translate_manager.is_available:
        return (
            '<div style="background:#f8d7da;color:#721c24;padding:12px;'
            'border-radius:8px;">'
            '<b>TranslateGemma not available.</b> Place the GGUF model '
            f'file in <code>{TRANSLATE_GEMMA_DIR}</code>.</div>',
            translate_manager.status_message,
        )

    lang_code = TRANSLATE_LANGUAGES.get(language_choice)
    if not lang_code:
        return (
            '<div style="color:#856404;padding:12px;text-align:center;">'
            'Select a target language.</div>',
            translate_manager.status_message,
        )

    try:
        translated = translate_manager.translate(
            report_text, lang_code, language_choice)
    except Exception as e:
        return (
            f'<div style="background:#f8d7da;color:#721c24;padding:12px;'
            f'border-radius:8px;"><b>Translation error:</b> {e}</div>',
            translate_manager.status_message,
        )

    result_html = (
        f'<div style="background:#e8f5e9;border:1px solid #4caf50;'
        f'border-radius:8px;padding:14px;">'
        f'<div style="font-weight:bold;margin-bottom:6px;color:#2e7d32;">'
        f'Report translated to {language_choice} ({lang_code})</div>'
        f'<div style="white-space:pre-wrap;font-size:0.95em;color:#333;">'
        f'{translated}</div></div>'
    )
    return result_html, translate_manager.status_message


# ============================================================
# WHO-ALIGNED CLINICAL SYMPTOM SCORING
# ============================================================
# WHO 4-symptom screen: any 1+ of (cough ≥2 weeks, fever, night sweats,
# weight loss) is a positive TB screen.  Risk factors (HIV, TB contact,
# prior TB) raise pre-test probability.  Combined with imaging and cough
# analysis to produce an overall risk level.

# Symptom weights (aligned with WHO sensitivity targets)
WHO_SYMPTOM_WEIGHTS = {
    "cough_2wk":     3,   # cardinal symptom — most specific
    "fever":         2,
    "night_sweats":  2,
    "weight_loss":   2,
}
RISK_FACTOR_WEIGHTS = {
    "hiv_positive":  3,   # HIV is the strongest known risk factor
    "tb_contact":    2,
    "prior_tb":      2,
}

MAX_SYMPTOM_SCORE  = sum(WHO_SYMPTOM_WEIGHTS.values())   # 9
MAX_RISK_SCORE     = sum(RISK_FACTOR_WEIGHTS.values())    # 7
MAX_CLINICAL_SCORE = MAX_SYMPTOM_SCORE + MAX_RISK_SCORE   # 16


def compute_who_symptom_score(symptoms, risk_factors):
    """Compute WHO-aligned clinical symptom score.

    Parameters
    ----------
    symptoms : list[str]
        Checked symptom keys from the form (e.g. ["cough_2wk", "fever"]).
    risk_factors : list[str]
        Checked risk-factor keys (e.g. ["hiv_positive", "tb_contact"]).

    Returns
    -------
    dict with:
        symptom_score, risk_factor_score, clinical_score,
        who_screen_positive (bool), risk_level, details (str).
    """
    symptoms     = symptoms     or []
    risk_factors = risk_factors or []

    symptom_score = sum(WHO_SYMPTOM_WEIGHTS.get(s, 0) for s in symptoms)
    rf_score      = sum(RISK_FACTOR_WEIGHTS.get(r, 0) for r in risk_factors)
    clinical_score = symptom_score + rf_score

    # WHO screen: positive if ANY of the 4 symptoms is present
    who_positive = len(symptoms) > 0

    # Clinical risk level from symptom + risk-factor score alone
    if clinical_score >= 10:
        risk_level = "HIGH"
    elif clinical_score >= 5:
        risk_level = "MODERATE"
    elif clinical_score >= 1:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    details_parts = []
    if symptoms:
        names = [s.replace("_", " ").title() for s in symptoms]
        details_parts.append(f"Symptoms: {', '.join(names)}")
    if risk_factors:
        names = [r.replace("_", " ").title() for r in risk_factors]
        details_parts.append(f"Risk factors: {', '.join(names)}")
    details_parts.append(
        f"Clinical score: {clinical_score}/{MAX_CLINICAL_SCORE} "
        f"(symptoms {symptom_score}/{MAX_SYMPTOM_SCORE} + "
        f"risk factors {rf_score}/{MAX_RISK_SCORE})"
    )

    return {
        "symptom_score":      symptom_score,
        "risk_factor_score":  rf_score,
        "clinical_score":     clinical_score,
        "who_screen_positive": who_positive,
        "risk_level":         risk_level,
        "details":            " | ".join(details_parts),
    }


def generate_notes_from_symptoms(symptoms, risk_factors):
    """Build a clinical-notes string from the structured symptom form.

    This is auto-appended to any existing typed / MedASR-transcribed notes
    so MedGemma receives structured symptom context in its prompt.
    """
    symptoms     = symptoms     or []
    risk_factors = risk_factors or []
    parts = []

    symptom_map = {
        "cough_2wk":    "productive cough for more than 2 weeks",
        "fever":        "persistent fever",
        "night_sweats": "night sweats",
        "weight_loss":  "unintentional weight loss",
    }
    rf_map = {
        "hiv_positive": "HIV positive",
        "tb_contact":   "known contact with active TB case",
        "prior_tb":     "history of prior tuberculosis",
    }

    present = [symptom_map[s] for s in symptoms if s in symptom_map]
    if present:
        parts.append("Patient reports " + ", ".join(present) + ".")

    present_rf = [rf_map[r] for r in risk_factors if r in rf_map]
    if present_rf:
        parts.append("Risk factors: " + ", ".join(present_rf) + ".")

    if not parts:
        return ""
    return " ".join(parts)


def compute_combined_risk(who_result, imaging_label,
                          imaging_confidence=None, cough_risk_level=None):
    """Combine WHO symptom score, imaging result, and cough analysis
    into a single overall TB risk assessment.

    Returns dict with overall_risk, overall_score (0-100), narrative.
    """
    # ---- Component scores (normalised to 0–100) ----
    # Clinical symptoms: 0–16 → 0–100
    clinical_norm = (who_result["clinical_score"] / MAX_CLINICAL_SCORE) * 100

    # Imaging: use model confidence directly
    if imaging_confidence is not None:
        conf = min(max(imaging_confidence, 0), 100)
        imaging_norm = conf if imaging_label == "TB" else (100 - conf)
    else:
        imaging_norm = 75 if imaging_label == "TB" else 25

    # Cough (optional): HIGH=90, MODERATE=50, LOW=15, missing=None
    cough_norm = None
    if cough_risk_level:
        cough_map = {"HIGH": 90, "MODERATE": 50, "LOW": 15}
        cough_norm = cough_map.get(cough_risk_level.upper())

    # ---- Weighted combination ----
    # Imaging is the strongest signal (gold standard is CXR), symptoms
    # raise/lower the overall risk, cough is supplementary.
    if cough_norm is not None:
        # With cough: imaging 50%, symptoms 30%, cough 20%
        overall = 0.50 * imaging_norm + 0.30 * clinical_norm + 0.20 * cough_norm
    else:
        # Without cough: imaging 60%, symptoms 40%
        overall = 0.60 * imaging_norm + 0.40 * clinical_norm

    # ---- Risk bucket ----
    if overall >= 65:
        level = "HIGH"
    elif overall >= 40:
        level = "MODERATE"
    elif overall >= 15:
        level = "LOW"
    else:
        level = "MINIMAL"

    # ---- Narrative ----
    components = [f"imaging {'positive' if imaging_label == 'TB' else 'negative'}"]
    if who_result["who_screen_positive"]:
        components.append("WHO symptom screen positive")
    else:
        components.append("WHO symptom screen negative")
    if cough_risk_level:
        components.append(f"cough risk {cough_risk_level.lower()}")

    narrative = (
        f"Overall TB risk: {level} ({overall:.0f}/100). "
        f"Based on {', '.join(components)}. "
    )
    if level == "HIGH":
        narrative += "Recommend immediate sputum testing and clinical evaluation."
    elif level == "MODERATE":
        narrative += "Recommend further investigation (sputum smear/GeneXpert)."
    elif level == "LOW":
        narrative += "Consider follow-up if symptoms persist. Rescreen in 2 weeks."
    else:
        narrative += "No immediate action required. Standard follow-up."

    return {
        "overall_risk":  level,
        "overall_score":  overall,
        "narrative":      narrative,
        "clinical_pct":   clinical_norm,
        "imaging_pct":    imaging_norm,
        "cough_pct":      cough_norm,
    }


# ============================================================
# INFERENCE (Tab 1)
# ============================================================
def analyze_xray(image, age, gender, model_choice, clinical_notes,
                  cough_context, symptoms, risk_factors):
    """Main inference function called by the Analyze button.

    Returns (badge_html, scores_text, response_text,
             combined_risk_html, combined_details).
    """
    empty_result = ("", "", "", "", "")

    if image is None:
        return (
            '<div style="color:#856404;padding:16px;text-align:center;">'
            'Upload a chest X-ray image and click Analyze.</div>',
            "", "", "", "",
        )

    # Check server prerequisites
    missing = []
    for label, p in [("GGUF Model", GGUF_MODEL),
                     ("Vision Projector (mmproj)", MMPROJ),
                     ("llama-server.exe", LLAMA_SERVER)]:
        if not Path(p).exists():
            missing.append(f"  - {label}: {p}")
    if missing:
        detail = "\n".join(missing)
        return (
            '<div style="background:#f8d7da;color:#721c24;padding:16px;'
            'border-radius:8px;">'
            '<b>Server files not found</b><br><pre style="font-size:0.85em;">'
            f'{detail}</pre>'
            'Edit the CONFIGURATION section at the top of <code>app.py</code>.'
            '</div>',
            "", "", "", "",
        )

    use_lora = model_choice == "Base + LoRA"
    logger.info("analyze_xray: model=%s, use_lora=%s, lora_file=%s",
                model_choice, use_lora, find_lora_file())

    # Start / reuse server
    try:
        port = server_manager.ensure_ready(use_lora)
    except Exception as e:
        return (
            f'<div style="background:#f8d7da;color:#721c24;padding:16px;'
            f'border-radius:8px;"><b>Server error:</b> {e}</div>',
            "", "", "", "",
        )

    # Encode image to JPEG data URL
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/jpeg;base64,{b64}"

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    session = requests.Session()

    # ── PASS 1: Classification (5-pass majority vote, evaluation prompt) ──
    classification_prompt = (
        "Is this chest X-ray normal or abnormal? "
        "Answer with: NORMAL or ABNORMAL, then confidence percentage."
    )
    classify_payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": classification_prompt},
            ],
        }],
        "max_tokens": 100,
        "temperature": 1.0,
        "min_p":       0.2,
        "stream": False,
    }

    RUNS = 5
    run_labels = []
    run_confidences = []
    run_raw = []

    try:
        for run_i in range(RUNS):
            r = session.post(url, json=classify_payload, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            lbl, conf = classify_response(txt)
            run_labels.append(lbl)
            run_confidences.append(conf)
            run_raw.append(txt.strip().replace("\n", " ")[:120])
            logger.info("  Classification run %d/%d: %s (conf=%s) — %r",
                        run_i + 1, RUNS, lbl, conf, run_raw[-1])
    except Exception as e:
        return (
            f'<div style="background:#f8d7da;color:#721c24;padding:16px;'
            f'border-radius:8px;"><b>Inference failed:</b> {e}</div>',
            "", "", "", "",
        )

    # Majority vote
    from collections import Counter as _Counter
    vote_counts = _Counter(run_labels)
    label, vote_n = vote_counts.most_common(1)[0]
    logger.info("  Majority vote: %s (%d/%d)  All: %s",
                label, vote_n, RUNS, "/".join(run_labels))

    # Average confidence across runs that match the majority label
    matching_confs = [c for l, c in zip(run_labels, run_confidences)
                      if l == label and c is not None]
    avg_confidence = (sum(matching_confs) / len(matching_confs)
                      if matching_confs else None)

    # ── PASS 2: Explanation (single pass, constrained to match label) ──
    try:
        age_str = str(int(age)) if age else "unknown"
    except (ValueError, TypeError):
        age_str = "unknown"
    gender_str = gender if gender else "unknown"

    if label == "TB":
        explain_verdict = "ABNORMAL — tuberculosis is suspected"
    else:
        explain_verdict = "NORMAL — no signs of tuberculosis"

    explain_prompt = (
        f"This chest X-ray has been classified as: {explain_verdict}.\n"
        f"Patient: {age_str}-year-old {gender_str.lower()}.\n"
    )

    # Add clinical context
    symptom_notes = generate_notes_from_symptoms(symptoms, risk_factors)
    notes = (clinical_notes or "").strip()
    cough = (cough_context or "").strip()
    all_context = "\n".join(
        part for part in [cough, symptom_notes, notes] if part
    )
    if all_context:
        explain_prompt += f"Clinical notes: {all_context}\n"

    explain_prompt += (
        "Describe the radiological findings visible in this image that "
        "support the classification above. Do not contradict the "
        "classification. Be specific about what you observe."
    )

    explain_payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": explain_prompt},
            ],
        }],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }

    try:
        r = session.post(url, json=explain_payload, timeout=120)
        r.raise_for_status()
        response_text = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        response_text = f"(Explanation unavailable: {e})"

    # ── Build UI outputs ──
    if label == "TB":
        badge_html = (
            '<div style="background:#dc3545;color:white;padding:18px;'
            'border-radius:10px;text-align:center;font-size:1.6em;'
            'font-weight:bold;letter-spacing:1px;">TB DETECTED</div>'
        )
    else:
        badge_html = (
            '<div style="background:#28a745;color:white;padding:18px;'
            'border-radius:10px;text-align:center;font-size:1.6em;'
            'font-weight:bold;letter-spacing:1px;">NORMAL</div>'
        )

    conf_str = f"{avg_confidence:.0f}%" if avg_confidence is not None else "N/A"
    scores_text = (
        f"Majority vote: {vote_n}/{RUNS} {label}  |  "
        f"Avg. confidence: {conf_str}"
    )

    # ---- WHO symptom score ----
    who_result = compute_who_symptom_score(symptoms, risk_factors)

    # ---- Parse cough risk level from context string ----
    # Expected format: "HeAR cough pre-screening: HIGH RISK (score: 0.72)"
    cough_risk_level = None
    if cough:
        cough_upper = cough.upper()
        for lvl in ("HIGH", "MODERATE", "LOW"):
            if f"{lvl} RISK" in cough_upper:
                cough_risk_level = lvl
                break

    # ---- Combined risk assessment ----
    combined = compute_combined_risk(
        who_result, label, avg_confidence, cough_risk_level)

    risk_colors = {
        "HIGH":     ("#dc3545", "white"),
        "MODERATE": ("#fd7e14", "white"),
        "LOW":      ("#ffc107", "#333"),
        "MINIMAL":  ("#28a745", "white"),
    }
    bg, fg = risk_colors.get(combined["overall_risk"], ("#6c757d", "white"))

    # Build progress bar segments (explicit dark text + white bg to avoid theme issues)
    _bar_style = (
        'display:flex;align-items:center;margin:6px 0;'
        'color:#212529 !important;'
    )
    _label_style = (
        'width:90px;font-size:0.9em;font-weight:600;'
        'color:#212529 !important;'
    )
    _pct_style = (
        'width:45px;text-align:right;font-size:0.9em;font-weight:600;'
        'color:#212529 !important;'
    )
    bar_segments = []
    bar_segments.append(
        f'<div style="{_bar_style}">'
        f'<span style="{_label_style}">Imaging</span>'
        f'<div style="flex:1;background:#e9ecef;border-radius:4px;height:16px;margin:0 10px;">'
        f'<div style="width:{combined["imaging_pct"]:.0f}%;background:'
        f'{"#dc3545" if combined["imaging_pct"] > 50 else "#28a745"};'
        f'height:100%;border-radius:4px;"></div></div>'
        f'<span style="{_pct_style}">'
        f'{combined["imaging_pct"]:.0f}%</span></div>'
    )
    bar_segments.append(
        f'<div style="{_bar_style}">'
        f'<span style="{_label_style}">Symptoms</span>'
        f'<div style="flex:1;background:#e9ecef;border-radius:4px;height:16px;margin:0 10px;">'
        f'<div style="width:{combined["clinical_pct"]:.0f}%;background:#fd7e14;'
        f'height:100%;border-radius:4px;"></div></div>'
        f'<span style="{_pct_style}">'
        f'{combined["clinical_pct"]:.0f}%</span></div>'
    )
    if combined["cough_pct"] is not None:
        bar_segments.append(
            f'<div style="{_bar_style}">'
            f'<span style="{_label_style}">Cough</span>'
            f'<div style="flex:1;background:#e9ecef;border-radius:4px;'
            f'height:16px;margin:0 10px;">'
            f'<div style="width:{combined["cough_pct"]:.0f}%;'
            f'background:#6f42c1;height:100%;border-radius:4px;"></div></div>'
            f'<span style="{_pct_style}">'
            f'{combined["cough_pct"]:.0f}%</span></div>'
        )

    combined_html = (
        f'<div style="background:{bg};color:{fg};padding:16px;'
        f'border-radius:10px;text-align:center;font-size:1.3em;'
        f'font-weight:bold;letter-spacing:1px;margin-bottom:8px;">'
        f'Overall TB Risk: {combined["overall_risk"]} '
        f'({combined["overall_score"]:.0f}/100)</div>'
        f'<div style="background:#ffffff !important;padding:14px;'
        f'border-radius:8px;margin-bottom:8px;'
        f'border:1px solid #dee2e6;">'
        f'{"".join(bar_segments)}</div>'
        f'<div style="font-size:0.95em;color:#212529 !important;'
        f'padding:8px 4px;line-height:1.5;">'
        f'{combined["narrative"]}</div>'
    )

    who_screen_label = "POSITIVE" if who_result["who_screen_positive"] else "NEGATIVE"
    combined_details = (
        f"WHO 4-symptom screen: {who_screen_label}\n"
        f"{who_result['details']}"
    )

    return badge_html, scores_text, response_text, combined_html, combined_details


def format_who_score(symptoms, risk_factors):
    """Format WHO score for display."""
    result = compute_who_symptom_score(symptoms, risk_factors)
    if result["clinical_score"] == 0:
        return "No symptoms selected"
    screen = "POSITIVE" if result["who_screen_positive"] else "NEGATIVE"
    return f"WHO screen: {screen} | {result['details']}"


# ============================================================
# CUSTOM CSS
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
.header-banner {
    background: linear-gradient(135deg, #0d7377 0%, #14919b 100%);
    color: white; padding: 24px; border-radius: 12px;
    margin-bottom: 8px; text-align: center;
}
.header-banner h1 { margin: 0 0 4px 0; font-size: 1.8em; }
.header-banner p  { margin: 0; opacity: 0.9; }
.disclaimer-box {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 8px; padding: 12px 16px; margin-top: 12px;
    color: #856404; font-size: 0.9em;
}
"""


# ============================================================
# BUILD GRADIO APP
# ============================================================
def build_app():
    with gr.Blocks(css=CUSTOM_CSS, title="FieldScreen AI — TB Screening") as demo:

        gr.HTML(
            '<div class="header-banner">'
            '<h1>FieldScreen AI — TB Screening Demo</h1>'
            '<p>MedGemma 4B + MedASR + HeAR + WHO Scoring + '
            'TranslateGemma: Multi-model pipeline for tuberculosis '
            'detection in chest X-rays</p></div>'
        )

        with gr.Tabs():
            # ── Tab 1: Clinical Workflow ────────────────────────
            with gr.Tab("Clinical Workflow"):
                gr.HTML(
                    '<div style="background:linear-gradient(135deg,'
                    '#1a5276 0%,#2e86c1 100%);color:white;padding:18px;'
                    'border-radius:10px;text-align:center;margin-bottom:'
                    '12px;"><h2 style="margin:0;">Guided TB Screening '
                    'Pipeline</h2><p style="margin:4px 0 0;opacity:0.9;">'
                    'Follow the steps below — cough, voice, symptoms, '
                    'X-ray, report</p></div>'
                )

                with gr.Row():
                    # ── Left: inputs (steps 1-4) ──
                    with gr.Column(scale=1):

                        # Step 1: Patient Info + WHO Symptoms
                        with gr.Accordion(
                            "Step 1 — Patient Info & WHO Symptom Screen",
                            open=True,
                        ):
                            with gr.Row():
                                wf_age = gr.Number(
                                    label="Age", precision=0)
                                wf_gender = gr.Dropdown(
                                    ["Male", "Female", "Unknown"],
                                    label="Gender", value="Unknown",
                                )
                            wf_symptoms = gr.CheckboxGroup(
                                choices=[
                                    ("Cough \u2265 2 weeks", "cough_2wk"),
                                    ("Fever", "fever"),
                                    ("Night sweats", "night_sweats"),
                                    ("Weight loss", "weight_loss"),
                                ],
                                label="WHO 4-Symptom Screen",
                                value=[],
                            )
                            wf_risk_factors = gr.CheckboxGroup(
                                choices=[
                                    ("HIV positive", "hiv_positive"),
                                    ("Known TB contact", "tb_contact"),
                                    ("Prior TB history", "prior_tb"),
                                ],
                                label="Risk Factors",
                                value=[],
                            )
                            wf_who_display = gr.Textbox(
                                label="Clinical Score",
                                interactive=False,
                                value="No symptoms selected",
                            )

                        # Step 2: Cough (HeAR)
                        with gr.Accordion(
                            "Step 2 — Record Cough (HeAR)",
                            open=True,
                        ):
                            wf_cough_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Cough Audio (2+ seconds)",
                            )
                            wf_cough_btn = gr.Button(
                                "Analyze Cough", variant="secondary",
                                size="sm",
                            )
                            wf_cough_result = gr.HTML(
                                value=(
                                    '<div style="color:#6c757d;'
                                    'padding:6px;text-align:center;">'
                                    'Waiting for cough audio...</div>'
                                ),
                            )
                            wf_cough_details = gr.Textbox(
                                label="HeAR Details",
                                interactive=False, lines=1,
                            )
                            wf_cough_summary = gr.Textbox(
                                visible=False,
                            )

                        # Step 3: Voice (MedASR)
                        with gr.Accordion(
                            "Step 3 — Speak Symptoms (MedASR)",
                            open=True,
                        ):
                            wf_voice_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Voice Input",
                            )
                            wf_transcribe_btn = gr.Button(
                                "Transcribe", variant="secondary",
                                size="sm",
                            )
                            wf_clinical_notes = gr.Textbox(
                                label="Clinical Notes",
                                placeholder="Transcribed or typed...",
                                lines=2, interactive=True,
                            )

                        # Step 4: X-ray (MedGemma)
                        with gr.Accordion(
                            "Step 4 — Upload Chest X-ray",
                            open=True,
                        ):
                            wf_image = gr.Image(
                                type="pil",
                                label="Chest X-ray",
                                height=250,
                            )
                            wf_model = gr.Radio(
                                ["Base Model", "Base + LoRA"],
                                label="Model", value="Base + LoRA",
                            )
                            wf_analyze_btn = gr.Button(
                                "Run Full Screening",
                                variant="primary", size="lg",
                            )

                    # ── Right: results (step 5) ──
                    with gr.Column(scale=1):

                        with gr.Accordion(
                            "Step 5 — Combined Results & Report",
                            open=True,
                        ):
                            gr.Markdown("##### Overall TB Risk")
                            wf_combined_html = gr.HTML(
                                value=(
                                    '<div style="color:#6c757d;'
                                    'padding:20px;text-align:center;'
                                    'background:#f8f9fa;border-radius:'
                                    '10px;">Complete steps 1-4 and '
                                    'click Run Full Screening</div>'
                                ),
                            )
                            wf_combined_details = gr.Textbox(
                                label="WHO Scoring",
                                interactive=False, lines=2,
                            )
                            gr.Markdown("##### Imaging Result")
                            wf_badge = gr.HTML(
                                value=(
                                    '<div style="color:#6c757d;'
                                    'padding:12px;text-align:center;">'
                                    'Waiting for X-ray analysis</div>'
                                ),
                            )
                            wf_scores = gr.Textbox(
                                label="Classification",
                                interactive=False,
                            )
                            wf_response = gr.Markdown(
                                value="*Upload an X-ray to generate the report.*",
                                label="MedGemma Report",
                            )

                            gr.Markdown("##### Translate Report")
                            with gr.Row():
                                wf_translate_lang = gr.Dropdown(
                                    choices=list(
                                        TRANSLATE_LANGUAGES.keys()),
                                    value="Hindi",
                                    label="Language", scale=2,
                                )
                                wf_translate_btn = gr.Button(
                                    "Translate",
                                    variant="secondary", size="sm",
                                    scale=1,
                                )
                            wf_translate_out = gr.HTML(
                                value=(
                                    '<div style="color:#6c757d;'
                                    'padding:6px;text-align:center;'
                                    'font-size:0.9em;">'
                                    'Run screening first</div>'
                                ),
                            )
                            wf_translate_status = gr.Textbox(
                                label="TranslateGemma Status",
                                interactive=False,
                                value=translate_manager.status_message,
                            )

                # ── Workflow callbacks ──

                wf_symptoms.change(
                    fn=format_who_score,
                    inputs=[wf_symptoms, wf_risk_factors],
                    outputs=[wf_who_display],
                )
                wf_risk_factors.change(
                    fn=format_who_score,
                    inputs=[wf_symptoms, wf_risk_factors],
                    outputs=[wf_who_display],
                )

                def _wf_cough(audio_path):
                    badge, details, summary = analyze_cough_audio(
                        audio_path)
                    return badge, details, summary

                wf_cough_btn.click(
                    fn=_wf_cough,
                    inputs=[wf_cough_audio],
                    outputs=[wf_cough_result, wf_cough_details,
                             wf_cough_summary],
                )

                def _wf_transcribe(audio_path):
                    if audio_path is None:
                        return gr.update()
                    return transcribe_audio(audio_path)

                wf_transcribe_btn.click(
                    fn=_wf_transcribe,
                    inputs=[wf_voice_audio],
                    outputs=[wf_clinical_notes],
                )

                wf_analyze_btn.click(
                    fn=analyze_xray,
                    inputs=[wf_image, wf_age, wf_gender,
                            wf_model, wf_clinical_notes,
                            wf_cough_summary,
                            wf_symptoms, wf_risk_factors],
                    outputs=[wf_badge, wf_scores, wf_response,
                             wf_combined_html,
                             wf_combined_details],
                )

                wf_translate_btn.click(
                    fn=translate_report,
                    inputs=[wf_response, wf_translate_lang],
                    outputs=[wf_translate_out,
                             wf_translate_status],
                )

                gr.HTML(
                    '<div class="disclaimer-box">'
                    '<strong>Medical Disclaimer:</strong> This tool is '
                    'for research and educational purposes only. It is '
                    'NOT a medical device and must not be used for '
                    'clinical diagnosis.</div>'
                )

            # ── Tab 3: Reports ─────────────────────────────────
            with gr.Tab("Reports"):
                gr.HTML(
                    '<div style="background:linear-gradient(135deg,'
                    '#5b2c6f 0%,#8e44ad 100%);color:white;padding:18px;'
                    'border-radius:10px;text-align:center;margin-bottom:'
                    '12px;"><h2 style="margin:0;">Screening Reports'
                    '</h2><p style="margin:4px 0 0;opacity:0.9;">'
                    'Export results for clinical records and health '
                    'information systems</p></div>'
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### PDF Report Export\n\n"
                            "Generate a printable PDF report containing:\n"
                            "- Patient demographics\n"
                            "- Chest X-ray image\n"
                            "- MedGemma findings\n"
                            "- WHO symptom score and risk factors\n"
                            "- HeAR cough analysis results\n"
                            "- Combined TB risk assessment\n"
                            "- Clinical recommendations\n\n"
                            "*Coming in next update — run a screening "
                            "from Tab 1 or Tab 2 first, then export "
                            "here.*"
                        )
                        gr.Button(
                            "Export PDF Report",
                            variant="secondary",
                            interactive=False,
                        )

                    with gr.Column():
                        gr.Markdown(
                            "### FHIR Export\n\n"
                            "Generate HL7 FHIR-compliant resources:\n"
                            "- **DiagnosticReport** — Screening findings\n"
                            "- **Observation** — TB risk score\n"
                            "- **Patient** — Demographics\n\n"
                            "FHIR JSON output can be imported directly "
                            "by electronic health record systems (OpenMRS, "
                            "DHIS2, Google Cloud Healthcare API).\n\n"
                            "*Coming in next update.*"
                        )
                        gr.Button(
                            "Export FHIR JSON",
                            variant="secondary",
                            interactive=False,
                        )

                gr.Markdown(
                    "---\n### Screening History\n\n"
                    "Previous screening results from this session will "
                    "appear here. *Coming in next update.*"
                )
                gr.DataFrame(
                    value=pd.DataFrame(
                        columns=["Time", "Patient", "Imaging",
                                 "WHO Score", "Cough", "Overall Risk"],
                    ),
                    label="Session History",
                    interactive=False,
                )

            # ── Tab 6: About ────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown("""
## Project Overview

**FieldScreen AI** is a proof-of-concept screening platform for
automated tuberculosis detection using Google's **HAI-DEF** (Health AI
Developer Foundations) model family. It combines multiple HAI-DEF models
into a clinically coherent pipeline:

| Model | Role | Runs on |
|---|---|---|
| **MedGemma 1.5 4B-It** | CXR analysis, fine-tuned with LoRA | GPU (~4 GB VRAM) |
| **MedASR** (105M) | Medical speech recognition (5.2% WER) | CPU |
| **HeAR** (ViT-L) | Cough-based TB pre-screening (512-dim) | CPU |
| **TranslateGemma 4B** | Report translation (15 languages) | CPU |
| **WHO 4-Symptom Screen** | Clinical scoring (symptoms + risk factors) | N/A |

The system combines all signals (imaging + symptoms + cough) into a
unified risk assessment with multilingual output, transforming a simple
image classifier into a clinically coherent screening tool. Designed
for community health workers (CHWs) in resource-limited settings,
running entirely offline on consumer hardware.

---

## Architecture

```
  Cough Audio          Voice Input          WHO Symptom Form
  (patient coughs)     (CHW speaks)         (structured checkboxes)
       |                    |                      |
       v                    v                      v
  HeAR (ViT-L)        MedASR (105M, CPU)    WHO Clinical Scoring
  512-dim embeddings   speech-to-text        4-symptom + risk factors
       |                    |                      |
       v                    v                      v
  Cough Risk Score     Clinical Context      Symptom Score (0-16)
       \\                   |                     /
        \\─────────────── + ─────────────────────/
                           |
  Chest X-ray ────────────>+
                           |
                           v
  llama-server (MedGemma GGUF + mmproj vision encoder)
       |  LoRA adapter (fieldscreen-tb-image-lora)
       v
  Natural Language Radiology Assessment
       |
       v
  Keyword Classifier (TB vs Normal)
       |
       v
  Combined Risk Engine
  (imaging 50-60% + symptoms 30-40% + cough 20%)
       |
       v
  Overall TB Risk Level + Clinical Recommendation
       |
       v (on demand)
  TranslateGemma (4B, GGUF, CPU)
       |
       v
  Translated Report (15 languages: Hindi, French, Spanish, ...)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base Model | MedGemma 1.5 4B-It |
| Quantization | Q4_K_M (GGUF) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| LoRA Dropout | 0.05 |
| Learning Rate | 2e-4 |
| Epochs | 5 |
| Batch Size | 1 (gradient accumulation: 4) |
| Training Data | Montgomery County CXR (~138) + Tawsifurrahman (~1,200 balanced) |
| Train/Eval Split | 90% / 10% |
| Precision | bfloat16 (vision encoder unquantized) |
| Context Size | 4096 tokens |

---

## Technical Challenges Solved

1. **Vision Encoder Quantization Crash** —
   4-bit quantization via BitsAndBytes triggered `CUDA device-side
   assert` inside the vision tower. Resolved by adding `vision_tower`
   to `llm_int8_skip_modules`, keeping the SigLIP vision encoder in
   full bfloat16 while quantizing only the Gemma language backbone.

2. **PaliGemma-to-GGUF LoRA Conversion** —
   The standard `convert_lora_to_gguf.py` failed due to `gguf`
   library version mismatches and PaliGemma's non-standard tensor
   naming (`model.language_model.model.layers...` vs llama.cpp's
   `blk.N...`). Built a custom converter (`simple_lora_converter.py`)
   that remaps tensor names and injects the required
   `general.architecture=gemma3` metadata for llama-server
   compatibility.

3. **Stable Multi-Modal Inference** —
   `llama-cpp-python` bindings were unstable with vision models
   (segfaults on repeated calls, inconsistent image encoding).
   Migrated to llama-server's HTTP API (OpenAI-compatible `/v1/chat/completions`),
   which provides reliable vision+text inference with base64-encoded
   images and proper multi-turn context handling.

4. **Offline-First Multi-Model Orchestration** —
   Four independent models (MedGemma GPU + MedASR CPU + HeAR CPU +
   TranslateGemma CPU) must coexist on 12 GB VRAM + system RAM.
   Implemented lazy loading with thread-safe managers, offline-first
   HuggingFace loading (`HF_HUB_OFFLINE=1`), and VRAM isolation
   (TranslateGemma runs CPU-only to avoid starving MedGemma).

5. **Multi-Dataset LoRA Training** —
   Combined the small Montgomery dataset (~138 images) with the
   larger Tawsifurrahman dataset (~4,200 images, balanced to ~1,200)
   while maintaining a clean evaluation split. Independent eval
   images from Tawsifurrahman are excluded from training via
   deterministic sampling with fixed random seeds.

---

## Current Results (GGUF Evaluation)

Evaluated on 200 independent test images (Tawsifurrahman dataset),
5 runs per image with majority voting, through llama-server HTTP API.

| Metric | Base Model | Base + LoRA | Change |
|---|---|---|---|
| **Accuracy** | 84.0% | **86.0%** | +2.0 pp |
| Sensitivity | 73.0% | 75.0% | +2.0 pp |
| Specificity | 95.0% | 97.0% | +2.0 pp |
| PPV | 93.6% | 96.2% | +2.6 pp |
| NPV | 77.9% | 79.5% | +1.6 pp |
| F1 Score | 0.820 | 0.843 | +0.023 |
| False Positives | 5 | 3 | -40% |

The LoRA adapter improves every metric. The gain in specificity
(95% to 97%) reduces false positive referrals by 40%.

---

## Future Roadmap

| Feature | Description |
|---|---|
| PDF / FHIR export | Printable reports and structured health data output |
| Longitudinal CXR | Compare current vs prior X-ray (MedGemma multi-image) |
| Anatomical overlays | Bounding boxes showing detected abnormalities |
| Batch screening | Queue-based processing for screening camps (50-200/day) |
| Cough classifier | Train HeAR classifier on labeled TB cough datasets |
""")

    return demo


# ============================================================
# BACKGROUND MODEL PRELOADING
# ============================================================
def _preload_models():
    """Pre-start models in background threads at app launch.

    MedGemma (llama-server) takes the longest to load; MedASR and HeAR
    are smaller and CPU-only.  All run in daemon threads so they don't
    block startup but are ready by the time the user submits a query.

    NOTE: transformers is imported here in the main thread first to avoid
    a race condition when HeAR and MedASR both try to do their first
    import of transformers in parallel background threads.
    """
    try:
        import transformers  # noqa: F401 — warm up module before threads
    except ImportError:
        pass

    def _start_medgemma():
        try:
            server_manager.ensure_ready(use_lora=True)
            logger.info("Pre-started MedGemma server (LoRA mode).")
        except Exception as e:
            logger.warning(f"MedGemma pre-start failed (will retry on use): {e}")

    def _start_medasr():
        try:
            medasr_manager.load()
        except Exception as e:
            logger.warning(f"MedASR pre-load failed (will retry on use): {e}")

    def _start_hear():
        try:
            hear_manager.load()
        except Exception as e:
            logger.warning(f"HeAR pre-load failed (will retry on use): {e}")

    for fn in (_start_medgemma, _start_medasr, _start_hear):
        threading.Thread(target=fn, daemon=True).start()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    _preload_models()
    demo = build_app()
    demo.queue()          # process events in background threads (prevents UI freeze)
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        inbrowser=True,
    )

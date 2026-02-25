# FieldScreen AI — Offline Multi-Model TB Screening

<div align="center">

**Chest X-ray analysis • Cough pre-screening • Medical speech recognition • 15 languages**

*Zero-cost, fully offline TB screening for community health workers in resource-limited settings*

</div>

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Marc-Dvci/FieldScreen_AI.git
cd fieldscreen-ai

# 2. Download models (~8 GB total)
python download_models.py

# 3. Set up environment and launch (Windows)
SETUP.bat          # first time only: creates venv + installs deps
START_APP.bat      # launches the app on http://127.0.0.1:7860
```

> **Hardware**: NVIDIA GPU with ≥4 GB VRAM (e.g. GTX 1650), 8 GB RAM, Windows 10/11.
> No internet required after initial setup.

---

## What is FieldScreen AI?

FieldScreen AI is a fully offline TB screening tool that integrates **three Google Health AI Developer Foundations (HAI-DEF) models** into a single clinical workflow:

| Model | Role | Runs on |
|-------|------|---------|
| **MedGemma 1.5 4B-It** | Chest X-ray interpretation (LoRA fine-tuned) | GPU (~4 GB VRAM) |
| **HeAR** (ViT-L, 512-dim) | Cough-based TB pre-screening | CPU |
| **MedASR** (105M params) | Medical speech-to-text (5.2% WER) | CPU |
| **TranslateGemma 4B** | Report translation (15 languages) | CPU |

Plus a **WHO-aligned 4-symptom scoring** module for structured clinical assessment.

### Architecture

```
  Cough Audio          Voice Input          WHO Symptom Form
       │                    │                      │
       ▼                    ▼                      ▼
  HeAR (ViT-L)        MedASR (105M)         WHO Scoring
  512-dim embeddings   speech-to-text        4-symptom + risk
       │                    │                      │
       ▼                    ▼                      ▼
  Cough Risk Score     Clinical Context      Symptom Score
       └────────────── + ─────────────────────────┘
                           │
  Chest X-ray ────────────►+
                           │
                           ▼
  MedGemma 4B (GGUF + LoRA) → Radiology Assessment
                           │
                           ▼
  Combined Risk Engine (imaging 50% + symptoms 30% + cough 20%)
                           │
                           ▼
  TranslateGemma → Report in 15 languages
```

---

## Evaluation Results

Evaluated on 200 independent test images (Tawsifurrahman TB Chest Radiography Database), 5 runs per image with majority voting:

| Metric | Base Model | + LoRA Fine-Tuned | Change |
|--------|-----------|-------------------|--------|
| **Accuracy** | 84.0% | **86.0%** | +2.0 pp |
| **Sensitivity** | 73.0% | 75.0% | +2.0 pp |
| **Specificity** | 95.0% | **97.0%** | +2.0 pp |
| **F1 Score** | 0.820 | **0.843** | +0.023 |
| **False Positives** | 5 | **3** | -40% |

The LoRA adapter reduces false positive referrals by 40% — critical in settings where confirmatory diagnostics are scarce.

---

## Model Downloads

The `download_models.py` script automates downloading. Or download manually:

| Model | Source | Size |
|-------|--------|------|
| MedGemma GGUF | [unsloth/medgemma-1.5-4b-it-GGUF-Q4_K_M](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF) | 2.5 GB |
| Vision Projector | [unsloth/gemma-3-4b-it-GGUF](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) (mmproj-BF16.gguf) | 850 MB |
| HeAR | [google/hear-pytorch](https://huggingface.co/google/hear-pytorch) | 1.2 GB |
| MedASR | [google/medasr](https://huggingface.co/google/medasr) | 420 MB |
| TranslateGemma | [google/translategemma-4b-it-GGUF-Q4_K_M](https://huggingface.co/google/translategemma-4b-it-GGUF) | 2.5 GB |
| llama-server | [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) | 1.8 GB |

The fine-tuned **LoRA adapter** (`image-lora.gguf`, 114 MB) is included in this repository under `Models/MedGemma/` via Git LFS.

---

## Repository Structure

```
├── README.md                         # This file
├── app.py                            # Main Gradio application (2100+ lines)
├── requirements.txt                  # Python dependencies
├── download_models.py                # Automated model downloader
│
├── SETUP.bat / setup.sh              # Environment setup (Windows / Linux)
├── START_APP.bat                     # Launch app (Windows)
│
├── training/                         # Training & evaluation code
│   ├── train_image_lora.py           # LoRA fine-tuning (vision encoder fix)
│   ├── train_cough_classifier.py     # HeAR cough classifier
│   ├── prepare_coswara_data.py       # Coswara dataset preparation
│   ├── evaluate_gguf.py              # GGUF evaluation pipeline
│   ├── simple_lora_converter.py      # HF→GGUF LoRA converter
│   ├── HEAR_TRAINING_REPORT.md       # HeAR training details
│   └── data_note.md                  # Dataset documentation
│
├── evaluation/                       # Pre-computed results
│   ├── evaluation_report_gguf.txt    # Full evaluation output
│   └── evaluation_results_gguf.json  # Per-image results
│
└── Models/                           # Model directory (mostly downloaded)
    ├── HeAR/
    │   ├── config.json               # ViT-L config (included)
    │   └── tb_cough_classifier.npz   # Trained classifier (included, 3 KB)
    ├── MedGemma/
    │   └── image-lora.gguf           # LoRA adapter (Git LFS, 114 MB)
    ├── MedASR/                       # Downloaded by setup
    └── TranslateGemma/               # Downloaded by setup

```

---

## Technical Highlights

1. **Vision Encoder Quantization Fix** — Resolved CUDA crashes by skipping 4-bit quantization for the SigLIP vision tower (`llm_int8_skip_modules=["vision_tower"]`).

2. **Custom GGUF LoRA Converter** — Built `simple_lora_converter.py` to handle PaliGemma's non-standard tensor naming for llama.cpp compatibility.

3. **5-Pass Majority Voting** — Each X-ray is classified 5 times with different sampling; majority vote ensures robust predictions.

4. **Multi-Model Orchestration** — Four models coexist on 4 GB VRAM and 8 GB RAM via lazy loading, CPU offloading, and thread-safe managers.

5. **HeAR Cough Classifier** — Trained on 2,150 Coswara cough recordings; see [HEAR_TRAINING_REPORT.md](training/HEAR_TRAINING_REPORT.md) for details.

---

## License

This project is built on Google's open-weight HAI-DEF models. The application code is released under the MIT License.

---

*Built for the [MedGemma Impact Challenge](https://ai.google.dev/gemma/docs/medgemma). Designed to bring TB screening where radiologists cannot go.*

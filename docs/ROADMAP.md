# FieldScreen AI — Development Roadmap

## Current State
- MedGemma fine-tuned with LoRA on Montgomery dataset (32.1% accuracy)
- GGUF quantized for edge deployment (Q4_K_M)
- Evaluation pipeline via llama-server HTTP API
- 4-tab Gradio demo app (Screening, Eval Dashboard, Dataset Explorer, About)

---

## Additions — Priority Order

### 1. MedASR Voice Input (PRIORITY: CRITICAL)
- **What**: Voice recording widget in Gradio, transcribed by MedASR (105M params)
- **Why**: 2nd HAI-DEF model. CHWs speak, don't type.
- **How**: `transformers` pipeline("automatic-speech-recognition", model="google/medasr").
  16kHz mono audio. CPU-viable. Transcribed text feeds into MedGemma as clinical context.
- **Demo narrative**: CHW speaks "Patient reports three weeks productive cough, night sweats,
  weight loss, HIV positive, known TB contact" while preparing the X-ray.
- **Status**: DONE

### 2. HeAR Cough Analysis (PRIORITY: HIGH)
- **What**: Record patient's cough, extract HeAR embeddings, classify TB risk
- **Why**: 3rd HAI-DEF model. Cough-based TB pre-screening is WHO-evaluated.
  HeAR achieves ~90% TB detection accuracy in published research.
- **How**: `google/hear-pytorch` on HuggingFace. ViT-L model. Takes 2-second 16kHz clips,
  outputs 512-dim embeddings. Train a small logistic regression classifier on TB cough
  datasets, or demonstrate the embedding pipeline with similarity scoring.
- **Demo narrative**: Before X-ray, CHW records cough. HeAR flags high-risk in 2 seconds.
- **Status**: DONE

### 3. WHO-Aligned Clinical Symptom Scoring (PRIORITY: HIGH)
- **What**: Structured symptom questionnaire (WHO 4-symptom screen: cough >2 weeks,
  fever, night sweats, weight loss) + risk factors (HIV, TB contact, prior TB).
  Combined score: symptoms + imaging → overall TB risk level.
- **Why**: Transforms "image classifier" into "clinical screening tool."
  Addresses product feasibility (20%) and real-world impact (15%).
- **How**: Scored form in Gradio. Risk logic combining symptom score + MedGemma output.
  Symptom text auto-generated from form OR transcribed from MedASR voice input.
- **Status**: DONE

### 4. TranslateGemma Multilingual Reports (PRIORITY: MEDIUM-HIGH)
- **What**: Generate screening reports in patient's language (Hindi, French, Spanish,
  Arabic, Swahili — major TB-burden regions). TranslateGemma 4B for edge deployment.
- **Why**: Multilingual medical output for LMIC deployment.
  Directly addresses LMIC deployment narrative.
- **How**: `google/translategemma-4b-it` gguf via llama.cpp pipeline. 55 languages supported.
  Translate the final report on demand. Load/unload to manage VRAM.
- **Note**: TranslateGemma is Google but not strictly HAI-DEF. Adds to feasibility/impact.
- **Status**: DONE

### 5. PDF Report Export (PRIORITY: MEDIUM)
- **What**: Printable PDF with X-ray image, findings, risk score, next steps, patient info.
- **Why**: In dire environments, you print results. Essential for real-world use.
  No electronic health system in remote screening camps.
- **How**: `reportlab` or HTML-to-PDF. Auto-generated from screening results.


### 6. FHIR Export (PRIORITY: MEDIUM)
- **What**: Generate FHIR-compliant DiagnosticReport + Observation JSON resources.
- **Why**: Standard interoperability format for health information systems.
- **How**: `fhir.resources` library (Pydantic-based). DiagnosticReport wrapping the
  screening result, Observation for TB risk score, Patient resource for demographics.


### 7. Longitudinal CXR Comparison (PRIORITY: MEDIUM)
- **What**: Upload current + prior X-ray. MedGemma 1.5 compares for progression/stability.
- **Why**: Key MedGemma 1.5 feature for treatment monitoring.
  This is how TB treatment monitoring actually works.
- **How**: Send both images in the prompt. MedGemma 1.5 natively supports this.
  Prompt: "Compare current and prior chest X-ray for disease progression or improvement."


### 8. Anatomical Bounding Boxes (PRIORITY: LOW-MEDIUM)
- **What**: Draw boxes on X-ray showing where model found abnormalities.
- **Why**: Visually impressive in demo video. Showcases MedGemma 1.5 capability.
- **How**: MedGemma 1.5 outputs `[y0, x0, y1, x1]` normalized to [0, 1000].
  Parse coordinates, overlay rectangles on image with PIL/matplotlib.


### 9. Batch Screening Mode (PRIORITY: LOW)
- **What**: Upload folder of X-rays, process queue, generate batch report.
- **Why**: Realistic for TB screening camps (50-200 patients/day).
- **How**: Queue UI in Gradio, progress bar, batch PDF report.


### 10. Offline System Dashboard (PRIORITY: LOW)
- **What**: Show GPU/memory usage, model load status, "No Internet Required" indicator.
- **Why**: Proves edge deployment claim visually.
- **How**: `psutil` + `GPUtil` for system stats. Status bar in Gradio.


---

## Updated App Architecture — IMPLEMENTED

```
Tab 1: TB Screening (Live Demo)
  - Image upload + WHO symptom form + voice (MedASR) + cough (HeAR)
  - Combined risk score (imaging + symptoms + cough)
  - Multilingual report translation (TranslateGemma)

Tab 2: Clinical Workflow (flagship for video) — DONE
  - Step-by-step guided pipeline with Accordion sections
  - Step 1: Patient info + WHO 4-symptom screen
  - Step 2: Record cough → HeAR analysis
  - Step 3: Speak symptoms → MedASR transcription
  - Step 4: Upload X-ray → MedGemma analysis
  - Step 5: Combined results + translation

Tab 3: Evaluation Dashboard
  - Pre-computed results from evaluation_results_gguf.json

Tab 4: Dataset Explorer
  - Montgomery + Tawsifurrahman browser

Tab 5: Reports (structure ready, export pending)
  - PDF export placeholder
  - FHIR export placeholder
  - Screening history table

Tab 6: About
  - Multi-model architecture diagram
  - All HAI-DEF + Google models used
  - Offline-first model loading
```

---

## Key Technical Details

### MedASR
- Model: `google/medasr` (HuggingFace)
- Size: 105M parameters
- Audio: 16kHz mono, int16 waveform
- API: `pipeline("automatic-speech-recognition", model="google/medasr")`
- Requires: transformers >= 5.0.0 (install from git)
- Language: English only
- Performance: 5.2% WER (vs Whisper 28.2%)

### HeAR
- Model: `google/hear-pytorch` (HuggingFace)
- Architecture: ViT-L Masked Auto Encoder
- Audio: 16kHz mono, 2-second clips
- Output: 512-dim embedding vector
- Needs: classifier head trained on TB cough data, or similarity-based scoring

### TranslateGemma
- Model: `google/translategemma-4b-it` (edge) or `27b-it` (quality)
- Languages: 55 (Hindi, French, Spanish, Arabic, etc.)
- API: transformers pipeline with source_lang_code / target_lang_code
- Max input: 2K tokens

### FHIR
- Library: `fhir.resources` (pip install)
- Resources: DiagnosticReport, Observation, Patient
- Output: JSON, directly importable by health information systems

---

## Video Script Outline (3 minutes)

0:00-0:30 — Problem: "1.6M TB deaths/year. 3.4M undiagnosed. 60 radiologists
             for 58M people in Tanzania."
0:30-1:00 — Setup: Portable X-ray + laptop at screening camp. "FieldScreen AI
             runs entirely offline on consumer hardware."
1:00-1:45 — Live demo: Record cough (HeAR pre-screen) → Speak symptoms (MedASR
             transcribes) → Upload X-ray (MedGemma analyzes) → Report generated
             in Hindi (TranslateGemma)
1:45-2:15 — Results: Fine-tuning accuracy table. 3-model pipeline diagram.
             "Uses 3 Google HAI-DEF models in a clinically coherent pipeline."
2:15-2:45 — Impact: "$0 per read vs $2-5 commercial. Open-weight, free for any
             health ministry or NGO. Runs offline. No data leaves the device."
2:45-3:00 — Close: "Every case caught prevents 10-15 infections. FieldScreen AI
             catches them." Logo + links.



# HeAR Cough Analysis — Training Report

## Overview

The HeAR (Health Acoustic Representations) module in FieldScreen AI provides
cough-based TB pre-screening by analyzing audio recordings. A logistic regression
classifier was trained on HeAR embeddings to distinguish respiratory illness
coughs from healthy coughs.

## Model Architecture

```
Audio (WAV) → Resample 16 kHz → 2-second chunks
    → Mel-PCEN Spectrogram (192×128)
    → HeAR ViT Encoder (24 layers, 1024 hidden, 16 heads)
    → 512-dim embedding (mean-pooled over chunks)
    → Logistic Regression → TB risk score [0–1]
```

- **Base model**: `google/hear-pytorch` — a Vision Transformer (ViT-Large)
  pre-trained on health-related audio
- **Classifier**: Scikit-learn LogisticRegression with `class_weight="balanced"`
- **Output**: `Models/HeAR/tb_cough_classifier.npz` (weights, bias, metadata)

## Training Data

### Dataset Used: Coswara Heavy Cough

- **Source**: [Kaggle — Coswara dataset](https://www.kaggle.com/datasets/sarabhian/coswara-dataset-heavy-cough)
- **Origin**: Indian Institute of Science (IISc), Bangalore
- **Content**: Forced cough recordings from 2,313 individuals with COVID-19 and
  respiratory illness metadata

#### Label Mapping

| Original Status (`covid_status`) | Our Label | Count |
|----------------------------------|-----------|-------|
| `positive_mild`                  | Positive  | 325   |
| `positive_moderate`              | Positive  | 127   |
| `resp_illness_not_identified`    | Positive  | 153   |
| `healthy`                        | Normal    | 1,404 |
| `no_resp_illness_exposed`        | Normal    | 172   |
| `recovered_full`                 | Excluded  | 103   |
| `positive_asymp`                 | Excluded  | 70    |

**Final split**: 574 positive, 1,576 normal (2,150 total)

### Results

| Metric         | Value    |
|----------------|----------|
| Test Accuracy  | 58.8%    |
| Train/Test     | 80% / 20% |
| Embedding Dim  | 512      |
| Classifier     | LogisticRegression (balanced) |

## Limitations

1. **Not TB-specific**: The Coswara dataset captures COVID-19 / general respiratory
   illness, not tuberculosis. The classifier functions as a *respiratory illness
   cough detector*, not a true TB classifier.

2. **Moderate accuracy**: 58.8% accuracy reflects the difficulty of the task with
   a proxy dataset. With TB-specific data, substantially higher accuracy is expected.

3. **Class imbalance**: 574 positive vs 1,576 normal (1:2.7 ratio), partially
   mitigated by `class_weight="balanced"`.

## Ideal Dataset: CODA TB

The **CODA TB dataset** (Sharma et al., *Science Advances*, 2024) would be the
optimal training resource:

- **Scale**: 700,000+ cough sounds from 2,143 individuals
- **Geography**: 7 countries (South Africa, India, Madagascar, etc.)
- **Labels**: Confirmed TB status via GeneXpert/sputum culture
- **Annotations**: Cough events individually annotated with quality scores
- **Access**: Available via [CODA TB website](https://coda-tb.github.io/) with
  data access request

**Citation:**
> Sharma, M., Kamble, M. R., Sinha, S., et al. (2024). "CODA TB: A large-scale
> cough dataset for the acoustic diagnosis of tuberculosis." *Science Advances*,
> 10(32), eadj0349. DOI: 10.1126/sciadv.adj0349

## Next Steps

### With CODA TB Dataset (Priority)
1. **Request access** from the CODA TB consortium
2. Re-run `train_cough_classifier.py` with CODA TB audio organized in
   `data/tb/` and `data/normal/`
3. Expected: significantly higher accuracy (literature reports AUC > 0.70
   for cough-based TB screening)

### Model Improvements
1. **Fine-tune HeAR** instead of training only a linear classifier
2. **Data augmentation**: pitch shifting, time stretching, noise injection
3. **Ensemble**: combine cough analysis with symptom questionnaire scores
4. **Threshold tuning**: optimize sensitivity/specificity trade-off for
   screening (high sensitivity preferred)

## Reproducing Training

```bash
# 1. Prepare data (from Coswara dataset)
python training_data/prepare_coswara_data.py

# 2. Train classifier
python training_data/train_cough_classifier.py \
  --data-dir training_data/data \
  --output Models/HeAR/tb_cough_classifier.npz

# 3. Verify
python -c "import numpy as np; d=np.load('Models/HeAR/tb_cough_classifier.npz'); print(d['accuracy'])"
```

## Technical Notes

- **PyTorch compatibility**: The HeAR model weights were converted from
  `pytorch_model.bin` to `model.safetensors` for compatibility with
  Transformers ≥ 4.57 (which enforces CVE-2025-32434 safe loading).
- **Preprocessing**: Audio is resampled to 16 kHz, chunked into 2-second
  segments, converted to mel-PCEN spectrograms, then fed to the ViT encoder.
- **Runtime**: Embedding extraction takes ~0.5s per file on CPU.

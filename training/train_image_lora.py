"""
Image-Based LoRA Fine-Tuning Script for MedGemma 4B — v2
=========================================================
Fine-tunes MedGemma for binary TB classification on chest X-rays.

Key changes from v1:
  - Training prompt matches evaluation prompt exactly (short binary answer)
  - Training data: Tawsifurrahman (~1200 balanced) + Montgomery (~138)
  - Eval images from Tawsifurrahman are excluded from training
  - LoRA rank 32, 5 epochs, LR 2e-4
  - Optional vision tower fine-tuning (last N layers, conservative for 12GB)
  - bf16 validation after training to verify LoRA works before GGUF conversion

Usage:
    python train_image_lora.py

Output: LoRA adapter saved to ./lora_output/fieldscreen-tb-image-lora/
"""

import os
import sys
import csv
import json
import random
import logging
from pathlib import Path

# === CONFIGURATION ===
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent

# Model path — edit for your system (expects HuggingFace format with safetensors)
MODEL_PATH = str(PROJECT_DIR / "Models" / "MedGemma" / "medgemma-1.5-4b-it")

# Datasets
MONTGOMERY_IMAGES = PROJECT_DIR / "Dataset Montgomery" / "images" / "images"
MONTGOMERY_CSV = PROJECT_DIR / "Dataset Montgomery" / "montgomery_metadata.csv"
TAWSIFUR_DIR = PROJECT_DIR / "Dataset tawsifurrahman" / "TB_Chest_Radiography_Database"
TAWSIFUR_NORMAL = TAWSIFUR_DIR / "Normal"
TAWSIFUR_TB = TAWSIFUR_DIR / "Tuberculosis"

# Evaluation dataset (to exclude from training)
EVAL_DIR = SCRIPT_DIR / "eval_dataset"
EVAL_CSV = EVAL_DIR / "labels.csv"
EVAL_IMAGES = EVAL_DIR / "images"
EVAL_SEED = 123  # Must match build_eval_dataset.py

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "lora_output" / "fieldscreen-tb-image-lora"

# === TRAINING HYPERPARAMETERS ===
LORA_RANK = 32              # Up from 16 — more capacity
LORA_ALPHA = 64             # Alpha = 2x rank
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4        # Up from 1e-4 — more data can support it
NUM_EPOCHS = 5              # Up from 3 — more passes over larger dataset
BATCH_SIZE = 1              # Must be 1 for 12GB VRAM with images
GRADIENT_ACCUMULATION = 4   # Effective batch = 4
MAX_SEQ_LENGTH = 512        # Image uses ~256 tokens + prompt ~40 + target ~10
WARMUP_RATIO = 0.05         # 5% of total steps
SAVE_STEPS = 300            # Save roughly once per epoch
LOGGING_STEPS = 10

# === VISION TOWER FINE-TUNING ===
# Fine-tune the last N layers of the vision encoder (SigLIP).
# This lets the model adapt its visual feature extraction for TB.
# Set to 0 to disable (saves ~1-2GB VRAM).
# Recommended: 2 for 12GB VRAM, 4 for 16GB+.
# NOTE: Vision tower LoRA tensors are saved but skipped during GGUF
# conversion (llama.cpp LoRA doesn't support vision adapters yet).
# They DO take effect during bf16 validation (step 6).
VISION_LAYERS_TO_TRAIN = 2

# === PROMPT — must match evaluate_gguf.py exactly ===
EVAL_PROMPT = (
    "Is this chest X-ray normal or abnormal? "
    "Answer with: NORMAL or ABNORMAL, then confidence percentage."
)

# === BF16 VALIDATION ===
BF16_VALIDATION_SAMPLES = 30  # Eval images to test after training

# ===================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required packages."""
    missing = []
    for pkg in ['torch', 'transformers', 'peft', 'bitsandbytes', 'datasets', 'accelerate', 'PIL']:
        try:
            __import__(pkg)
        except ImportError:
            if pkg == 'PIL':
                missing.append('Pillow')
            else:
                missing.append(pkg)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)

    import torch
    import transformers
    logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Transformers: {transformers.__version__}")


def get_eval_exclusion_set():
    """Replicate the eval dataset sampling to know which Tawsifurrahman
    images to exclude from training.

    This reproduces the exact logic from build_eval_dataset.py:
    seed=123, sorted glob, sample 100 per class.
    """
    normal_files = sorted(TAWSIFUR_NORMAL.glob("*.png"))
    tb_files = sorted(TAWSIFUR_TB.glob("*.png"))

    rng = random.Random(EVAL_SEED)
    eval_normal = set(p.name for p in rng.sample(normal_files, min(100, len(normal_files))))
    eval_tb = set(p.name for p in rng.sample(tb_files, min(100, len(tb_files))))

    excluded = eval_normal | eval_tb
    logger.info(f"Excluding {len(excluded)} Tawsifurrahman images reserved for evaluation")
    return excluded


def load_tawsifurrahman_dataset(exclusion_set):
    """Load Tawsifurrahman dataset, excluding eval images.
    Returns class-balanced samples (~600 TB + ~600 Normal).
    """
    normal_samples = []
    tb_samples = []

    for img_path in sorted(TAWSIFUR_NORMAL.glob("*.png")):
        if img_path.name not in exclusion_set:
            normal_samples.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'is_tb': False,
                'source': 'tawsifurrahman',
            })

    for img_path in sorted(TAWSIFUR_TB.glob("*.png")):
        if img_path.name not in exclusion_set:
            tb_samples.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'is_tb': True,
                'source': 'tawsifurrahman',
            })

    logger.info(f"Tawsifurrahman (after exclusion): {len(normal_samples)} Normal, {len(tb_samples)} TB")

    # Balance: subsample Normal to match TB count
    target_normal = len(tb_samples)
    if len(normal_samples) > target_normal:
        normal_samples = random.sample(normal_samples, target_normal)
        logger.info(f"  Subsampled Normal to {target_normal} for class balance")

    all_samples = normal_samples + tb_samples
    logger.info(f"  Tawsifurrahman training set: {len(all_samples)} images (balanced)")
    return all_samples


def load_montgomery_dataset():
    """Load the full Montgomery dataset (small enough to include entirely)."""
    samples = []
    with open(MONTGOMERY_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['study_id'].strip()
            img_path = MONTGOMERY_IMAGES / filename

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            findings = row.get('findings', '').strip()
            is_tb = findings.lower() != 'normal'

            samples.append({
                'image_path': str(img_path),
                'filename': filename,
                'is_tb': is_tb,
                'source': 'montgomery',
            })

    tb_count = sum(1 for s in samples if s['is_tb'])
    normal_count = sum(1 for s in samples if not s['is_tb'])
    logger.info(f"Montgomery: {normal_count} Normal, {tb_count} TB ({len(samples)} total)")
    return samples


def build_target_response(is_tb):
    """Build short classification target matching evaluation format.

    Matches the eval prompt: "NORMAL or ABNORMAL, then confidence percentage."
    Slight confidence variation prevents the model from memorizing one number.
    """
    if is_tb:
        conf = random.randint(80, 95)
        return f"ABNORMAL. Confidence: {conf}%."
    else:
        conf = random.randint(90, 99)
        return f"NORMAL. Confidence: {conf}%."


def build_lora_target_modules(model):
    """Build the LoRA target_modules regex, optionally including vision tower layers."""
    # Language model: all attention + MLP projections
    lang = r"language_model.*(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

    if VISION_LAYERS_TO_TRAIN > 0:
        # Detect actual number of vision encoder layers
        n_vision_layers = 27  # SigLIP SoViT-400M default
        try:
            vision_layers = model.model.vision_tower.vision_model.encoder.layers
            n_vision_layers = len(vision_layers)
            logger.info(f"  Vision tower has {n_vision_layers} encoder layers")
        except Exception as e:
            logger.warning(f"  Could not detect vision tower layers ({e}), assuming {n_vision_layers}")

        n_to_train = min(VISION_LAYERS_TO_TRAIN, n_vision_layers)
        start_layer = n_vision_layers - n_to_train

        # Build regex matching only the last N layers
        # SigLIP uses: q_proj, k_proj, v_proj, out_proj (note: out_proj not o_proj)
        layer_nums = "|".join(str(i) for i in range(start_layer, n_vision_layers))
        vision = rf"vision_tower.*layers\.(?:{layer_nums})\..*(?:q_proj|k_proj|v_proj|out_proj)"

        combined = rf".*(?:{lang}|{vision})"
        logger.info(f"  LoRA targets: language_model (all) + vision_tower (layers {start_layer}-{n_vision_layers - 1})")
    else:
        combined = rf".*{lang}"
        logger.info(f"  LoRA targets: language_model only (vision tower frozen)")

    return combined


def main():
    logger.info("=" * 60)
    logger.info("MedGemma Image LoRA Training for TB Classification v2")
    logger.info("=" * 60)

    check_dependencies()

    # Validate paths
    for label, p in [
        ("Model", MODEL_PATH),
        ("Montgomery CSV", MONTGOMERY_CSV),
        ("Montgomery images", MONTGOMERY_IMAGES),
        ("Tawsifurrahman Normal", TAWSIFUR_NORMAL),
        ("Tawsifurrahman TB", TAWSIFUR_TB),
    ]:
        if not Path(p).exists():
            logger.error(f"{label} not found at: {p}")
            sys.exit(1)

    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # ---- Step 1: Load processor ----
    logger.info(f"\n[1/6] Loading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"
    logger.info(f"Processor loaded. Vocab size: {processor.tokenizer.vocab_size}")

    # ---- Step 2: Load model (bfloat16) ----
    logger.info(f"\n[2/6] Loading MedGemma in bfloat16...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    logger.info(f"Model loaded. Type: {type(model).__name__}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU memory after loading: {allocated:.2f} GB")

    # ---- Step 3: Configure LoRA ----
    logger.info(f"\n[3/6] Configuring LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")

    target_modules = build_lora_target_modules(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,
    )

    # Workaround for peft/AWQ version conflict
    import peft.tuners.lora.awq as _peft_awq
    _peft_awq.is_auto_awq_available = lambda: False

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"GPU memory after LoRA setup: {allocated:.2f} GB")

    # ---- Step 4: Prepare dataset ----
    logger.info(f"\n[4/6] Preparing training dataset...")

    # Exclude Tawsifurrahman images used in the eval set
    exclusion_set = get_eval_exclusion_set()

    # Load both datasets
    random.seed(42)
    tawsifur_samples = load_tawsifurrahman_dataset(exclusion_set)
    montgomery_samples = load_montgomery_dataset()

    # Combine, shuffle, split
    all_samples = tawsifur_samples + montgomery_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    eval_samples = all_samples[split_idx:]

    tb_train = sum(1 for s in train_samples if s['is_tb'])
    normal_train = sum(1 for s in train_samples if not s['is_tb'])
    logger.info(f"\nCombined dataset: {len(all_samples)} images")
    logger.info(f"  Train: {len(train_samples)} ({tb_train} TB, {normal_train} Normal)")
    logger.info(f"  Eval:  {len(eval_samples)}")

    # Preprocess all samples
    def preprocess_sample(sample):
        """Convert a sample into tokenized model inputs with short binary labels."""
        image = Image.open(sample['image_path']).convert('RGB')
        target_text = build_target_response(sample['is_tb'])

        # Full conversation (user prompt + assistant answer)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EVAL_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)

        # Find prompt length to create label mask
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EVAL_PROMPT},
                ],
            },
        ]
        prompt_only = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        prompt_length = prompt_only["input_ids"].shape[1]

        # Labels: -100 for prompt tokens, actual IDs for target tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100

        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
            labels = labels[:MAX_SEQ_LENGTH]

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "token_type_ids" in inputs:
            ttids = inputs["token_type_ids"].squeeze(0)
            if len(ttids) > MAX_SEQ_LENGTH:
                ttids = ttids[:MAX_SEQ_LENGTH]
            result["token_type_ids"] = ttids

        return result

    logger.info("\nTokenizing training samples (this may take a few minutes)...")
    train_data = []
    failed = 0
    for i, sample in enumerate(train_samples):
        try:
            processed = preprocess_sample(sample)
            train_data.append(processed)
            if i == 0:
                n_total = len(processed["input_ids"])
                n_valid = (processed["labels"] != -100).sum().item()
                logger.info(f"  First sample: {n_total} total tokens, "
                            f"{n_valid} label tokens (should be ~8-12 for short target)")
                if "pixel_values" in processed:
                    pv = processed["pixel_values"]
                    logger.info(f"  pixel_values shape: {pv.shape}, dtype: {pv.dtype}")
            if (i + 1) % 200 == 0:
                logger.info(f"  Tokenized {i+1}/{len(train_samples)}...")
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"  Failed to process {sample['filename']}: {e}")

    if failed > 5:
        logger.warning(f"  ... and {failed - 5} more failures")

    logger.info("Tokenizing eval samples...")
    eval_data = []
    for sample in eval_samples:
        try:
            eval_data.append(preprocess_sample(sample))
        except Exception:
            pass

    logger.info(f"Tokenized: {len(train_data)} train, {len(eval_data)} eval")

    if not train_data:
        logger.error("No training samples could be processed!")
        sys.exit(1)

    avg_len = sum(len(d['input_ids']) for d in train_data) / len(train_data)
    logger.info(f"Average sequence length: {avg_len:.0f} tokens")

    # Data collator for multimodal inputs
    class MultimodalDataCollator:
        def __init__(self, pad_token_id):
            self.pad_token_id = pad_token_id

        def __call__(self, features):
            max_len = max(len(f["input_ids"]) for f in features)
            batch = {"input_ids": [], "labels": [], "attention_mask": []}

            has_pv = "pixel_values" in features[0]
            has_tti = "token_type_ids" in features[0]
            if has_pv:
                batch["pixel_values"] = []
            if has_tti:
                batch["token_type_ids"] = []

            for f in features:
                pad_len = max_len - len(f["input_ids"])
                batch["input_ids"].append(
                    torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                )
                batch["labels"].append(
                    torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                )
                batch["attention_mask"].append(
                    torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
                if has_tti:
                    batch["token_type_ids"].append(
                        torch.cat([f["token_type_ids"], torch.zeros(pad_len, dtype=torch.long)])
                    )
                if has_pv:
                    batch["pixel_values"].append(f["pixel_values"])

            batch["input_ids"] = torch.stack(batch["input_ids"])
            batch["labels"] = torch.stack(batch["labels"])
            batch["attention_mask"] = torch.stack(batch["attention_mask"])
            if has_tti:
                batch["token_type_ids"] = torch.stack(batch["token_type_ids"])
            if has_pv:
                batch["pixel_values"] = torch.stack(batch["pixel_values"])

            return batch

    data_collator = MultimodalDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # ---- Step 5: Train ----
    steps_per_epoch = len(train_data) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    logger.info(f"\n[5/6] Starting LoRA training...")
    logger.info(f"  Training images:    {len(train_data)}")
    logger.info(f"  Epochs:             {NUM_EPOCHS}")
    logger.info(f"  Batch:              {BATCH_SIZE} x {GRADIENT_ACCUMULATION} accum = {BATCH_SIZE * GRADIENT_ACCUMULATION} effective")
    logger.info(f"  Steps/epoch:        ~{steps_per_epoch}")
    logger.info(f"  Total steps:        ~{total_steps}")
    logger.info(f"  Warmup steps:       {warmup_steps}")
    logger.info(f"  Learning rate:      {LEARNING_RATE}")
    logger.info(f"  LoRA rank/alpha:    {LORA_RANK}/{LORA_ALPHA}")
    logger.info(f"  Vision layers:      {VISION_LAYERS_TO_TRAIN}")
    logger.info(f"  Output:             {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=SAVE_STEPS if eval_data else None,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    # Trainer subclass to cast pixel_values to bf16
    class VisionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    trainer = VisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data if eval_data else None,
        data_collator=data_collator,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING STARTED — monitor loss below")
    logger.info("=" * 60 + "\n")

    result = trainer.train()

    # Save adapter
    logger.info(f"\nSaving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Training loss: {result.training_loss:.4f}")
    logger.info(f"  Total steps:   {result.global_step}")
    logger.info(f"  Adapter saved: {OUTPUT_DIR}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        logger.info(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # ---- Step 6: bf16 Validation ----
    logger.info(f"\n[6/6] bf16 validation — verifying LoRA works before GGUF conversion...")
    validate_bf16(model, processor, torch)


def validate_bf16(model, processor, torch):
    """Run quick inference validation at bf16 precision.

    Uses images from eval_dataset/ (independent Tawsifurrahman set, never
    used for training). This confirms the LoRA actually changes model
    behavior before going through the GGUF conversion pipeline.
    """
    from PIL import Image

    if not EVAL_CSV.exists():
        logger.warning("  eval_dataset/ not found — skipping bf16 validation")
        logger.warning("  Run BUILD_EVAL_DATASET.bat first to create it")
        return

    # Load eval labels
    eval_items = []
    with open(EVAL_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = EVAL_IMAGES / row['filename'].strip()
            if img_path.exists():
                eval_items.append({
                    'image_path': str(img_path),
                    'filename': row['filename'].strip(),
                    'ground_truth': row['ground_truth'].strip(),
                })

    # Deterministic subset
    random.seed(99)
    n = min(BF16_VALIDATION_SAMPLES, len(eval_items))
    eval_items = random.sample(eval_items, n)

    # Free training memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = len(eval_items)

    logger.info(f"  Testing {total} images from independent eval set at bf16...")

    for item in eval_items:
        try:
            image = Image.open(item['image_path']).convert('RGB')

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": EVAL_PROMPT},
                    ],
                },
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Classify response
            text_upper = response.upper()
            if "ABNORMAL" in text_upper:
                pred = "TB"
            elif "NORMAL" in text_upper:
                pred = "NORMAL"
            else:
                pred = "UNCLEAR"

            ok = pred == item['ground_truth']
            if ok:
                correct += 1

            mark = "OK" if ok else "FAIL"
            logger.info(f"    {item['filename']}  GT={item['ground_truth']:<6} "
                        f"Pred={pred:<7} [{mark}]  \"{response[:60]}\"")

        except Exception as e:
            logger.warning(f"    {item['filename']}  ERROR: {e}")

    acc = correct / total * 100 if total else 0
    logger.info(f"\n  bf16 Validation: {correct}/{total} = {acc:.1f}%")
    logger.info(f"  (Base model baseline is ~83.5% — improvement above this confirms LoRA works)")

    if acc <= 83.5:
        logger.warning("  LoRA did not improve over base model at bf16. Consider:")
        logger.warning("    - Training for more epochs (increase NUM_EPOCHS)")
        logger.warning("    - Increasing LoRA rank (increase LORA_RANK)")
        logger.warning("    - Checking training loss — if it didn't decrease, check data pipeline")
    else:
        logger.info(f"  LoRA improvement confirmed: +{acc - 83.5:.1f}% over base model")

    model.train()


if __name__ == "__main__":
    main()

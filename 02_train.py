"""
02_train.py - Fine-tune a model with LoRA for EV financial extraction

=============================================================================
WHAT THIS SCRIPT DOES (High Level)
=============================================================================
1. Load a pre-trained model (Qwen2.5-3B - small but capable)
2. Apply LoRA adapters (small trainable matrices)
3. Train on our EV financial data
4. Save the fine-tuned model

=============================================================================
KEY CONCEPTS EXPLAINED IN THIS FILE
=============================================================================
- Quantization: Loading model in 4-bit to save VRAM
- LoRA Config: The hyperparameters that control adaptation
- SFTTrainer: Supervised Fine-Tuning trainer from TRL library
- Training Arguments: Learning rate, batch size, epochs, etc.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================

# Model to fine-tune
# Qwen2.5-3B is a good balance of capability vs. speed for learning
# Other options: "Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-3B", "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Paths
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "models/ev-extractor-lora"

# LoRA Hyperparameters
LORA_R = 16          # Rank: higher = more capacity, more memory
LORA_ALPHA = 32      # Scaling factor: typically 2x rank
LORA_DROPOUT = 0.05  # Regularization: prevents overfitting

# Training Hyperparameters
LEARNING_RATE = 2e-4      # How big steps to take (2e-4 is standard for LoRA)
BATCH_SIZE = 2            # Examples per step (limited by VRAM)
GRADIENT_ACCUMULATION = 4 # Effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3            # Full passes through data
MAX_SEQ_LENGTH = 1024     # Maximum tokens per example


def print_section(title):
    """Helper to print section headers"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_section("STEP 1: CHECK GPU AVAILABILITY")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Show GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = torch.cuda.get_device_properties(i).total_memory
        print(f"  GPU {i}: {props.name}")
        print(f"         Total Memory: {props.total_memory / 1024**3:.1f} GB")

    # ==========================================================================
    print_section("STEP 2: LOAD TOKENIZER")
    # ==========================================================================
    # The tokenizer converts text to numbers (tokens) and back.
    # Each model has its own vocabulary - must use matching tokenizer.

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Important: Set padding token
    # Some models don't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: '{tokenizer.pad_token}'")

    print(f"  Vocabulary size: {len(tokenizer):,} tokens")

    # ==========================================================================
    print_section("STEP 3: CONFIGURE QUANTIZATION (4-bit)")
    # ==========================================================================
    # THEORY: Quantization reduces model precision to save memory
    #
    # Full precision (float32):  32 bits per weight
    # Half precision (float16):  16 bits per weight  → 2x memory savings
    # 4-bit quantization:         4 bits per weight  → 8x memory savings!
    #
    # A 3B model normally needs ~6GB. In 4-bit: ~1.5GB
    # This lets us fit larger models on your 3090s

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                     # Use 4-bit quantization
        bnb_4bit_quant_type="nf4",             # NormalFloat4 - best for transformers
        bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 for stability
        bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
    )

    print("Quantization config:")
    print("  - 4-bit precision (NF4)")
    print("  - bfloat16 compute")
    print("  - Double quantization enabled")
    print("  - Expected memory: ~2GB for 3B model")

    # ==========================================================================
    print_section("STEP 4: LOAD MODEL")
    # ==========================================================================
    # This downloads and loads the model with our quantization settings

    print(f"Loading {MODEL_NAME}...")
    print("(This may take a few minutes on first run - downloading ~6GB)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically place on available GPUs
        trust_remote_code=True,
    )

    # Prepare for k-bit training (required for quantized models)
    model = prepare_model_for_kbit_training(model)

    print(f"  Model loaded!")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")

    # ==========================================================================
    print_section("STEP 5: CONFIGURE LoRA")
    # ==========================================================================
    # THEORY: LoRA adds small trainable matrices to attention layers
    #
    # Instead of updating W (huge), we learn A and B where:
    #   W_new = W_original + A @ B
    #
    # target_modules: which layers to add LoRA to
    #   - q_proj, k_proj, v_proj: attention projections (most important)
    #   - o_proj: attention output
    #   - gate_proj, up_proj, down_proj: MLP layers (optional, more capacity)

    lora_config = LoraConfig(
        r=LORA_R,                               # Rank of adaptation
        lora_alpha=LORA_ALPHA,                  # Scaling factor
        lora_dropout=LORA_DROPOUT,              # Dropout for regularization
        target_modules=[                        # Which layers to adapt
            "q_proj", "k_proj", "v_proj",       # Attention
            "o_proj",                           # Attention output
            "gate_proj", "up_proj", "down_proj" # MLP (optional, remove if OOM)
        ],
        bias="none",                            # Don't train biases
        task_type="CAUSAL_LM",                  # We're doing language modeling
    )

    print(f"LoRA Configuration:")
    print(f"  - Rank (r): {LORA_R}")
    print(f"  - Alpha: {LORA_ALPHA}")
    print(f"  - Scaling factor: {LORA_ALPHA / LORA_R}")
    print(f"  - Dropout: {LORA_DROPOUT}")
    print(f"  - Target modules: {lora_config.target_modules}")

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Show trainable parameters
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"\n  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.4f}%")

    # ==========================================================================
    print_section("STEP 6: LOAD TRAINING DATA")
    # ==========================================================================

    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"  Loaded {len(dataset)} examples")

    # Show a sample
    print(f"\n  Sample (first 200 chars of first message):")
    print(f"  {str(dataset[0])[:200]}...")

    # ==========================================================================
    print_section("STEP 7: CONFIGURE TRAINING")
    # ==========================================================================
    # THEORY: Training arguments control how we update weights
    #
    # learning_rate: How big of a step to take. Too high = unstable, too low = slow
    # batch_size: Examples per step. Limited by VRAM.
    # gradient_accumulation: Simulate larger batches by accumulating gradients
    # epochs: How many times to see each example

    # SFTConfig combines TrainingArguments + SFT-specific settings
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,                  # L2 regularization
        warmup_ratio=0.1,                   # Warm up learning rate for first 10%
        lr_scheduler_type="cosine",         # Decay learning rate smoothly
        logging_steps=1,                    # Log every step (we have few examples)
        save_strategy="epoch",              # Save after each epoch
        bf16=True,                          # Use bfloat16 for training
        gradient_checkpointing=True,        # Save memory by recomputing activations
        optim="paged_adamw_8bit",           # Memory-efficient optimizer
        report_to="none",                   # Disable wandb for now
        max_grad_norm=0.3,                  # Clip gradients to prevent explosion
        # SFT-specific settings (new in TRL 0.27+)
        max_length=MAX_SEQ_LENGTH,          # Maximum sequence length
        packing=False,                      # Don't pack multiple examples
    )

    print("Training configuration:")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Warmup: 10% of steps")
    print(f"  - LR scheduler: cosine decay")

    # ==========================================================================
    print_section("STEP 8: CREATE TRAINER")
    # ==========================================================================
    # SFTTrainer (Supervised Fine-Tuning Trainer) handles:
    # - Tokenizing conversations into the right format
    # - Masking system/user messages (only train on assistant responses)
    # - The training loop itself

    def formatting_func(example):
        """Convert our chat format to the model's expected format"""
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_func,
    )

    print("SFTTrainer created!")
    print(f"  - Max sequence length: {MAX_SEQ_LENGTH}")

    # ==========================================================================
    print_section("STEP 9: TRAIN!")
    # ==========================================================================
    print("Starting training...\n")
    print("Watch the loss go down! Lower = model is learning your format.\n")

    # Train the model
    train_result = trainer.train()

    # Print results
    print_section("TRAINING COMPLETE!")
    print(f"  Training loss: {train_result.training_loss:.4f}")
    print(f"  Training time: {train_result.metrics['train_runtime']:.1f} seconds")

    # ==========================================================================
    print_section("STEP 10: SAVE MODEL")
    # ==========================================================================

    # Save the LoRA adapter (small, ~50MB)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nFiles saved:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f}: {size / 1024:.1f} KB")

    # ==========================================================================
    print_section("NEXT STEPS")
    # ==========================================================================
    print("""
1. Test your model:
   python 03_inference.py
   python 04_evaluate.py --mode lora

2. The LoRA adapter is saved separately from the base model.
   To use it, you load base model + adapter together.

3. To improve results:
   - Add more training examples (aim for 50-100)
   - Train for more epochs
   - Adjust LoRA rank (higher = more capacity)
""")


if __name__ == "__main__":
    main()

"""
04_evaluate.py - Field-level evaluation of the base model vs. the LoRA adapter

Runs every example in data/test.jsonl through a model, parses the JSON answer,
and scores each field against the reference:

- company:          correct if the distinctive first word of the name matches
- period:           exact match after whitespace/case normalisation
- monetary / units: correct if within 1% of the reference
- gross_margin_pct: correct if within 0.1 percentage points

Usage:
    python 04_evaluate.py --mode base
    python 04_evaluate.py --mode lora
    python 04_evaluate.py --summarize          # print a table from saved results
"""

import argparse
import json
import time
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/ev-extractor-lora"
TEST_PATH = Path("data/test.jsonl")
RESULTS_DIR = Path("results")

FIELDS = ["company", "period", "revenue", "cost_of_revenue", "gross_profit",
          "gross_margin_pct", "units_delivered", "operating_income"]
RELATIVE_FIELDS = {"revenue", "cost_of_revenue", "gross_profit", "units_delivered", "operating_income"}


# =============================================================================
# Scoring
# =============================================================================

def field_correct(field: str, predicted, expected) -> bool:
    if predicted is None:
        return False
    try:
        if field == "company":
            return str(predicted).split()[0].lower() == str(expected).split()[0].lower()
        if field == "period":
            return " ".join(str(predicted).split()).lower() == str(expected).lower()
        if field == "gross_margin_pct":
            return abs(float(predicted) - float(expected)) <= 0.1
        if field in RELATIVE_FIELDS:
            return abs(float(predicted) - float(expected)) <= 0.01 * abs(float(expected))
    except (TypeError, ValueError):
        return False
    return False


def parse_json(text: str):
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# =============================================================================
# Model loading and generation
# =============================================================================

def load(mode: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        # Same 4-bit setup as training
        kwargs = dict(device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    else:
        # Apple Silicon / CPU: bitsandbytes 4-bit is CUDA-only, so load in fp16
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        kwargs = dict(dtype=torch.float16, device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    if mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict]) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# =============================================================================
# Main
# =============================================================================

def evaluate(mode: str) -> None:
    examples = [json.loads(line) for line in TEST_PATH.open()]
    model, tokenizer = load(mode)

    records = []
    for i, example in enumerate(examples, 1):
        prompt_messages = example["messages"][:2]
        expected = json.loads(example["messages"][2]["content"])

        t0 = time.perf_counter()
        raw = generate(model, tokenizer, prompt_messages)
        latency = time.perf_counter() - t0

        predicted = parse_json(raw)
        scores = {f: field_correct(f, (predicted or {}).get(f), expected[f]) for f in FIELDS}
        records.append({"expected": expected, "raw": raw, "parsed": predicted is not None,
                        "scores": scores, "latency_s": latency})
        print(f"[{mode}] {i:2d}/{len(examples)}  fields correct {sum(scores.values())}/{len(FIELDS)}"
              f"  ({latency:.1f}s)  {expected['company']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{mode}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"Saved {out}")


def summarize() -> None:
    runs = {mode: json.loads((RESULTS_DIR / f"eval_{mode}.json").read_text())
            for mode in ("base", "lora") if (RESULTS_DIR / f"eval_{mode}.json").exists()}

    header = "| Metric | " + " | ".join(runs) + " |"
    print(header)
    print("|---|" + "---:|" * len(runs))

    def row(label, fn):
        print(f"| {label} | " + " | ".join(fn(r) for r in runs.values()) + " |")

    row("Valid JSON", lambda r: f"{sum(x['parsed'] for x in r)}/{len(r)}")
    row("All 8 fields correct", lambda r: f"{sum(all(x['scores'].values()) for x in r)}/{len(r)}")
    for field in FIELDS:
        row(f"`{field}`", lambda r, f=field: f"{sum(x['scores'][f] for x in r)}/{len(r)}")
    row("Field accuracy (overall)",
        lambda r: f"{100 * sum(sum(x['scores'].values()) for x in r) / (len(r) * len(FIELDS)):.1f}%")
    row("Median latency (s)", lambda r: f"{sorted(x['latency_s'] for x in r)[len(r) // 2]:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "lora"])
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        evaluate(args.mode)

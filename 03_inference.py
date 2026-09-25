"""
03_inference.py - Test your fine-tuned EV financial extractor

This script loads your LoRA adapter and runs inference on new reports.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "models/ev-extractor-lora"

SYSTEM_PROMPT = """You are an EV financial data extraction assistant.
Extract key financial metrics from electric vehicle company reports and output them as JSON.
Always include: company, period, revenue, cost_of_revenue, gross_profit, gross_margin_pct, units_delivered, operating_income.
All monetary values should be in USD as integers. Percentages as floats."""


def load_model():
    """Load the base model with LoRA adapter"""
    print("Loading model...")

    # Quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print(f"  Loaded adapter from {ADAPTER_PATH}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model ready!\n")
    return model, tokenizer


def extract_financials(model, tokenizer, report_text: str) -> dict:
    """
    Extract financial data from an EV report.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        report_text: Raw text from an EV company report

    Returns:
        dict: Extracted financial data
    """
    # Create the chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract financial data from this report:\n\n{report_text}"}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # Add the assistant prompt to trigger generation
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,      # Low temperature for consistent output
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode (only the new tokens)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Try to parse as JSON
    try:
        # Find JSON in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    # Return raw response if JSON parsing fails
    return {"raw_response": response, "error": "Could not parse JSON"}


def main():
    # Load model
    model, tokenizer = load_model()

    # Test reports (not in training data)
    test_reports = [
        # Test 1: Tesla format
        """Tesla Q1 2025 Preliminary Results:
First quarter saw revenue of $23.8 billion with cost of revenue at $19.9 billion.
Vehicle deliveries totaled 387,000 units. Gross margin was 16.4%.
Operating income came in at $1.8 billion.""",

        # Test 2: Different style
        """BYD Financial Update - Q4 2024:
Revenue reached $32.1B USD this quarter, with COGS of $25.7B.
The company shipped 1,287,000 electric vehicles globally.
Gross profit margin: 19.9%. Operating profit: $3.2 billion.""",

        # Test 3: More challenging format
        """Rivian Q4 2024 Highlights
- Total revenues: $1.52B
- Cost of sales: $1.61B (negative gross margin of -5.9%)
- Deliveries: 14,200 vehicles
- Operating loss: $(920)M"""
    ]

    print("=" * 70)
    print(" TESTING FINE-TUNED MODEL")
    print("=" * 70)

    for i, report in enumerate(test_reports, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}")
        print(f"{'─' * 70}")
        print(f"INPUT:\n{report[:200]}...")
        print(f"\nOUTPUT:")

        result = extract_financials(model, tokenizer, report)
        print(json.dumps(result, indent=2))

    # Interactive mode
    print("\n" + "=" * 70)
    print(" INTERACTIVE MODE")
    print("=" * 70)
    print("Enter EV report text (or 'quit' to exit):\n")

    while True:
        user_input = input("Report> ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        result = extract_financials(model, tokenizer, user_input)
        print("\nExtracted:")
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()

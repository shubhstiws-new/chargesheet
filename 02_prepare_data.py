"""
02_prepare_data.py - Create training data for EV Financial Extractor

THEORY RECAP:
Fine-tuning teaches the model input→output patterns.
We need many examples of: "raw EV report text" → "structured JSON output"

The model learns: "When I see text like THIS, I should output THAT format"
"""

import json
from pathlib import Path

# =============================================================================
# STEP 1: Define your output schema
# =============================================================================
# This is what we want the model to output EVERY time.
# Modify these fields based on your accounting needs.

SCHEMA = {
    "company": "string - company name",
    "period": "string - fiscal period (e.g., Q3 2024)",
    "revenue": "integer - total revenue in USD",
    "cost_of_revenue": "integer - COGS in USD",
    "gross_profit": "integer - revenue minus COGS",
    "gross_margin_pct": "float - gross profit / revenue * 100",
    "units_delivered": "integer - vehicles delivered",
    "operating_income": "integer - operating profit in USD",
}

print("=" * 60)
print("OUTPUT SCHEMA (fields the model will extract)")
print("=" * 60)
for field, description in SCHEMA.items():
    print(f"  {field:20} : {description}")
print()


# =============================================================================
# STEP 2: Create training examples
# =============================================================================
# Each example is a pair: (input_text, expected_output)
#
# IMPORTANT: The more examples, the better. But quality > quantity.
# - 50-100 examples: decent results
# - 200-500 examples: good results
# - 1000+ examples: great results
#
# For now, we'll create ~20 examples to test the pipeline.

TRAINING_EXAMPLES = [
    # Example 1: Tesla Q3 2024
    {
        "input": """Tesla Q3 2024 Financial Results:
Total revenue reached $25.18 billion for the quarter. Cost of revenues was $20.88 billion.
The company delivered 462,890 vehicles during the period. Gross margin came in at 17.1%.
Operating income was $2.72 billion.""",
        "output": {
            "company": "Tesla",
            "period": "Q3 2024",
            "revenue": 25180000000,
            "cost_of_revenue": 20880000000,
            "gross_profit": 4300000000,
            "gross_margin_pct": 17.1,
            "units_delivered": 462890,
            "operating_income": 2720000000
        }
    },

    # Example 2: Tesla Q2 2024
    {
        "input": """Tesla Second Quarter 2024:
Revenue: $25.5B. COGS: $21.1B. Vehicles delivered: 443,956 units.
Gross profit margin was 18.0%. Operating profit reached $1.6 billion.""",
        "output": {
            "company": "Tesla",
            "period": "Q2 2024",
            "revenue": 25500000000,
            "cost_of_revenue": 21100000000,
            "gross_profit": 4400000000,
            "gross_margin_pct": 18.0,
            "units_delivered": 443956,
            "operating_income": 1600000000
        }
    },

    # Example 3: BYD Q3 2024 (numbers are illustrative)
    {
        "input": """BYD Company Limited Q3 2024 Results:
The Chinese EV giant reported quarterly revenue of RMB 201.1 billion (approximately $28.3B USD).
Cost of sales totaled $23.4 billion. Vehicle deliveries hit 1,134,892 units for the quarter.
Gross margin expanded to 21.9%. Operating income was $2.1 billion.""",
        "output": {
            "company": "BYD",
            "period": "Q3 2024",
            "revenue": 28300000000,
            "cost_of_revenue": 23400000000,
            "gross_profit": 4900000000,
            "gross_margin_pct": 21.9,
            "units_delivered": 1134892,
            "operating_income": 2100000000
        }
    },

    # Example 4: Rivian Q3 2024
    {
        "input": """Rivian Automotive Q3 2024 Earnings:
Rivian reported revenue of $1.34 billion. Total cost of revenue was $1.53 billion.
The company delivered 13,157 vehicles. Gross margin remained negative at -14.2%.
Operating loss was $1.1 billion.""",
        "output": {
            "company": "Rivian",
            "period": "Q3 2024",
            "revenue": 1340000000,
            "cost_of_revenue": 1530000000,
            "gross_profit": -190000000,
            "gross_margin_pct": -14.2,
            "units_delivered": 13157,
            "operating_income": -1100000000
        }
    },

    # Example 5: Lucid Q2 2024
    {
        "input": """Lucid Group Q2 2024 Financial Summary:
Total revenues were $200.6 million with cost of revenue at $403.2 million.
Lucid delivered 2,394 vehicles in Q2. The gross margin was -101%, reflecting
early production challenges. Operating loss totaled $790 million.""",
        "output": {
            "company": "Lucid",
            "period": "Q2 2024",
            "revenue": 200600000,
            "cost_of_revenue": 403200000,
            "gross_profit": -202600000,
            "gross_margin_pct": -101.0,
            "units_delivered": 2394,
            "operating_income": -790000000
        }
    },

    # Example 6: NIO Q3 2024
    {
        "input": """NIO Inc. Third Quarter 2024:
Vehicle sales generated $2.41 billion in revenue. Total revenue including services
was $2.66 billion. Cost of sales was $2.26 billion. NIO delivered 61,855 vehicles.
Gross margin improved to 15.1%. The company posted an operating loss of $680 million.""",
        "output": {
            "company": "NIO",
            "period": "Q3 2024",
            "revenue": 2660000000,
            "cost_of_revenue": 2260000000,
            "gross_profit": 400000000,
            "gross_margin_pct": 15.1,
            "units_delivered": 61855,
            "operating_income": -680000000
        }
    },

    # Example 7: XPeng Q3 2024
    {
        "input": """XPeng Q3 2024 Report:
XPeng achieved total revenues of RMB 10.1 billion ($1.42B USD). Vehicle deliveries
reached 46,533 units. Cost of revenue: $1.21 billion. Gross margin: 14.8%.
Operating expenses led to an operating loss of $320 million.""",
        "output": {
            "company": "XPeng",
            "period": "Q3 2024",
            "revenue": 1420000000,
            "cost_of_revenue": 1210000000,
            "gross_profit": 210000000,
            "gross_margin_pct": 14.8,
            "units_delivered": 46533,
            "operating_income": -320000000
        }
    },

    # Example 8: Tesla Q4 2023
    {
        "input": """Tesla Q4 2023 Financial Highlights:
Fourth quarter revenue was $25.17 billion. Cost of revenues totaled $21.06 billion.
A total of 484,507 vehicles were delivered. Gross margin: 17.6%.
Operating income for the quarter: $2.06 billion.""",
        "output": {
            "company": "Tesla",
            "period": "Q4 2023",
            "revenue": 25170000000,
            "cost_of_revenue": 21060000000,
            "gross_profit": 4110000000,
            "gross_margin_pct": 17.6,
            "units_delivered": 484507,
            "operating_income": 2060000000
        }
    },

    # Example 9: Ford EV Division Q3 2024 (Model e)
    {
        "input": """Ford Model e (EV Division) Q3 2024:
Ford's electric vehicle segment generated $1.2 billion in revenue.
Segment costs were $1.8 billion. Ford sold 23,509 EVs in the quarter.
The division's gross margin was -50.0%, with an operating loss of $1.2 billion.""",
        "output": {
            "company": "Ford Model e",
            "period": "Q3 2024",
            "revenue": 1200000000,
            "cost_of_revenue": 1800000000,
            "gross_profit": -600000000,
            "gross_margin_pct": -50.0,
            "units_delivered": 23509,
            "operating_income": -1200000000
        }
    },

    # Example 10: GM Ultium/EV Q3 2024
    {
        "input": """General Motors EV Update Q3 2024:
GM's EV portfolio achieved revenues of $4.3 billion with production costs of $3.9B.
The company delivered 32,095 electric vehicles. EV gross margin reached 9.3%.
EV-related operating income: $86 million.""",
        "output": {
            "company": "GM EV",
            "period": "Q3 2024",
            "revenue": 4300000000,
            "cost_of_revenue": 3900000000,
            "gross_profit": 400000000,
            "gross_margin_pct": 9.3,
            "units_delivered": 32095,
            "operating_income": 86000000
        }
    },

    # Example 11: Polestar Q3 2024
    {
        "input": """Polestar Automotive Q3 2024:
Revenue came in at $536 million. Total cost of sales: $548 million.
Polestar delivered 11,900 vehicles globally. Gross margin: -2.2%.
Operating loss was $236 million for the quarter.""",
        "output": {
            "company": "Polestar",
            "period": "Q3 2024",
            "revenue": 536000000,
            "cost_of_revenue": 548000000,
            "gross_profit": -12000000,
            "gross_margin_pct": -2.2,
            "units_delivered": 11900,
            "operating_income": -236000000
        }
    },

    # Example 12: VinFast Q2 2024
    {
        "input": """VinFast Auto Q2 2024 Results:
The Vietnamese EV maker reported $377 million in revenue against costs of $590M.
Vehicle deliveries totaled 9,689 units. Gross margin was deeply negative at -56.5%.
Operating loss expanded to $620 million.""",
        "output": {
            "company": "VinFast",
            "period": "Q2 2024",
            "revenue": 377000000,
            "cost_of_revenue": 590000000,
            "gross_profit": -213000000,
            "gross_margin_pct": -56.5,
            "units_delivered": 9689,
            "operating_income": -620000000
        }
    },
]


# =============================================================================
# STEP 3: Format for fine-tuning
# =============================================================================
# Models expect a specific format. The most common is "instruction format":
#
# <instruction>
# Extract financial data from this EV report.
# </instruction>
#
# <input>
# [the raw report text]
# </input>
#
# <output>
# [the JSON we want]
# </output>

SYSTEM_PROMPT = """You are an EV financial data extraction assistant.
Extract key financial metrics from electric vehicle company reports and output them as JSON.
Always include: company, period, revenue, cost_of_revenue, gross_profit, gross_margin_pct, units_delivered, operating_income.
All monetary values should be in USD as integers. Percentages as floats."""

def format_for_training(example: dict) -> dict:
    """
    Convert our example into the format expected by the training script.

    This creates a "conversation" format that most modern models expect:
    - system: Instructions for the model
    - user: The input (EV report text)
    - assistant: The expected output (JSON)
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract financial data from this report:\n\n{example['input']}"},
            {"role": "assistant", "content": json.dumps(example['output'], indent=2)}
        ]
    }


def main():
    # Create formatted training data
    formatted_data = [format_for_training(ex) for ex in TRAINING_EXAMPLES]

    # Save to data directory
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / "train.jsonl"

    with open(output_file, "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    print("=" * 60)
    print("TRAINING DATA CREATED")
    print("=" * 60)
    print(f"Examples: {len(formatted_data)}")
    print(f"Output: {output_file}")
    print()

    # Show one formatted example
    print("=" * 60)
    print("SAMPLE FORMATTED EXAMPLE")
    print("=" * 60)
    sample = formatted_data[0]
    print(f"System: {sample['messages'][0]['content'][:100]}...")
    print()
    print(f"User: {sample['messages'][1]['content'][:200]}...")
    print()
    print(f"Assistant: {sample['messages'][2]['content'][:200]}...")
    print()

    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review the examples - add more for better results!")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python 03_train.py")
    print()


if __name__ == "__main__":
    main()

# ChargeSheet

**Parameter-efficient fine-tuning (QLoRA) of a 3B language model to extract structured financial metrics from EV company reports.**

Quarterly results from electric-vehicle manufacturers are reported in inconsistent formats: prose, bullet lists, mixed currencies, losses written as negatives or in parentheses, and figures in millions or billions. ChargeSheet fine-tunes `Qwen2.5-3B-Instruct` to return a fixed eight-field JSON record from that text, suitable for loading directly into a spreadsheet or database.

**Input**

```
Rivian Automotive Q3 2024 Earnings:
Rivian reported revenue of $1.34 billion. Total cost of revenue was $1.53 billion.
The company delivered 13,157 vehicles. Gross margin remained negative at -14.2%.
Operating loss was $1.1 billion.
```

**Output**

```json
{
  "company": "Rivian",
  "period": "Q3 2024",
  "revenue": 1340000000,
  "cost_of_revenue": 1530000000,
  "gross_profit": -190000000,
  "gross_margin_pct": -14.2,
  "units_delivered": 13157,
  "operating_income": -1100000000
}
```

The target schema requires normalisation beyond copying text: amounts converted to whole USD, RMB figures replaced with the stated USD equivalent, gross profit derived when it is not reported, and losses expressed as negative values.

---

## Approach

| Step | Script | Detail |
|---|---|---|
| Data | `01_prepare_data.py` | 12 hand-written training examples covering 10 manufacturers and several reporting styles, formatted as system / user / assistant chat messages. |
| Training | `02_train.py` | QLoRA: base model loaded in 4-bit NF4 with double quantisation; LoRA adapters (rank 16, alpha 32, dropout 0.05) on all attention and MLP projections; TRL `SFTTrainer`, paged 8-bit AdamW, cosine schedule, learning rate 2e-4, effective batch size 8, 3 epochs. |
| Inference | `03_inference.py` | Loads base model plus adapter; extracts JSON from the generated text. |
| Evaluation | `04_evaluate.py` | Field-level scoring of the base model and the fine-tuned model on a held-out test set. |

## Training run

The adapter trained in 6 optimiser steps on a single GPU. It has 30M trainable parameters, about 1% of the 3.1B-parameter base model.

| Step | Epoch | Loss | Token accuracy |
|---:|---:|---:|---:|
| 1 | 0.7 | 1.97 | 65.3% |
| 2 | 1.0 | 1.98 | 65.2% |
| 3 | 1.7 | 1.47 | 69.4% |
| 4 | 2.0 | 1.20 | 73.5% |
| 5 | 2.7 | 1.08 | 74.6% |
| 6 | 3.0 | 1.03 | 75.2% |

Training loss is not evidence of extraction quality on unseen reports, so a held-out evaluation is included.

## Evaluation design

`data/test.jsonl` contains 12 held-out reports for **fictional** manufacturers. Fictional companies ensure that correct answers come from reading the text, not from facts memorised during pre-training. The set deliberately varies the input:

- Prose, bullet lists, and single-line summaries
- Figures in millions and billions; RMB amounts with a stated USD equivalent
- Losses written as negatives, as "loss", and in accounting parentheses, for example `$(372)M`
- Two reports that state gross profit but omit cost of revenue, the reverse of the training pattern, to test whether the derivation generalises

Scoring (`04_evaluate.py`):

| Field | Counted correct when |
|---|---|
| `company` | Distinctive first word of the name matches |
| `period` | Exact match, for example `Q3 2025` |
| Amounts and units | Within 1% of the reference; the sign must match |
| `gross_margin_pct` | Within 0.1 percentage points |

The script reports valid-JSON rate, fully correct records, per-field accuracy, and latency for both the base model and the base model with the adapter. It uses 4-bit loading on CUDA and fp16 elsewhere (Apple Silicon or CPU).

```bash
python 04_evaluate.py --mode base
python 04_evaluate.py --mode lora
python 04_evaluate.py --summarize
```

## Getting started

Requirements: Python 3.10+, an NVIDIA GPU with at least 12 GB of memory for training.

```bash
git clone https://github.com/shubhstiws-new/chargesheet.git
cd chargesheet
pip install -r requirements.txt

python 01_prepare_data.py      # writes data/train.jsonl
python 02_train.py             # writes models/ev-extractor-lora (not tracked in git)
python 03_inference.py
```

## Current status and limitations

| Item | Status |
|---|---|
| Training pipeline | Implemented; adapter trained |
| Held-out evaluation | Test set and scoring script implemented; **base vs. fine-tuned results not yet recorded** |
| Training data size | 12 examples is enough to teach the output format, not robust extraction. The next step is to generate a few hundred varied examples programmatically and measure how accuracy scales with data size. |
| Baselines | The comparison should also include few-shot prompting of the base model, which may match a fine-tune trained on this little data. |
| Real filings | Inputs are short, clean summaries. Full 10-Q / annual-report text would require retrieval or chunking before extraction. |
| Loss masking | With the current configuration, loss is computed over the full sequence, including the prompt. Restricting it to the assistant response is a standard improvement. |

## Capabilities developed

Built on February 21, 2026 as a first hands-on fine-tuning project.

- QLoRA mechanics: 4-bit quantisation, low-rank adapters, choice of target modules, and memory-saving training settings
- Supervised fine-tuning with chat-formatted data using Hugging Face TRL and PEFT
- Designing an evaluation that separates reading ability from memorised knowledge
- Defining field-level metrics for structured-output tasks

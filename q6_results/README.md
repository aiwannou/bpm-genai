# Q6: Evaluation of the timestamp correction

Comparison of the 2 LLMs × 2 correction methods of Q5 (see `../q5_results/README.md`).
Official runs of 14 Nov 2026. Written up in section 7 of the report.

## Results (final, after the self-check)
Rows: method · columns: LLM · ground truth: the 29 wrong events of Q1 (500 events in total).

**Accuracy**

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 1.00 | 1.00 |
| B: Machine learning | 1.00 | 1.00 |

**Precision**

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 1.00 | 1.00 |
| B: Machine learning | 1.00 | 1.00 |

**Recall**

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 1.00 | 1.00 |
| B: Machine learning | 1.00 | 1.00 |

**F1**

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 1.00 | 1.00 |
| B: Machine learning | 1.00 | 1.00 |

**Correct repair rate** (the fixed timestamp is valid and puts the case back in the correct order)

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 1.00 | 1.00 |
| B: Machine learning | 1.00 | 1.00 |

**Hallucination rate** (invented events + invalid values + correct events that were changed, over all outputs)

| | GPT-OSS 120B | Qwen 3.8 27B |
|---|---|---|
| A: Statistical | 0.00 | 0.00 |
| B: Machine learning | 0.00 | 0.00 |

All 4 combinations: TP 29 · FP 0 · FN 0 · TN 471 · invented events 0 · invalid values 0.

### First pass: the LLM's first (blind) answer, before the self-check

| Metric | GPT-OSS · A | GPT-OSS · B | Qwen · A | Qwen · B |
|---|---|---|---|---|
| TP / FP / FN | 27 / 1 / 2 | 28 / 0 / 1 | 29 / 1 / 0 | 25 / 0 / 4 |
| Accuracy | 0.99 | 1.00 | 1.00 | 0.99 |
| Precision | 0.96 | 1.00 | 0.97 | 1.00 |
| Recall | 0.93 | 0.97 | 1.00 | 0.86 |
| F1 | 0.95 | 0.98 | 0.98 | 0.93 |
| Correct repair rate | 0.90 | 0.93 | 0.93 | 0.72 |
| Hallucination rate | 0.04 | 0.00 | 0.03 | 0.00 |
| Self-check calls | 4 (1 batch + 3 single-case retries after an empty answer) | 1 | 2 | 2 |

How to read this:
- The **self-check raised every run to 100%**.
- **Qwen B's first pass was the weakest** (recall 0.86, repair 0.72), most likely because one first-pass batch hit Groq's "request too large" error and returned nothing. The self-check found and fixed those cases.
- Method A gave the best first-pass recall for Qwen. Method B gave the best first-pass precision for both models.
- Accuracy is high in every case partly because 471 of the 500 events were already correct, so precision, recall, F1 and repair rate are the more informative metrics.

## Files in this folder
| File | Content |
|---|---|
| `q6_metrics.csv` | all metrics, per method × LLM × stage (final / first pass) |
| `q6_first_pass.png` | first-pass precision, recall, F1 and repair rate per run (figure in the report) |
| `make_q6_figure.py` | creates the figure from the CSV: `python make_q6_figure.py` |

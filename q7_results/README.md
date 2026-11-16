# Q7: Time and cost of the timestamp correction

Seconds and tokens (reported by the Groq API) of the 4 runs of Q5 (see `../q5_results/README.md`). Free tier, so the cost is given in tokens.
Official runs of 14 Nov 2026. Written up in section 8 of the report.

## Results

| Run | Calls | Time (s) | Input tokens | Output tokens | Total tokens |
|---|---|---|---|---|---|
| GPT-OSS · A | 14 | 284.0 | 34,522 | 27,856 | 62,378 |
| GPT-OSS · B | 11 | 267.0 | 40,068 | 26,160 | 66,228 |
| Qwen · A | 12 | 45.6 | 36,386 | 5,884 | 42,270 |
| Qwen · B | 12 | 158.4 | 47,473 | 8,901 | 56,374 |

Split by stage:

| Run | First pass (s / in / out) | Self-check (s / in / out) |
|---|---|---|
| GPT-OSS · A | 188.1 / 27,600 / 18,777 | 96.0 / 6,922 / 9,079 |
| GPT-OSS · B | 251.3 / 38,597 / 23,704 | 15.7 / 1,471 / 2,456 |
| Qwen · A | 20.7 / 32,612 / 5,321 | 24.9 / 3,774 / 563 |
| Qwen · B | 155.8 / 42,968 / 7,866 | 2.7 / 4,505 / 1,035 |

- **Qwen is about 6× faster than GPT-OSS on method A** (46 s vs 284 s) and writes far fewer output tokens (5.9k vs 27.9k). GPT-OSS is a reasoning model, so it "thinks" before answering.
- **Method B sends more input tokens** than method A for both models (GPT-OSS 40k vs 35k, Qwen 47k vs 36k), because the Random Forest predictions are added to the prompt.
- **Qwen B took longer than Qwen A** (158 s vs 46 s). Most of the extra time was spent waiting for Groq's per-minute output limit.
- **For GPT-OSS, the first pass is most of the cost.** The self-check took 6% (B) to 34% (A) of the time. For Qwen A, the self-check was about half of a run that was short anyway (25 of 46 s).

Local steps without an LLM (0 tokens): method A statistics ≈ 0.15 s, method B Random Forest ≈ 3.6 s.

## Files in this folder
| File | Content |
|---|---|
| `q7_time_tokens.csv` | time and tokens per run and stage |
| `q7_time_tokens.png` | time and tokens per run (figure in the report) |
| `make_q7_figure.py` | creates the figure from the CSV: `python make_q7_figure.py` |

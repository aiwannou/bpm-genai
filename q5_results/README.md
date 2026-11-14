# Q5–Q7: Timestamp correction with two LLMs

**Code:** `app/q5_fix.py` (logic) · `app/q5_tab.py` (GUI tab "Fix timestamps") · `app/run_q5_all.py` (terminal runner)
**Version used for the official runs:** `Q5 final · prompt v4 + self-check · v6 (GPT-OSS + Qwen)`, run on 14 Nov 2026

## What Q5 does
The LLM gets the raw cases of the insurance-claims log and must **find and fix the wrong timestamps by itself**.
It is **not** told which events are wrong. The answer is scored against the 29 wrong events found in Q1
(E1 missing ×5, E2 malformed ×5, E3 negative duration ×5, E4 out of order: 7 swapped pairs = 14 events).

| | |
|---|---|
| **LLM 1** | `openai/gpt-oss-120b`: GPT-OSS 120B (OpenAI), via Groq |
| **LLM 2** | `qwen/qwen3.8-27b`: Qwen 3.8 27B (Alibaba), via Groq |
| **Method A: statistical** | median duration of each step, plus the typical (P5–P95) and observed (min–max) range, learned from the 78 internally consistent cases |
| **Method B: machine learning** | Random Forest regressor (300 trees, out-of-fold) that predicts each step's duration from activity, user, previous user, cost, hour and weekday |
| **Self-check** | A rule-based checker (valid value, correct order, same day, step length in the observed range, known hour kept) sends cases that break a rule back to the same LLM, for at most 2 rounds (Self-Refine idea, Madaan et al. 2023). The Q1 answer key is never used here. |

Llama 3.3 70B was planned as LLM 2, but Groq retired it from the free tier on 16 Aug 2026, so Qwen 3.8 27B was used instead.

## The 4 timestamp error types (from Q1)
| Type | What is wrong | Example in the log | How it must be fixed |
|---|---|---|---|
| **E1: missing** (5 events) | the timestamp is empty | Case_28 · Verify Documents · *(empty)* | put a time between the step before and the step after, using a typical step duration |
| **E2: malformed** (5 events) | only the date and hour, no minutes or seconds | Case_13 · Evaluate Claim · `2026-09-13T16` | keep the known hour (16:00–16:59) and choose minutes that fit the neighbouring steps |
| **E3: negative duration** (5 events) | Approve is about 15 min **before** Evaluate | Case_5 · Approve 13:55, but Evaluate 14:10 | move Approve to after Evaluate and before Close |
| **E4: out of order** (7 pairs = 14 events) | the Evaluate and Approve timestamps are **swapped** | Case_4 · Evaluate 14:22, Approve 12:27 | swap them back (Evaluate 12:27, Approve 14:22) |

Normal order of every case: Register Claim → Verify Documents → Evaluate Claim → Approve Claim → Close Claim. Each step normally takes 30–120 minutes.

### How each fix is scored
- **TP**: a wrong event that was changed. **FN**: a wrong event that was missed. **FP**: a correct event that was changed (should not happen). **TN**: a correct event left alone.
- **Correct repair**: the new value is a valid timestamp, it keeps the known hour (E2), and the whole case is back in the correct order on the same day.
- **Hallucination**: an invented event, an invalid value, or a correct event that was changed.

## How to run it yourself (step by step)
Everything needed is in the repo: the code in `app/`, the event log `app/insurance_claims_log.csv`, and `app/requirements.txt`.

**1. Install (once).** Needs Python 3.10 or newer.
```
cd app
pip install -r requirements.txt
```

**2. Groq API key (free, no card).** Get one at console.groq.com → API Keys. Then copy
`app/.streamlit/secrets.toml.example` to `app/.streamlit/secrets.toml` and paste the key:
```
GROQ_API_KEY = "gsk_..."
```
⚠️ Never upload `secrets.toml` to GitHub; it holds your private key.

**3. Start the app.**
```
python -m streamlit run app.py
```
The browser opens. Go to the tab **"🛠️ Fix timestamps (Q5)"**:
1. Choose the **LLM** (GPT-OSS 120B or Qwen 3.8 27B) and the **method** (A or B). Leave "Cases per call" at 10 and "Self-check rounds" at 2.
2. Click **Run correction**. It takes about 1–5 minutes.
3. Check the **error analysis**: 29 fixed correctly · 0 missed · 0 bad repair · 0 changed a correct event.
4. Repeat for the other combinations **without refreshing the page**. The Q6 and Q7 tables at the bottom then fill with all runs.

**No key?** Turn on **Demo mode** in the same tab. It uses rule-based answers instead of an LLM and costs no tokens, so you can see the whole GUI.

**Free-tier limits:** each model allows about 200k tokens per day, and one run uses about 40–70k. So all 4 runs fit in one day.

**Without the GUI:** `python run_q5_all.py` (inside `app/`) runs all 4 combinations and writes the CSVs to `app/q5_output/`.

## Q6: results (final, after the self-check)
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

## Q7: time and tokens

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

All of these numbers are in `q7_time_tokens.csv`. The app also shows them at the bottom of the "Fix timestamps" tab and in the "Usage & cost" tab.

## ⚠️ Red or yellow messages during a run: please DON'T change the code
These messages come from **Groq's free tier**, not from a bug. The code handles them automatically.
**The only thing that matters is the error analysis under the results:** if it shows
**29 fixed correctly · 0 missed · 0 bad repair · 0 changed a correct event**, the run is complete and valid.

| Message you may see | What it means | What the code does |
|---|---|---|
| `1 of N calls failed: empty answer from the model (no JSON)` | Groq sometimes returns an empty reply | asks again with smaller batches; the self-check catches anything left |
| `Error code: 429 - Request too large ... output tokens per minute (OTPM)` | Qwen's free tier allows only 1,000 output tokens per minute | that batch is skipped, then the self-check fixes those cases |
| `Error code: 429 - Rate limit reached ... tokens per minute (TPM)` | too many tokens in one minute | waits exactly as long as Groq says, then continues |
| `Failed to validate JSON` | the model's JSON had a small format error | asks again without strict JSON mode |
| `Groq's DAILY limit for this model is used up` | 200k tokens/day per model on the free tier | the run stops; **try again the next day**. Results of a stopped run are not valid. |
| `The Groq API key was refused (401)` | wrong key in `secrets.toml` | fix the key |
| `... is not available for this API key` | Groq has retired that model | choose the other model, and don't add Llama back |

What happened in the official runs (14 Nov 2026): GPT-OSS A had 1 empty answer (1 of 14 calls), and Qwen B had 1 "request too large" (1 of 12 calls).
Both were recovered, and **all 4 runs still fixed 29/29 errors**.

Other things that are **normal**:
- **First pass below 100%:** the first answer is a blind guess; the self-check rounds fix the rest. This is the point of the self-check, not a bug.
- **Slightly different numbers on a re-run:** LLMs are not fully deterministic.
- **The Q6 tables say "not run yet":** the tables only remember runs since the page was opened. Refreshing the page clears them.
- **In the fixed log, a swapped pair keeps its original row position:** the timestamps are correct, and process-mining tools sort by time.

**Please don't edit `q5_fix.py` or `q5_tab.py`:** the reported results were produced with exactly this version (v6).

## Files in this folder
| File | Content |
|---|---|
| `q6_metrics.csv` | all metrics, per method × LLM × stage (final / first pass) |
| `q7_time_tokens.csv` | time and tokens per run and stage (Q7) |
| `fixed_log_A_gpt-oss.csv`, `fixed_log_B_gpt-oss.csv`, `fixed_log_A_qwen.csv`, `fixed_log_B_qwen.csv` | the corrected event log of each run (input for rerunning Q2) |

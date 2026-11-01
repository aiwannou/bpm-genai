# BPM-GenAI assignment (MHYP 521)

Insurance-claims event log (100 cases, 500 events) analysed with process mining
(PM4PY) and an LLM assistant (Streamlit + Groq).

| Question | Status | Where |
| --- | --- | --- |
| Q1 Timestamp errors | Done | `q1_q2/q1_timestamp_errors.py` |
| Q2 Petri nets (Alpha / Heuristic / Inductive) | Done | `q1_q2/q2_petri_nets.py` |
| Q3 LLM assistant (questions → table + chart) | Done | `app/` (tab "Ask the log") |
| Q4 Trace summary with enhancement | Done | `app/` (tab "Trace summary") |
| Q5 Timestamp correction, 2 LLMs × 2 methods | **To do** | `app/` (tab "Fix timestamps" is a placeholder) |
| Q6 Evaluation tables (2×2 per metric) | **To do** | ground truth: `q1_q2/q1_error_events.csv` |
| Q7 Time and token cost | **To do** | the app already logs seconds + tokens per call (tab "Usage & cost") |
| Deploy to Streamlit Cloud + link in report | **To do** (after Q5) | |

## Folder structure

```
bpm-genai/
├── README.md
├── report/                     report (Q1–Q4 written, Q5–Q7 empty sections)
├── q1_q2/                      Questions 1 and 2 (run in the terminal)
│   ├── q1_timestamp_errors.py  finds the timestamp errors (no LLM)
│   ├── q2_petri_nets.py        discovers + evaluates the Petri nets
│   ├── insurance_claims_log.csv / .xes
│   ├── q1_error_events.csv     GROUND TRUTH: the 29 wrong events (for Q5/Q6)
│   ├── q1_summary.csv, q1_output.txt, q2_output.txt   saved results
│   ├── q2_output/              Petri net images + q2_metrics.csv
│   └── q2_analysis_notes.md    notes used for the report
└── app/                        Questions 3, 4 (and 5) – Streamlit web app
    ├── app.py                  GUI (tabs)
    ├── assistant.py            Q3 + Q4 logic (prompts, safe code execution)
    ├── data.py                 builds the events/cases tables + schema for the LLM
    ├── llm.py                  Groq API wrapper (measures seconds + tokens)
    ├── q1_timestamp_errors.py  same file as in q1_q2 (reused for error flags)
    └── .streamlit/secrets.toml.example
```

## How to run (Mac / Linux)

Python 3.10+ is needed. Each folder has its own virtual environment.

### Questions 1 and 2

```
cd q1_q2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 q1_timestamp_errors.py          # add insurance_claims_log.xes to run on the XES
python3 q2_petri_nets.py                 # needs Graphviz for the images: brew install graphviz
```

Expected: Q1 prints "Cases with at least one error: 22 / 100" and saves
`q1_error_events.csv` (29 events). Q2 prints a table with 8 rows and saves
12 images in `q2_output/`.

### The app (Questions 3–5)

```
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
open -e .streamlit/secrets.toml          # paste YOUR Groq key: GROQ_API_KEY = "gsk_..."
streamlit run app.py
```

Get a free Groq key (no credit card) at console.groq.com → API Keys.
**Each person uses their own key. Never commit `secrets.toml` to GitHub**
(it is listed in `.gitignore`).

## Notes for Questions 5–7

- **Ground truth:** `q1_q2/q1_error_events.csv` lists the 29 wrong events
  (E1 missing 5, E2 malformed 5, E3 negative duration 5 = the Approve event only,
  E4 out of order 14 = both events of each swapped pair).
- **Useful facts from Q1:** every normal step takes 30–120 min; in E3 the
  Approve timestamp is the wrong one (moved 15 min before Evaluate); in E4 the
  Evaluate/Approve timestamps are swapped.
- **Two LLMs on one free key:** Groq hosts models from different companies,
  e.g. `openai/gpt-oss-120b` (OpenAI) and `llama-3.3-70b-versatile` (Meta).
  `llm.py` → `GroqLLM(api_key, model).generate(system, prompt, json_mode)` returns
  text + seconds + input/output tokens.
- **Data tables:** `data.build_tables()` returns `events` and `cases` with the
  columns `data_issue`, `ts_reliable`, `duration_min`, `duration_reliable`.
- **Q7:** every LLM call made through the app is listed in the tab
  "Usage & cost" (downloadable CSV).
- **After Q5 (optional):** rerun Q2 on the corrected log:
  `python3 q2_petri_nets.py insurance_claims_log.xes fixed_log.csv`
  (CSV with columns CaseID, Activity, Timestamp).

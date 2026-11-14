"""Run all 4 combinations (2 LLMs x 2 methods) from the terminal and save the Q6/Q7 tables.

    cd app
    python run_q5_all.py            (reads the key from .streamlit/secrets.toml, like the app)
Outputs in q5_output/: corrections_*.csv, error_analysis_*.csv, fixed_log_*.csv,
         q6_tables.md, q6_metrics.csv, q7_usage.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd

import q5_fix as q
from llm import GroqLLM  # the app's Groq wrapper

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "insurance_claims_log.csv")
out = Path(__file__).resolve().parent / "q5_output"
out.mkdir(exist_ok=True)
df = q.load_log(path)
gt = q.ground_truth(df)
def read_key():
    """Same places as the app: .streamlit/secrets.toml next to this file, then the environment."""
    import tomllib
    f = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if f.exists():
        k = tomllib.loads(f.read_text(encoding="utf-8-sig")).get("GROQ_API_KEY", "")
        if k:
            return k.strip()
    k = os.environ.get("GROQ_API_KEY", "")
    if not k:
        sys.exit("No Groq key: put GROQ_API_KEY = \"gsk_...\" in app/.streamlit/secrets.toml")
    return k


key = read_key()

results, calls, rows = {}, [], []
for method in q.METHODS:
    # non-LLM reference with the same hints
    ref = q.run_rules(df, method)
    rows.append({"stage": "reference", "method": method, "model": "rules (no LLM)", **q.evaluate(df, gt, ref)})
    for label, model_id in q.MODELS.items():
        print(f"Method {method} - {label} ...", flush=True)
        r = q.run_llm(df, GroqLLM(key, model_id), label, method, batch_size=10, pause=8, rounds=2)
        if r.stopped:
            print("  !!", r.stopped)
        tag = f"{method}_{model_id.split('/')[-1]}"
        r.corrections.to_csv(out / f"corrections_{tag}.csv", index=False)
        q.diagnose(df, gt, r.corrections).to_csv(out / f"error_analysis_{tag}.csv", index=False)
        q.apply_corrections(df, r.corrections)[["CaseID", "Activity", "Timestamp", "User", "Cost"]] \
            .to_csv(out / f"fixed_log_{tag}.csv", index=False)
        results[(method, label)] = q.evaluate(df, gt, r.corrections)
        rows.append({"stage": "final (after self-check)", "method": method, "model": label,
                     **results[(method, label)]})
        rows.append({"stage": "first pass", "method": method, "model": label, **q.evaluate(df, gt, r.first)})
        calls.append(r.calls)

pd.DataFrame(rows).to_csv(out / "q6_metrics.csv", index=False)
usage = q.usage_summary(pd.concat(calls))
usage.to_csv(out / "q7_usage.csv", index=False)
with open(out / "q6_tables.md", "w") as f:
    for m in ["Accuracy", "Precision", "Recall", "F1", "Correct repair rate", "Hallucination rate"]:
        f.write(f"\n### {m}\n\n{q.table_2x2(results, m).round(3).to_markdown()}\n")
    f.write(f"\n### Q7 usage\n\n{usage.to_markdown(index=False)}\n")
print((out / "q6_tables.md").read_text())

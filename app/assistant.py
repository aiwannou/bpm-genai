"""
Logic of Question 3 (questions about the log) and Question 4 (trace summary).

Q3 - text-to-pandas:
    1. The LLM receives the question + the description of the tables and
       returns JSON: pandas code + a chart specification.
    2. The code is checked (no imports, no file/system access) and executed
       on the real data. The numbers therefore come from the data, not from
       the LLM, which avoids hallucinated results.
    3. If the code fails, the error is sent back to the LLM to fix it.
    4. A second short call turns the result table into a sentence.

Q4 - trace summary with enhancement:
    All facts (users, durations, costs, bottleneck, comparison with the
    typical duration, data-quality issues) are computed in Python and given
    to the LLM, which only writes the narrative.
"""

import ast
import json
import re

import numpy as np
import pandas as pd

from data import SCHEMA_DESCRIPTION

MAX_FIX_ATTEMPTS = 2

# ----------------------------------------------------------------------
# Q3: prompts
# ----------------------------------------------------------------------
Q3_SYSTEM = f"""You are a process-mining data analyst. You answer questions about an
insurance-claims event log by writing Python pandas code.
{SCHEMA_DESCRIPTION}
Rules for the code:
- Use only the variables `events`, `cases`, `pd` and `np`. They already exist.
- Do NOT import anything, do NOT read or write files, do NOT print.
- Do NOT use .query() or .eval(); filter with boolean masks instead.
- Do not modify `events` or `cases` in place; work on copies or filtered views.
- Store the final answer in a variable called `result`, as a pandas DataFrame with
  clear column names (a few rows/columns, ready to show to a manager).
- Percentages must be numbers between 0 and 100, rounded to 1 decimal.
- Durations in minutes, rounded to 1 decimal.
- Always compute the answer from the data with filters/aggregations; never hard-code
  a result, not even an empty one.
- If the question mentions a value that does not exist in the data (e.g. a user or case
  that is not in the log), return an EMPTY DataFrame with the expected columns and
  explain it in "note". Never invent data.

Answer ONLY with a JSON object of this form:
{{
  "code": "<python code that sets result>",
  "chart": {{"type": "bar" | "line" | "pie" | "scatter" | "none",
             "x": "<column of result>", "y": "<column of result>", "title": "<short title>"}},
  "note": "<one sentence on assumptions, e.g. which rows were excluded>"
}}"""

Q3_FIX = """The code you wrote failed.
Question: {question}
Code:
{code}
Error: {error}
Return the corrected JSON object in the same format."""

Q3_ANSWER_SYSTEM = """You explain data-analysis results to a business user in plain English.
Use ONLY the numbers in the table you are given. Do not add facts that are not in the
table. Never generalize about all rows (e.g. "all claims followed ...") unless every row
of the table shows it. If the table is only a preview of a longer result, do not make
claims about the rows you did not see. If the analyst note says rows were excluded,
mention it. 2-4 sentences, no markdown headings."""

Q3_ANSWER = """Question: {question}
Analyst note: {note}
Result table ({n} rows, first rows shown as CSV):
{table}"""


# ----------------------------------------------------------------------
# Safe execution of LLM-generated code
# ----------------------------------------------------------------------
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                   "getattr", "setattr", "delattr", "input", "breakpoint", "exit",
                   "quit", "vars", "dir", "help", "memoryview", "os", "sys",
                   "subprocess", "shutil", "pathlib", "builtins"}
FORBIDDEN_ATTR = re.compile(
    r"^(_.*|read_.*|to_(csv|pickle|excel|json|html|sql|parquet|feather|hdf|clipboard|"
    r"latex|markdown|stata|xml|orc|gbq)|eval|query|system|popen|remove|unlink|rmdir)$")
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in ("len", "range", "min", "max", "sum", "sorted", "round", "abs",
                           "list", "dict", "set", "tuple", "str", "int", "float", "bool",
                           "enumerate", "zip", "any", "all", "isinstance", "map", "filter",
                           "reversed", "ValueError", "KeyError", "Exception")}


class UnsafeCode(Exception):
    pass


def check_code(code):
    """Reject code that imports modules or touches files / the system."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise UnsafeCode("imports are not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"'{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and FORBIDDEN_ATTR.match(node.attr):
            raise UnsafeCode(f"'.{node.attr}' is not allowed")


def run_code(code, events, cases):
    check_code(code)
    env = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np,
           "events": events.copy(), "cases": cases.copy()}
    exec(compile(code, "<llm_code>", "exec"), env)          # noqa: S102 (checked above)
    if "result" not in env:
        raise ValueError("the code did not set a variable called `result`")
    return to_frame(env["result"])


def to_frame(res):
    if isinstance(res, pd.DataFrame):
        return res.reset_index(drop=not _has_meaningful_index(res))
    if isinstance(res, pd.Series):
        return res.reset_index()
    return pd.DataFrame({"value": [res]})


def _has_meaningful_index(df):
    """Keep the index only if it carries information (e.g. after a groupby),
    not if it is just the row numbers left over from a filter."""
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        return True
    return not pd.api.types.is_integer_dtype(df.index)


def parse_json(text):
    """Parse the JSON answer of the LLM (tolerates ```json fences)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


# ----------------------------------------------------------------------
# Q3: main function
# ----------------------------------------------------------------------
def ask_log(llm, question, events, cases):
    """Returns a dict with code, chart spec, result table, answer and the LLM calls."""
    calls = []
    out = {"question": question, "code": None, "chart": None, "note": "",
           "result": None, "answer": "", "error": None, "calls": calls, "attempts": 0,
           "failed_attempts": []}

    r = llm.generate(Q3_SYSTEM, f"Question: {question}", json_mode=True)
    calls.append(("q3_code", r))
    for attempt in range(MAX_FIX_ATTEMPTS + 1):
        out["attempts"] = attempt + 1
        try:
            spec = parse_json(r.text)
            out["code"] = spec.get("code", "")
            out["chart"] = spec.get("chart") or {"type": "none"}
            out["note"] = spec.get("note", "")
            out["result"] = run_code(out["code"], events, cases)
            out["error"] = None
            break
        except Exception as e:                          # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            out["failed_attempts"].append({"code": out["code"], "error": out["error"]})
            if attempt == MAX_FIX_ATTEMPTS:
                return out
            r = llm.generate(Q3_SYSTEM, Q3_FIX.format(
                question=question, code=out["code"] or r.text, error=out["error"]),
                json_mode=True)
            calls.append(("q3_fix", r))

    res = out["result"]
    shown = min(len(res), 30)
    table = res.head(30).to_csv(index=False) if len(res) else "(empty table)"
    if len(res) > shown:
        table += f"\n(preview: only the first {shown} of {len(res)} rows are shown)"
    a = llm.generate(Q3_ANSWER_SYSTEM, Q3_ANSWER.format(
        question=question, note=out["note"], n=len(res), table=table))
    calls.append(("q3_answer", a))
    out["answer"] = a.text.strip()
    return out


def make_chart(result, chart):
    """Build a Plotly figure from the chart spec of the LLM (None if not possible)."""
    import plotly.express as px
    if result is None or result.empty or not chart or chart.get("type", "none") == "none":
        return None
    x, y, kind = chart.get("x"), chart.get("y"), chart.get("type")
    cols = list(result.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(result[c])]
    if x not in cols:
        x = next((c for c in cols if c not in num), cols[0])
    if y not in cols:
        y = next((c for c in num if c != x), None)
    if y is None:
        return None
    title = chart.get("title", "")
    if kind == "pie":
        return px.pie(result, names=x, values=y, title=title)
    if kind == "line":
        return px.line(result, x=x, y=y, title=title, markers=True)
    if kind == "scatter":
        return px.scatter(result, x=x, y=y, title=title)
    return px.bar(result, x=x, y=y, title=title, text_auto=True)


# ----------------------------------------------------------------------
# Q4: trace summary
# ----------------------------------------------------------------------
Q4_SYSTEM = """You are a claims-process analyst. You turn the event trace of ONE insurance
claim into a short narrative summary for a manager, enriched with performance information
(enhancement), in the style of:
"This claim was registered by User A, verified within 2 hours, but was under evaluation for
5 days, costing the company $450 in operational overhead. The customer experienced a
bottleneck at stage 3."

Rules:
- Use ONLY the facts in the JSON you receive. Never invent numbers, users, dates or
  time zones (the log has no time zone, so never write "UTC").
- Mention who did each step, how long the important steps took, the total cost, the total
  cycle time and the bottleneck (the step that took longest compared with its typical duration).
- Compare with the typical (median) duration when it is useful ("twice as long as usual").
- If the JSON lists data-quality issues, say clearly which durations are unreliable and why.
- One paragraph of 3-6 sentences, then one line starting with "Recommendation:".
"""


def fmt_minutes(m):
    if m is None or pd.isna(m):
        return None
    m = int(round(m))
    h, mm = divmod(m, 60)
    return f"{h} h {mm} min" if h else f"{mm} min"


def trace_facts(case_id, events, cases):
    """All numbers for the summary, computed in Python."""
    ev = events[events["CaseID"] == case_id]
    if ev.empty:
        return None
    case = cases[cases["CaseID"] == case_id].iloc[0]
    typical = (events[events["duration_reliable"]]
               .groupby("Activity")["duration_min"].median())
    steps = []
    for _, r in ev.iterrows():
        reliable = bool(r["duration_reliable"])
        dur = r["duration_min"] if reliable else None
        med = typical.get(r["Activity"])
        steps.append({
            "step": int(r["step_no"]),
            "activity": r["Activity"],
            "user": r["User"],
            "completed_at": r["timestamp_raw"] if isinstance(r["timestamp_raw"], str) else None,
            "duration_since_previous_step": fmt_minutes(dur),
            "duration_minutes": None if dur is None else round(float(dur), 1),
            "typical_duration": fmt_minutes(med) if r["step_no"] > 1 else None,
            "times_typical": (round(float(dur / med), 2)
                              if dur is not None and med and med > 0 else None),
            "cost_usd": int(r["Cost"]),
            "data_issue": r["data_issue"] or None,
        })
    timed = [s for s in steps if s["times_typical"] is not None]
    bottleneck = max(timed, key=lambda s: s["times_typical"]) if timed else None
    longest = max(timed, key=lambda s: s["duration_minutes"]) if timed else None
    avg_cost = cases["total_cost"].mean()
    return {
        "case_id": case_id,
        "sequence": case["sequence"],
        "steps": steps,
        "total_cost_usd": int(case["total_cost"]),
        "average_claim_cost_usd": round(float(avg_cost), 0),
        "cycle_time": fmt_minutes(case["cycle_time_min"]),
        "start": str(case["start"]), "end": str(case["end"]),
        "bottleneck_step": (f"step {bottleneck['step']} ({bottleneck['activity']}), "
                            f"{bottleneck['times_typical']}x its typical duration")
        if bottleneck else None,
        "longest_step": (f"step {longest['step']} ({longest['activity']}), "
                         f"{longest['duration_since_previous_step']}") if longest else None,
        "data_quality_issues": case["issues"] or None,
        "note_on_durations": "Durations are the time between the previous event and this "
                             "one (waiting + processing). Unreliable durations are null.",
    }


def summarize_trace(llm, case_id, events, cases):
    facts = trace_facts(case_id, events, cases)
    if facts is None:
        return None, None, []
    r = llm.generate(Q4_SYSTEM, "Trace facts (JSON):\n" + json.dumps(facts, indent=1))
    return facts, r.text.strip(), [("q4_summary", r)]

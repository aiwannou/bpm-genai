"""
MHYP 521 - BPM-GenAI assignment
Digital assistant for the insurance-claims event log (Questions 3, 4, 5).

Run locally:   streamlit run app.py
The Groq API key is read ONLY from .streamlit/secrets.toml (GROQ_API_KEY = "...")
locally, or from the app's Secrets on Streamlit Community Cloud. It is never typed
into or shown by the app.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from assistant import ask_log, make_chart, summarize_trace, trace_facts
from data import build_tables
from llm import GroqLLM, find_working_model, friendly_error

st.set_page_config(page_title="Claims Process Assistant", page_icon="📋", layout="wide")


# ----------------------------------------------------------------------
# Data and LLM setup
# ----------------------------------------------------------------------
@st.cache_data
def load_tables():
    return build_tables()


@st.cache_data(show_spinner=False, ttl=3600)
def available_models(api_key):
    return GroqLLM.list_models(api_key)


def get_api_key(name):
    try:
        key = st.secrets.get(name, "")
    except Exception:                                   # no secrets.toml file
        key = ""
    return key or os.environ.get(name, "")


def md_safe(text):
    """Streamlit renders $...$ as LaTeX; escape dollar signs in LLM text."""
    return (text or "").replace("$", "\\$")


def log_calls(tab, calls):
    """Store every LLM call (time + tokens) for Question 7."""
    for step, r in calls:
        st.session_state.usage.append({
            "tab": tab, "step": step, "model": r.model,
            "seconds": round(r.seconds, 2),
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
        })


if "usage" not in st.session_state:
    st.session_state.usage = []

events, cases = load_tables()

# ---- sidebar ----
with st.sidebar:
    st.header("Settings")
    api_key = get_api_key("GROQ_API_KEY")
    llm = None
    if not api_key:
        st.error("No API key found. Add the line GROQ_API_KEY = \"gsk_...\" to the file "
                 ".streamlit/secrets.toml and restart the app.")
    else:
        try:
            models = available_models(api_key)
            if st.button("🔍 Find a working model", width="stretch",
                         help="Tries the models one by one with a tiny request and "
                              "selects the first one that works with your key."):
                with st.spinner("Testing models..."):
                    found, tries = find_working_model(GroqLLM, api_key, models)
                for name, status in tries:
                    st.caption(f"{'✅' if status == 'works' else '❌'} {name}: {status}")
                if found:
                    st.session_state.model = found
                else:
                    st.error("No model worked. The free quota may be used up; "
                             "try again later or check the key.")
            default = st.session_state.get("model") or GroqLLM.default_model(models)
            model = st.selectbox("Model (via Groq)", models,
                                 index=models.index(default) if default in models else 0)
            st.session_state.model = model
            llm = GroqLLM(api_key, model)
            st.success(f"Connected: {model}")
        except Exception as e:                          # noqa: BLE001
            st.error(f"Could not connect to Groq: {friendly_error(e)}")

    st.divider()
    st.caption("Event log")
    st.write(f"{len(cases)} claims · {len(events)} events · "
             f"{int(cases['has_data_issue'].sum())} claims with timestamp errors")

st.title("📋 Insurance Claims Process Assistant")
st.caption("MHYP 521 · BPM-GenAI assignment · ask questions about the event log in plain English")

tab3, tab4, tab5, tab_data, tab_usage = st.tabs(
    ["💬 Ask the log (Q3)", "🧾 Trace summary (Q4)", "🛠️ Fix timestamps (Q5)",
     "🗂️ Data", "⏱️ Usage & cost"])

# ----------------------------------------------------------------------
# Q3 - questions about the log
# ----------------------------------------------------------------------
EXAMPLES = [
    "Show me all claims handled by User_42 that exceeded $500",
    "Show me all claims handled by User_C that exceeded $1500",
    "Which activities have the longest waiting times and what is their percentage over the total?",
    "What is the total cost per user?",
    "How many claims were registered each month?",
    "Which user is involved in the most expensive claims on average?",
]

with tab3:
    st.subheader("Ask a question about the claims")
    st.write("The LLM writes pandas code, the code runs on the real data, and the result "
             "is shown as a table and a chart.")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.q3_question = ex
    question = st.text_area("Your question", key="q3_question", height=80,
                            placeholder="e.g. Which user handled the most claims?")
    if st.button("Ask", type="primary", disabled=llm is None or not question.strip()):
        with st.spinner("Thinking..."):
            try:
                st.session_state.q3_out = ask_log(llm, question.strip(), events, cases)
                log_calls("Q3", st.session_state.q3_out["calls"])
            except Exception as e:                      # noqa: BLE001
                st.error(friendly_error(e))

    out = st.session_state.get("q3_out")
    if out:
        st.markdown(f"**Question:** {out['question']}")
        if out["error"]:
            st.error(f"The generated code could not run after {out['attempts']} attempts: "
                     f"{out['error']}")
        else:
            st.info(md_safe(out["answer"]))
            res = out["result"]
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Result** ({len(res)} rows)")
                st.dataframe(res, width="stretch", hide_index=True)
            with c2:
                fig = make_chart(res, out["chart"])
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")
                elif res.empty:
                    st.warning("No matching data, so there is nothing to plot.")
            if out["note"]:
                st.caption(md_safe(f"Note from the analyst: {out['note']}"))
        with st.expander("Show generated code"):
            st.code(out["code"] or "", language="python")
            for i, fa in enumerate(out.get("failed_attempts", []), 1):
                if fa["code"] != out["code"] or out["error"]:
                    st.caption(f"Attempt {i} failed: {fa['error']}")
                    st.code(fa["code"] or "", language="python")
            secs = sum(r.seconds for _, r in out["calls"])
            toks = sum(r.input_tokens + r.output_tokens for _, r in out["calls"])
            st.caption(f"{len(out['calls'])} LLM calls · {secs:.1f} s · {toks} tokens · "
                       f"{out['attempts']} attempt(s)")

# ----------------------------------------------------------------------
# Q4 - trace summary
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Summarize one claim")
    st.write("All numbers are computed from the log; the LLM only writes the narrative.")
    case_ids = cases["CaseID"].tolist()
    c1, c2 = st.columns([2, 1])
    case_id = c1.selectbox("Case", case_ids, index=case_ids.index("Case_3"))
    typed = c2.text_input("…or type a case id", placeholder="Case_77")
    if typed.strip():
        case_id = typed.strip()

    facts = trace_facts(case_id, events, cases)
    if facts is None:
        st.error(f"{case_id} does not exist in the log (valid: Case_1 … Case_100).")
    else:
        ev = events[events["CaseID"] == case_id].copy()
        st.markdown(f"**{case_id}** · {facts['sequence']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total cost", f"${facts['total_cost_usd']:,}",
                  f"{facts['total_cost_usd'] - facts['average_claim_cost_usd']:+,.0f} vs average",
                  delta_color="inverse")
        m2.metric("Cycle time", facts["cycle_time"] or "unreliable")
        m3.metric("Bottleneck", (facts["bottleneck_step"] or "n/a").split(",")[0])
        if facts["data_quality_issues"]:
            st.warning(f"Timestamp errors in this case: {facts['data_quality_issues']}")

        st.dataframe(pd.DataFrame(facts["steps"]), width="stretch", hide_index=True)

        tl = ev.dropna(subset=["ts"]).copy()
        tl["start"] = tl["ts"] - pd.to_timedelta(tl["duration_min"].where(
            tl["duration_reliable"], 0).fillna(0), unit="m")
        tl = tl[tl["start"] < tl["ts"]]
        if len(tl):
            fig = px.timeline(tl, x_start="start", x_end="ts", y="Activity", color="User",
                              title="Timeline (time since previous step)",
                              category_orders={"Activity": ev["Activity"].tolist()})
            st.plotly_chart(fig, width="stretch")

        if st.button("Generate summary", type="primary", disabled=llm is None):
            with st.spinner("Writing the summary..."):
                try:
                    _, text, calls = summarize_trace(llm, case_id, events, cases)
                    st.session_state.q4_out = (case_id, text)
                    log_calls("Q4", calls)
                except Exception as e:                  # noqa: BLE001
                    st.error(friendly_error(e))
        q4 = st.session_state.get("q4_out")
        if q4 and q4[0] == case_id:
            st.success(md_safe(q4[1]))
        with st.expander("Facts sent to the LLM (JSON)"):
            st.json(facts)

# ----------------------------------------------------------------------
# Q5 - placeholder
# ----------------------------------------------------------------------
with tab5:
    st.subheader("Fix the timestamps with two LLMs")
    st.info("Coming next: statistical and machine-learning correction of the timestamp "
            "errors with two LLMs, evaluated against the Question 1 ground truth.")

# ----------------------------------------------------------------------
# Data and usage tabs
# ----------------------------------------------------------------------
with tab_data:
    st.subheader("Events")
    st.dataframe(events, width="stretch", hide_index=True)
    st.subheader("Cases")
    st.dataframe(cases, width="stretch", hide_index=True)

with tab_usage:
    st.subheader("LLM calls in this session (for Question 7)")
    usage = pd.DataFrame(st.session_state.usage)
    if usage.empty:
        st.write("No LLM calls yet.")
    else:
        st.dataframe(usage, width="stretch", hide_index=True)
        tot = usage.groupby(["tab", "model"])[["seconds", "input_tokens", "output_tokens"]].sum()
        st.dataframe(tot, width="stretch")
        st.download_button("Download CSV", usage.to_csv(index=False),
                           "llm_usage.csv", "text/csv")

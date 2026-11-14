"""Streamlit tab for Question 5 (timestamp correction) + the Q6/Q7 tables.

Used in app.py:
    from q5_tab import render_q5_tab
    with tab5:
        render_q5_tab(raw_df, make_llm, models, on_calls)

raw_df    : the CSV read with dtype=str (timestamps kept as text)
make_llm  : function model_id -> GroqLLM (None when there is no API key)
models    : model ids available for the key (to warn if one of the two is missing)
on_calls  : callback that receives the calls DataFrame (logged in "Usage & cost")
"""
import json
import time

import pandas as pd
import plotly.express as px
import streamlit as st

import q5_fix as q

METRICS = ["Accuracy", "Precision", "Recall", "F1", "Correct repair rate", "Hallucination rate"]


class DemoLLM:
    """Fake LLM that returns the rule-based answer: lets you test the GUI without tokens."""

    def __init__(self, df):
        self.df = df

    def generate(self, system, prompt, json_mode=True):
        cases = [line[:-1] for line in prompt.splitlines() if line.startswith("Case_")]
        method = "B" if "ml_gap" in prompt else "A"
        r = q.run_rules(self.df, method)
        r = r[r.case.isin(cases)].assign(reason="demo (rule-based, no LLM)")
        return {"text": json.dumps({"corrections": r.to_dict("records")}), "seconds": 0.0,
                "input_tokens": 0, "output_tokens": 0}


def _timeline(df_before, df_after, case):
    a = df_before[df_before.CaseID == case].assign(version="original", t=lambda d: d.ts)
    b = df_after[df_after.CaseID == case].copy()
    b["t"] = pd.to_datetime(b.Timestamp.where(b.Timestamp.str.match(q.TS_RE)), errors="coerce")
    b["version"] = "corrected"
    both = pd.concat([a[["Activity", "t", "version"]], b[["Activity", "t", "version"]]])
    fig = px.scatter(both, x="t", y="Activity", color="version", symbol="version",
                     category_orders={"Activity": q.ORDER[::-1]}, height=280,
                     title=f"{case}: original vs corrected timestamps", labels={"t": "Time"})
    fig.update_traces(marker_size=12)
    return fig


def render_q5_tab(raw_df, make_llm, models=None, on_calls=None, pause=8):
    st.subheader("Fix the timestamps with two LLMs")
    st.caption(f"Version: **{getattr(q, 'VERSION', 'OLD q5_fix.py - replace it')}**")
    st.write("Two LLMs × two methods. The LLM is **not** told which events are wrong: it receives "
             "the raw cases plus the hints of the chosen method, finds the errors and returns "
             "corrected timestamps as JSON. The result is scored against the 29 wrong events of Q1.")
    df = q.load_log(raw_df)
    gt = q.ground_truth(df)
    ss = st.session_state.setdefault("q5_runs", {})

    c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
    model_label = c1.selectbox("LLM", list(q.MODELS), key="q5_model")
    method = c2.radio("Method", list(q.METHODS), key="q5_method", horizontal=True,
                      format_func=lambda m: f"{m}: {q.METHODS[m]}")
    batch = c3.number_input("Cases per call", 5, 25, 10, key="q5_batch")
    rounds = c4.number_input("Self-check rounds", 0, 2, 2, key="q5_rounds",
                             help="After the first answer, cases that break a process rule are sent back "
                                  "to the same LLM with the problem in words (0 = first answer only).")
    model_id = q.MODELS[model_label]
    demo = st.toggle("Demo mode (no LLM, no tokens: rule-based answers to test the GUI)",
                     value=make_llm is None, disabled=make_llm is None, key="q5_demo")
    if models and model_id not in models and not demo:
        st.warning(f"{model_id} is not available for this API key.")

    with st.expander("What the LLM receives (hints of this method)"):
        if method == "A":
            st.markdown("Duration of each step (minutes since the previous step), whole log:")
            st.dataframe(q.stats_hints(df)[1], width="stretch")
        else:
            info = q.ml_info(df)
            st.markdown(f"Random Forest on {info['train_rows']} steps · out-of-fold MAE "
                        f"**{info['MAE_min']:.1f} min** (median baseline "
                        f"{info['baseline_MAE_min']:.1f} min)")
            st.json(info["top_features"])
        st.code(q.build_prompt(q.hints(df, method), ["Case_4", "Case_5"], method), language="text")

    if st.button("Run correction", type="primary", key="q5_run"):
        llm = DemoLLM(df) if demo else make_llm(model_id)
        bar = st.progress(0.0, "Calling the LLM...")
        t0 = time.time()
        res = q.run_llm(df, llm, model_label + (" [demo]" if demo else ""), method, int(batch),
                        pause=0 if demo else pause, rounds=int(rounds),
                        progress=lambda p: bar.progress(p, f"{p:.0%} done"))
        bar.empty()
        ss[(method, model_label)] = res
        if on_calls is not None and not demo:
            c = res.calls.copy()
            c["batch"] = c["stage"] + " " + c["batch"].astype(str)
            on_calls(c.assign(model_id=model_id))
        if res.stopped:
            st.error(res.stopped)
        failed = (res.calls.error != "").sum()
        if failed:
            st.error(f"{failed} of {len(res.calls)} calls failed: {res.calls.error[res.calls.error != ''].iloc[0]}")
        st.toast(f"Done in {time.time() - t0:.0f} s")

    res = ss.get((method, model_label))
    if res is not None:
        ev = q.evaluate(df, gt, res.corrections)
        first = getattr(res, "first", None)
        ev1 = q.evaluate(df, gt, first) if first is not None else ev
        k = st.columns(5)
        for col, m in zip(k, ["Precision", "Recall", "F1", "Correct repair rate", "Hallucination rate"]):
            d = ev[m] - ev1[m]
            col.metric(m, f"{ev[m]:.0%}", f"{d:+.0%} vs first pass" if abs(d) > 1e-9 else None,
                       delta_color="inverse" if m == "Hallucination rate" else "normal")
        n_fix = int((res.calls.stage.str.startswith("self-check")).sum()) if "stage" in res.calls else 0
        st.caption(f"First pass (blind LLM answer): precision {ev1['Precision']:.0%} · recall {ev1['Recall']:.0%} · "
                   f"F1 {ev1['F1']:.0%} · repair {ev1['Correct repair rate']:.0%} · "
                   f"self-check calls: {n_fix}"
                   + (f" · removed {res.dropped} answers about cases that were not shown" if getattr(res, "dropped", 0) else ""))
        st.caption(f"{res.model} · method {method} · TP {ev['TP']} · FP {ev['FP']} · FN {ev['FN']} · "
                   f"invented events {ev['Invented events']} · invalid values {ev['Invalid values']} · "
                   f"{res.calls.seconds.sum():.1f} s · "
                   f"{int(res.calls.input_tokens.sum() + res.calls.output_tokens.sum()):,} tokens")
        corr = res.corrections.copy()
        corr["event_id"] = corr["case"].astype(str) + "|" + corr["activity"].astype(str)
        corr = corr.merge(df[["event_id", "ts_raw"]], on="event_id", how="left")
        corr["really wrong (Q1)"] = corr.event_id.isin(gt.event_id)
        st.dataframe(corr[["case", "activity", "ts_raw", "corrected", "error_type", "reason",
                           "really wrong (Q1)"]].rename(columns={"ts_raw": "original"}),
                     width="stretch", hide_index=True)
        st.markdown("#### 🔍 Error analysis: what went wrong and why")
        diag = q.diagnose(df, gt, res.corrections)
        counts = diag.verdict.value_counts()
        chips = st.columns(4)
        for col, (label, icon) in zip(chips, [("fixed correctly", "✅"), ("found, but bad repair", "🟡"),
                                              ("MISSED error", "❌"), ("changed a CORRECT event", "⚠️")]):
            col.metric(f"{icon} {label}", int(counts.get(label, 0)))
        extra = {k: int(counts.get(k, 0)) for k in ("INVENTED event", "INVALID value", "duplicate (ignored)")}
        if any(extra.values()):
            st.caption(" · ".join(f"{k}: {v}" for k, v in extra.items()))
        problems = diag[diag.verdict != "fixed correctly"]
        colors = {"found, but bad repair": "#fff3cd", "MISSED error": "#f8d7da",
                  "changed a CORRECT event": "#ffe5b4", "INVENTED event": "#f8d7da",
                  "INVALID value": "#f8d7da", "duplicate (ignored)": "#e2e3e5"}
        if len(problems):
            st.dataframe(problems.style.apply(
                lambda r: [f"background-color: {colors.get(r.verdict, '')}; color: black"] * len(r), axis=1),
                width="stretch", hide_index=True)
        else:
            st.success("No problems: every wrong event was found and repaired, nothing else was changed.")
        st.download_button("Download error analysis (CSV)", diag.to_csv(index=False),
                           f"q5_error_analysis_{method}_{model_id.split('/')[-1]}.csv", "text/csv")
        fixed = q.apply_corrections(df, res.corrections)
        case = st.selectbox("Compare a case", sorted(set(gt.CaseID), key=lambda c: int(c.split("_")[1])),
                            key="q5_case")
        st.plotly_chart(_timeline(df, fixed, case), width="stretch")
        st.download_button("Download fixed_log.csv (input for rerunning Q2)",
                           fixed[["CaseID", "Activity", "Timestamp", "User", "Cost"]].to_csv(index=False),
                           f"fixed_log_{method}_{model_id.split('/')[-1]}.csv", "text/csv")

    if ss:
        st.divider()
        st.subheader("Q6 - comparison (rows: method, columns: LLM)")
        results = {k: q.evaluate(df, gt, r.corrections) for k, r in ss.items()}
        results1 = {k: q.evaluate(df, gt, r.first if getattr(r, "first", None) is not None else r.corrections)
                    for k, r in ss.items()}
        cols = st.columns(2)
        for i, m in enumerate(METRICS):
            t = q.table_2x2(results, m).map(lambda v: "not run yet" if pd.isna(v) else f"{v:.2f}")
            cols[i % 2].markdown(f"**{m}**")
            cols[i % 2].table(t)
        with st.expander("Same tables for the FIRST PASS only (before the self-check rounds)"):
            cols1 = st.columns(2)
            for i, m in enumerate(METRICS):
                t = q.table_2x2(results1, m).map(lambda v: "not run yet" if pd.isna(v) else f"{v:.2f}")
                cols1[i % 2].markdown(f"**{m}**")
                cols1[i % 2].table(t)
        st.subheader("Q7 - time and tokens")
        calls = pd.concat([r.calls for r in ss.values()])
        st.dataframe(q.usage_summary(calls), width="stretch", hide_index=True)
        if "stage" in calls:
            st.caption("Split by stage (first pass vs self-check rounds):")
            st.dataframe(calls.groupby(["method", "model", "stage"])[["seconds", "input_tokens", "output_tokens"]]
                         .sum().reset_index(), width="stretch", hide_index=True)
        rows = [{"stage": "final (after self-check)", "method": m, "model": lab, **v} for (m, lab), v in results.items()]
        rows += [{"stage": "first pass", "method": m, "model": lab, **v} for (m, lab), v in results1.items()]
        st.download_button("Download Q6 metrics (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           "q6_metrics.csv", "text/csv")
        st.caption(f"{len(ss)}/4 combinations run. Ground truth: the {len(gt)} wrong events of Q1.")

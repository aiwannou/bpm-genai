"""
MHYP 521 - BPM-GenAI assignment
Question 1: find, point out and explain the timestamp errors in the log.

Version 2 (fixes + extra checks on top of the first version).

IDEA OF THE SCRIPT:
  1. Load the event log (CSV or XES) into a pandas table, keeping the
     timestamps as raw text so that broken values are not hidden.
  2. Run the checks below. Every check is plain code (pandas / PM4PY),
     no LLM is involved anywhere.
       E1  Missing timestamp           (empty value)
       E2  Malformed timestamp         (value exists, wrong format)
       E3  Negative duration           (time goes backwards inside a case)
       E4  Out-of-order activities     (Approve Claim before Evaluate Claim)
       C1  Duplicate timestamps        (two events of a case at the same time)
       C2  Duration outliers           (IQR rule per transition)
       C3  Case spanning several days  (events of one case on different dates)
       C4  Out-of-range dates          (before / after the period of the log)
  3. Print where each error is (case, activity, event number) and why.
  4. Save:
       q1_summary.csv          one row per error type (for the report)
       q1_error_events.csv     one row per affected event = GROUND TRUTH
                               that Questions 5-6 will be evaluated against

Usage:
  python q1_timestamp_errors.py                          -> uses the CSV
  python q1_timestamp_errors.py insurance_claims_log.xes -> uses the XES
"""

import sys
import xml.etree.ElementTree as ET  # XES is plain XML

import pandas as pd

# PM4PY is imported only inside main() (for the final check), so other
# programs (e.g. the Streamlit app) can reuse the detection functions
# without loading PM4PY.

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DEFAULT_FILE = "insurance_claims_log.csv"
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"                     # e.g. 2026-09-01T12:53:00
TS_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"  # same format as a regex
IQR_FACTOR = 3.0  # "extreme outlier" rule: > Q3 + 3*IQR or < Q1 - 3*IQR


# ----------------------------------------------------------------------
# STEP 1: Loading
# ----------------------------------------------------------------------
def load_csv(path):
    # dtype=str: keep the Timestamp column as text, so broken values stay visible
    return pd.read_csv(path, dtype={"Timestamp": str})


def load_xes(path):
    """
    Read the XES as plain XML and build the same table as the CSV.
    pm4py.read_xes() is NOT used here on purpose: it silently turns
    '2026-09-13T16' into 16:00:00, so the malformed values would be hidden.
    """
    rows = []
    for trace in ET.parse(path).getroot().iter("trace"):
        case_id = None
        for child in trace:
            if child.tag in ("string", "id") and child.get("key") == "concept:name":
                case_id = child.get("value")
                break
        for event in trace.findall("event"):
            attrs = {a.get("key"): a.get("value") for a in event}
            rows.append({
                "CaseID": case_id,
                "Activity": attrs.get("concept:name"),
                "Timestamp": attrs.get("time:timestamp"),  # None if missing
                "User": attrs.get("org:resource"),
                "Cost": attrs.get("cost"),
            })
    return pd.DataFrame(rows)


def load_log(path):
    df = load_xes(path) if path.lower().endswith(".xes") else load_csv(path)
    df["event_no"] = df.index + 1          # position in the file (1, 2, 3, ...)
    # ts = real datetime; missing / wrong format -> NaT ("not a time")
    df["ts"] = pd.to_datetime(df["Timestamp"], format=TS_FORMAT, errors="coerce")
    return df


# ----------------------------------------------------------------------
# STEP 2a: Deriving the expected process order FROM THE DATA
# (instead of hard-coding it)
# ----------------------------------------------------------------------
def variants_in_file_order(df):
    """Count how many cases follow each sequence of activities (file order)."""
    seqs = df.groupby("CaseID", sort=False)["Activity"].apply(lambda s: " -> ".join(s))
    return seqs.value_counts()


# ----------------------------------------------------------------------
# STEP 2b: Error detection
# ----------------------------------------------------------------------
def find_missing(df):
    """E1 - the Timestamp field is empty."""
    return df[df["Timestamp"].isna()]


def find_malformed(df):
    """E2 - a value exists but does not follow YYYY-MM-DDTHH:MM:SS."""
    has_value = df["Timestamp"].notna()
    wrong_format = ~df["Timestamp"].astype(str).str.match(TS_REGEX)
    return df[has_value & wrong_format]


def find_negative_durations(df):
    """
    E3 - an event has an EARLIER timestamp than the previous event of the case.

    - Uses the ORIGINAL file order (sorting by time would hide the error).
    - FIX vs version 1: each event is compared with the LAST VALID timestamp
      before it, not only with its direct neighbour. Otherwise a reversal
      right after a missing/malformed timestamp would be skipped.
    """
    found = []
    for case_id, g in df.groupby("CaseID", sort=False):
        last_act, last_ts = None, None
        for act, ts, n in zip(g["Activity"], g["ts"], g["event_no"]):
            if pd.isna(ts):
                continue                      # no time -> nothing to compare
            if last_ts is not None and ts < last_ts:
                found.append({"CaseID": case_id,
                              "From": last_act, "From_time": last_ts,
                              "To": act, "To_time": ts,
                              "Minutes_back": (last_ts - ts).total_seconds() / 60,
                              "event_no": n})
            last_act, last_ts = act, ts
    return pd.DataFrame(found)


def find_out_of_order(df, expected_order):
    """
    E4 - the sequence of activities differs from the expected order.
    Here the timestamps ARE increasing, but they say that Approve Claim
    happened before Evaluate Claim, which is not possible in this process.
    For each case we also measure how much earlier Approve was recorded.
    """
    found = []
    for case_id, g in df.groupby("CaseID", sort=False):
        seq = g["Activity"].tolist()
        if seq != expected_order:
            t = dict(zip(g["Activity"], g["ts"]))
            gap = None
            if pd.notna(t.get("Evaluate Claim")) and pd.notna(t.get("Approve Claim")):
                gap = (t["Evaluate Claim"] - t["Approve Claim"]).total_seconds() / 60
            found.append({"CaseID": case_id,
                          "Sequence": " -> ".join(seq),
                          "Approve_before_Evaluate_min": gap,
                          "event_nos": g.loc[g["Activity"].isin(
                              ["Approve Claim", "Evaluate Claim"]), "event_no"].tolist()})
    return pd.DataFrame(found)


def find_duplicates(df):
    """C1 - two events of the same case with exactly the same timestamp."""
    valid = df.dropna(subset=["ts"])
    return valid[valid.duplicated(subset=["CaseID", "ts"], keep=False)]


def transitions(df):
    """Duration (minutes) between consecutive events of each case, file order,
    only when both timestamps are valid."""
    rows = []
    for case_id, g in df.groupby("CaseID", sort=False):
        ev = list(zip(g["Activity"], g["ts"], g["event_no"]))
        for (a1, t1, _), (a2, t2, n2) in zip(ev, ev[1:]):
            if pd.notna(t1) and pd.notna(t2):
                rows.append({"CaseID": case_id, "Transition": f"{a1} -> {a2}",
                             "Minutes": (t2 - t1).total_seconds() / 60,
                             "event_no": n2})
    return pd.DataFrame(rows)


def find_outliers(trans):
    """C2 - extreme durations per transition type (IQR rule)."""
    out = []
    for name, g in trans.groupby("Transition"):
        if len(g) < 10:            # too few values for a statistic
            continue
        q1, q3 = g["Minutes"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
        out.append(g[(g["Minutes"] < lo) | (g["Minutes"] > hi)])
    return pd.concat(out) if out else pd.DataFrame()


def find_multi_day_cases(df):
    """C3 - a case whose events fall on different calendar dates."""
    d = df.dropna(subset=["ts"]).groupby("CaseID")["ts"].apply(lambda s: s.dt.date.nunique())
    return d[d > 1]


def find_out_of_range(df):
    """C4 - dates far before / after the rest of the log (e.g. 1970, 2099).
    Rule: more than 30 days outside the 1st-99th percentile of all dates."""
    ts = df["ts"].dropna()
    lo = ts.quantile(0.01) - pd.Timedelta(days=30)
    hi = ts.quantile(0.99) + pd.Timedelta(days=30)
    return df[(df["ts"] < lo) | (df["ts"] > hi)]


def neighbours(df, case_id, event_no):
    g = df[df["CaseID"] == case_id].reset_index(drop=True)
    i = g.index[g["event_no"] == event_no][0]
    prev_ts = g.loc[i - 1, "ts"] if i > 0 else None
    next_ts = g.loc[i + 1, "ts"] if i < len(g) - 1 else None
    return prev_ts, next_ts


def sort_cases(cases):
    return sorted(set(cases), key=lambda c: int(str(c).split("_")[1]))


def title(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ----------------------------------------------------------------------
# STEP 3 + 4: run everything, print, save
# ----------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    df = load_log(path)

    title("DATASET")
    print("File:", path)
    print("Events:", len(df), "| Cases:", df["CaseID"].nunique(),
          "| Activities:", df["Activity"].nunique(),
          "| Users:", df["User"].nunique())
    print("Period:", df["ts"].min(), "->", df["ts"].max())

    # ---- expected order derived from the data ----
    title("VARIANTS (sequence of activities in file order)")
    var = variants_in_file_order(df)
    for seq, n in var.items():
        print(f"  {n:3d} cases : {seq}")
    expected_order = var.index[0].split(" -> ")
    print("\nThe most frequent variant is used as the expected (normal) process:")
    print("  " + " -> ".join(expected_order))

    missing = find_missing(df)
    malformed = find_malformed(df)
    negative = find_negative_durations(df)
    ooo = find_out_of_order(df, expected_order)
    dups = find_duplicates(df)
    trans = transitions(df)
    outl = find_outliers(trans)
    multi = find_multi_day_cases(df)
    rng = find_out_of_range(df)

    # ---- E1 ----
    title(f"E1 - MISSING TIMESTAMPS ({len(missing)} events)")
    for _, r in missing.iterrows():
        p, n = neighbours(df, r["CaseID"], r["event_no"])
        print(f"  {r['CaseID']:8s} {r['Activity']:17s} event #{r['event_no']:<4d}"
              f" previous: {p}  next: {n}")

    # ---- E2 ----
    title(f"E2 - MALFORMED TIMESTAMPS ({len(malformed)} events)")
    for _, r in malformed.iterrows():
        p, n = neighbours(df, r["CaseID"], r["event_no"])
        # If we read the value as HH:00:00 (what pm4py.read_xes does),
        # does it still fit between the neighbouring events?
        as_hour = pd.to_datetime(r["Timestamp"], format="%Y-%m-%dT%H", errors="coerce")
        fits = ((p is None or pd.isna(p) or as_hour >= p) and
                (n is None or pd.isna(n) or as_hour <= n))
        print(f"  {r['CaseID']:8s} {r['Activity']:17s} event #{r['event_no']:<4d}"
              f" value='{r['Timestamp']}'  read as {as_hour.time()} ->"
              f" {'fits' if fits else 'DOES NOT fit'} between {p} and {n}")

    # ---- E3 ----
    title(f"E3 - NEGATIVE DURATIONS ({len(negative)} transitions)")
    for _, r in negative.iterrows():
        g = df[df["CaseID"] == r["CaseID"]].set_index("Activity")["ts"]
        to_close = (g["Close Claim"] - r["To_time"]).total_seconds() / 60
        print(f"  {r['CaseID']:8s} {r['From']} ({r['From_time']}) -> "
              f"{r['To']} ({r['To_time']}) = -{r['Minutes_back']:.0f} min"
              f"   | {r['To']} -> Close = {to_close:.0f} min")
    print("\n  Every normal step in this log takes 30-120 min. Here the step into")
    print("  Evaluate is normal, but Approve -> Close is 160-250 min: the Approve")
    print("  timestamp was moved back. So the WRONG value is the Approve timestamp.")

    # ---- E4 ----
    title(f"E4 - OUT-OF-ORDER ACTIVITIES ({len(ooo)} cases)")
    for _, r in ooo.iterrows():
        print(f"  {r['CaseID']:8s} {r['Sequence']}   "
              f"(Approve recorded {r['Approve_before_Evaluate_min']:.0f} min before Evaluate)")

    # ---- extra checks ----
    title("EXTRA CHECKS (generic, not tied to a known error)")
    print(f"  C1 duplicate timestamps inside a case : {len(dups)} events")
    print(f"  C2 extreme duration outliers (IQR x{IQR_FACTOR:g}): {len(outl)} transitions")
    if len(outl):
        print(outl.to_string(index=False))
    print(f"  C3 cases spanning more than one day   : {len(multi)}")
    print(f"  C4 dates outside the log period       : {len(rng)} events")
    print("\n  Duration statistics per transition (valid, forward pairs only, minutes):")
    stats = (trans[trans["Minutes"] >= 0].groupby("Transition")["Minutes"]
             .describe()[["count", "min", "50%", "max"]].round(0))
    print(stats.to_string())

    # ---- summary ----
    neg_cases = list(negative["CaseID"]) if len(negative) else []
    ooo_cases = list(ooo["CaseID"]) if len(ooo) else []
    affected = (set(missing["CaseID"]) | set(malformed["CaseID"]) |
                set(neg_cases) | set(ooo_cases) | set(dups.get("CaseID", [])) |
                set(outl.get("CaseID", [])) | set(multi.index) | set(rng.get("CaseID", [])))

    summary = pd.DataFrame({
        "Error type": ["E1 Missing timestamp", "E2 Malformed timestamp",
                       "E3 Negative duration (Approve < Evaluate)",
                       "E4 Out-of-order (Approve before Evaluate)",
                       "C1 Duplicate timestamps", "C2 Duration outliers",
                       "C3 Multi-day cases", "C4 Out-of-range dates"],
        "Count": [len(missing), len(malformed), len(negative), len(ooo),
                  len(dups), len(outl), len(multi), len(rng)],
        "Cases": [", ".join(sort_cases(missing["CaseID"])),
                  ", ".join(sort_cases(malformed["CaseID"])),
                  ", ".join(sort_cases(neg_cases)),
                  ", ".join(sort_cases(ooo_cases)), "", "", "", ""],
        "Explanation": [
            "Timestamp is empty: the event cannot be placed in time, durations "
            "around it cannot be computed. PM4PY drops it, so the trace loses an activity.",
            "Minutes and seconds are cut (e.g. 2026-09-13T16). A strict parser rejects it; "
            "pm4py.read_xes silently reads it as HH:00:00, so the true minute is lost.",
            "Rows are in the correct order, but Approve Claim has a timestamp exactly "
            "15 min EARLIER than Evaluate Claim: time goes backwards inside the case. "
            "Approve->Close becomes 160-250 min (normal max 120), so the Approve "
            "timestamp is the wrong one.",
            "Timestamps increase, but they place Approve Claim before Evaluate Claim "
            "(61-115 min earlier). All gaps stay in the normal 30-120 min range, so the "
            "two timestamps (or labels) were swapped. A claim cannot be approved "
            "before it is evaluated.",
            "No two events of a case share the same timestamp.",
            "Only Case_5 (Approve->Close = 250 min), which is a side effect of its E3 "
            "error, not a new error.",
            "Every case happens within a single day.",
            "No dates such as 1970 or 2099.",
        ],
    })

    title("SUMMARY")
    print(summary[["Error type", "Count", "Cases"]].to_string(index=False))
    print(f"\nCases with at least one error: {len(affected)} / {df['CaseID'].nunique()}")
    print(f"Clean cases: {df['CaseID'].nunique() - len(affected)}")

    # ---- ground truth: one row per affected event ----
    gt = []
    for _, r in missing.iterrows():
        gt.append((r["event_no"], r["CaseID"], r["Activity"], r["Timestamp"], "E1 missing"))
    for _, r in malformed.iterrows():
        gt.append((r["event_no"], r["CaseID"], r["Activity"], r["Timestamp"], "E2 malformed"))
    for _, r in negative.iterrows():   # only the Approve event is wrong (see E3)
        row = df[df["event_no"] == r["event_no"]].iloc[0]
        gt.append((r["event_no"], row["CaseID"], row["Activity"], row["Timestamp"],
                   "E3 negative duration"))
    for _, r in ooo.iterrows():        # swapped pair: both events are affected
        for n in r["event_nos"]:
            row = df[df["event_no"] == n].iloc[0]
            gt.append((n, row["CaseID"], row["Activity"], row["Timestamp"], "E4 out of order"))
    gt = pd.DataFrame(gt, columns=["event_no", "CaseID", "Activity", "Timestamp", "Error"])
    gt = gt.sort_values("event_no")

    summary.to_csv("q1_summary.csv", index=False)
    gt.to_csv("q1_error_events.csv", index=False)
    print(f"\nSaved: q1_summary.csv  and  q1_error_events.csv ({len(gt)} events)")

    # ---- what PM4PY sees ----
    try:
        import pm4py
        HAS_PM4PY = True
    except ImportError:
        HAS_PM4PY = False
    if HAS_PM4PY:
        clean = df.dropna(subset=["ts"]).rename(columns={
            "CaseID": "case:concept:name", "Activity": "concept:name",
            "ts": "time:timestamp"})
        clean = pm4py.format_dataframe(clean, case_id="case:concept:name",
                                       activity_key="concept:name",
                                       timestamp_key="time:timestamp")
        title("PM4PY CHECK - variants after sorting by timestamp")
        print(f"Events with a valid timestamp: {len(clean)} (left out: {len(df) - len(clean)})")
        for v, n in sorted(pm4py.get_variants(clean).items(), key=lambda x: -x[1]):
            print(f"  {n:3d} cases : {' -> '.join(v)}")
        print("\nNote: after sorting by time, E3 and E4 cases look the same "
              "(Approve before Evaluate), and E1/E2 cases lose one activity.")
    else:
        print("\n(pm4py not installed - PM4PY check skipped)")


if __name__ == "__main__":
    main()

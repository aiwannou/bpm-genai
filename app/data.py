"""
Data layer of the assistant (Questions 3-5).

Builds two pandas tables from the event log that the LLM-generated code
works on:

  events : one row per event (500 rows)
  cases  : one row per case / claim (100 rows)

The timestamp errors found in Question 1 are flagged with the SAME detection
functions (imported from q1_timestamp_errors.py), so time-based answers can
exclude unreliable values instead of silently using them.
"""

import pandas as pd

from q1_timestamp_errors import (
    load_log, find_missing, find_malformed,
    find_negative_durations, find_out_of_order, variants_in_file_order,
)

LOG_FILE = "insurance_claims_log.csv"


def build_tables(path=LOG_FILE):
    df = load_log(path)                       # raw text Timestamp + parsed ts
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")

    # ---- event-level error flags (Question 1) ----
    df["data_issue"] = ""
    df.loc[find_missing(df).index, "data_issue"] = "E1 missing timestamp"
    df.loc[find_malformed(df).index, "data_issue"] = "E2 malformed timestamp"
    neg = find_negative_durations(df)
    if len(neg):
        df.loc[df["event_no"].isin(neg["event_no"]), "data_issue"] = "E3 negative duration"
    expected = variants_in_file_order(df).index[0].split(" -> ")
    ooo = find_out_of_order(df, expected)
    for nos in (ooo["event_nos"] if len(ooo) else []):
        df.loc[df["event_no"].isin(nos), "data_issue"] = "E4 out of order"

    # ---- durations (minutes since the previous event of the same case) ----
    # The log has one timestamp per event (completion time), so the time
    # spent on / waiting for an activity = its timestamp - previous timestamp.
    df["step_no"] = df.groupby("CaseID").cumcount() + 1
    df["prev_ts"] = df.groupby("CaseID")["ts"].shift(1)
    df["prev_issue"] = df.groupby("CaseID")["data_issue"].shift(1).fillna("")
    df["duration_min"] = (df["ts"] - df["prev_ts"]).dt.total_seconds() / 60
    # A duration is reliable only if both timestamps are valid and correct
    df["duration_reliable"] = (df["duration_min"].notna() & (df["duration_min"] >= 0)
                               & (df["data_issue"] == "") & (df["prev_issue"] == ""))
    # the timestamp of the event itself is valid and has no Question-1 error
    df["ts_reliable"] = df["ts"].notna() & (df["data_issue"] == "")

    events = df[["event_no", "CaseID", "step_no", "Activity", "User", "Cost",
                 "Timestamp", "ts", "ts_reliable", "duration_min", "duration_reliable",
                 "data_issue"]].rename(columns={"Timestamp": "timestamp_raw"})

    # ---- case-level table ----
    g = events.groupby("CaseID", sort=False)
    cases = pd.DataFrame({
        "total_cost": g["Cost"].sum(),
        "n_events": g.size(),
        "start": g["ts"].min(),
        "end": g["ts"].max(),
        "users": g["User"].apply(lambda s: ", ".join(sorted(s.unique()))),
        "n_users": g["User"].nunique(),
        "sequence": g["Activity"].apply(lambda s: " -> ".join(s)),
        "has_data_issue": g["data_issue"].apply(lambda s: (s != "").any()),
        "issues": g["data_issue"].apply(lambda s: "; ".join(sorted(set(s) - {""}))),
    }).reset_index()
    cases["cycle_time_min"] = (cases["end"] - cases["start"]).dt.total_seconds() / 60
    # cycle time is only trustworthy when the case has no timestamp error
    cases.loc[cases["has_data_issue"], "cycle_time_min"] = float("nan")
    cases["case_num"] = cases["CaseID"].str.split("_").str[1].astype(int)
    cases = cases.sort_values("case_num").drop(columns="case_num").reset_index(drop=True)
    return events, cases


# Text given to the LLM so it knows exactly what the tables contain
SCHEMA_DESCRIPTION = """
Two pandas DataFrames are available.

`events` - one row per event of an insurance-claims process (500 rows, 100 cases)
  event_no       int       position of the event in the original file (1..500)
  CaseID         str       claim id, e.g. "Case_1" ... "Case_100"
  step_no        int       position of the event inside its case (1..5)
  Activity       str       one of: "Register Claim", "Verify Documents", "Evaluate Claim",
                           "Approve Claim", "Close Claim"
  User           str       resource who did the activity: "User_A", "User_B", "User_C", "User_D", "User_E"
  Cost           int       operational cost of the event in dollars ($50-$500)
  timestamp_raw  str       timestamp exactly as in the file (may be empty or malformed)
  ts             datetime  parsed completion time (NaT if missing/malformed)
  ts_reliable    bool      True if ts is valid and has no timestamp error (use it when you
                           need the DATE/TIME of an event, e.g. counting per month)
  duration_min   float     minutes between the previous event of the case and this one
                           = waiting + processing time of this activity (always NaN for
                           step 1 "Register Claim", because there is no previous event)
  duration_reliable bool   True if duration_min can be trusted (both timestamps correct).
                           Always False for step 1. Use it ONLY for duration statistics.
  data_issue     str       "" or the timestamp error found in Question 1
                           ("E1 missing timestamp", "E2 malformed timestamp",
                            "E3 negative duration", "E4 out of order")

`cases` - one row per claim (100 rows)
  CaseID          str
  total_cost      int       sum of Cost of all events of the claim (= cost of the claim)
  n_events        int
  start, end      datetime  first / last valid timestamp
  cycle_time_min  float     end - start in minutes (NaN if the case has a timestamp error)
  users           str       comma separated users that worked on the claim
  n_users         int
  sequence        str       activities in file order joined with " -> "
  has_data_issue  bool
  issues          str       list of timestamp errors of the case

Notes:
- The process is Register Claim -> Verify Documents -> Evaluate Claim -> Approve Claim -> Close Claim.
- "Claims handled by a user" = cases where that user appears in at least one event.
- "Claim amount / claim cost" = cases.total_cost.
- For any duration/waiting-time statistic use only rows with duration_reliable == True
  and say so. Do NOT use duration_reliable to filter dates or counts.
- Because unreliable rows are excluded, activities have DIFFERENT numbers of reliable rows.
  So when you compare durations/waiting times between activities, rank them by the MEAN
  (average) duration, not by the sum. If a percentage "over the total" is asked, report each
  activity's share of the summed mean durations, and also give the mean and the count.
- To count or group claims by date/month, use cases.start (first valid timestamp of the
  claim), so that claims whose first event has a broken timestamp are not lost.
- The period of the log is 2026-09-01 to 2026-12-09.
"""


if __name__ == "__main__":
    ev, cs = build_tables()
    print(ev.head(12).to_string())
    print(cs.head().to_string())
    print(ev["data_issue"].value_counts())
    print("reliable durations:", ev["duration_reliable"].sum(),
          "| reliable timestamps:", ev["ts_reliable"].sum())

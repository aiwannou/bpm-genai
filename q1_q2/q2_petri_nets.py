"""
MHYP 521 - BPM-GenAI assignment
Question 2: discover the Petri net of the process with the Alpha Miner,
Heuristic Miner and Inductive Miner (PM4PY) and compare the models,
taking the timestamp errors of Question 1 into account.

IDEA OF THE SCRIPT:
  1. Build two event logs:
       DIRTY  = the log exactly as PM4PY reads it (pm4py.read_xes), with
                no cleaning. Events without a timestamp have to be dropped
                (PM4PY cannot order them); malformed values like
                '2026-09-13T16' are read by PM4PY as 16:00:00.
       CLEAN  = only the 78 cases that have no error in Question 1
                (reference model: what the process really looks like).
     (Optional) FIXED = a corrected log from Question 5, given as 2nd argument.
  2. On each log run: Alpha Miner, Heuristic Miner, Inductive Miner
     (+ Inductive Miner infrequent, noise 0.2, as an extra comparison).
  3. For every model compute the standard quality dimensions:
       fitness (token-based replay), precision (ETC), generalization,
       simplicity, and whether the Petri net is a sound workflow net.
  4. Save one PNG per model and a CSV with all the metrics.

Needs: pip install pm4py   +   Graphviz installed (for the PNG images)
Run AFTER q1_timestamp_errors.py (it uses q1_error_events.csv).

Usage:
  python q2_petri_nets.py
  python q2_petri_nets.py insurance_claims_log.xes fixed_log.csv   (after Q5)
"""

import os
import sys
import warnings

import pandas as pd
import pm4py

warnings.filterwarnings("ignore")

XES_FILE = sys.argv[1] if len(sys.argv) > 1 else "insurance_claims_log.xes"
FIXED_FILE = sys.argv[2] if len(sys.argv) > 2 else None   # from Question 5
GROUND_TRUTH = "q1_error_events.csv"                       # from Question 1
OUT_DIR = "q2_output"
os.makedirs(OUT_DIR, exist_ok=True)


def title(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ----------------------------------------------------------------------
# STEP 1: the event logs
# ----------------------------------------------------------------------
def build_logs():
    raw = pm4py.read_xes(XES_FILE)          # standard PM4PY loading
    print(f"pm4py.read_xes: {len(raw)} events, "
          f"{raw['time:timestamp'].isna().sum()} without a timestamp")

    dirty = raw.dropna(subset=["time:timestamp"]).copy()

    bad_cases = set(pd.read_csv(GROUND_TRUTH)["CaseID"])
    clean = raw[~raw["case:concept:name"].isin(bad_cases)].copy()

    logs = {"dirty": dirty, "clean78": clean}

    if FIXED_FILE:                           # corrected log from Question 5
        fx = pd.read_csv(FIXED_FILE)
        fx = pm4py.format_dataframe(fx, case_id="CaseID", activity_key="Activity",
                                    timestamp_key="Timestamp")
        logs["fixed"] = fx

    for name, lg in logs.items():
        lg.sort_values(["case:concept:name", "time:timestamp"], inplace=True)
        title(f"LOG '{name}': {lg['case:concept:name'].nunique()} cases, "
              f"{len(lg)} events - variants (sorted by time)")
        for v, n in sorted(pm4py.get_variants(lg).items(), key=lambda x: -x[1]):
            print(f"  {n:3d} : {' -> '.join(v)}")
    return logs


# ----------------------------------------------------------------------
# STEP 2 + 3: discovery and evaluation
# ----------------------------------------------------------------------
MINERS = {
    "alpha": lambda log: pm4py.discover_petri_net_alpha(log),
    "heuristic": lambda log: pm4py.discover_petri_net_heuristics(
        log, dependency_threshold=0.5),          # PM4PY default threshold
    "inductive": lambda log: pm4py.discover_petri_net_inductive(
        log, noise_threshold=0.0),               # plain Inductive Miner (IM)
    "inductive_IMf0.2": lambda log: pm4py.discover_petri_net_inductive(
        log, noise_threshold=0.2),               # Inductive Miner infrequent
}


def is_sound(net, im, fm):
    try:
        return bool(pm4py.check_soundness(net, im, fm)[0])
    except Exception:
        return False


def evaluate(log, net, im, fm):
    fit = pm4py.fitness_token_based_replay(log, net, im, fm)
    return {
        "fitness": round(fit["log_fitness"], 3),
        "fitting_traces_%": round(fit["perc_fit_traces"], 1),
        "precision": round(pm4py.precision_token_based_replay(log, net, im, fm), 3),
        "generalization": round(pm4py.generalization_tbr(log, net, im, fm), 3),
        "simplicity": round(pm4py.simplicity_petri_net(net, im, fm), 3),
        "sound": is_sound(net, im, fm),
        "places": len(net.places),
        "transitions": len(net.transitions),
        "silent_transitions": sum(1 for t in net.transitions if t.label is None),
        "arcs": len(net.arcs),
    }


def main():
    logs = build_logs()
    rows = []
    for log_name, log in logs.items():
        for miner_name, miner in MINERS.items():
            net, im, fm = miner(log)
            m = evaluate(log, net, im, fm)
            rows.append({"log": log_name, "miner": miner_name, **m})
            png = os.path.join(OUT_DIR, f"{log_name}_{miner_name}.png")
            pm4py.save_vis_petri_net(net, im, fm, png)

        # the Heuristic Miner's own model (dependency graph with frequencies)
        hn = pm4py.discover_heuristics_net(log, dependency_threshold=0.5)
        pm4py.save_vis_heuristics_net(hn, os.path.join(OUT_DIR, f"{log_name}_heuristics_net.png"))
        # directly-follows graph: the raw input that every miner starts from
        dfg, sa, ea = pm4py.discover_dfg(log)
        pm4py.save_vis_dfg(dfg, sa, ea, os.path.join(OUT_DIR, f"{log_name}_dfg.png"))

    res = pd.DataFrame(rows)
    title("RESULTS (one row per log x miner)")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT_DIR, "q2_metrics.csv"), index=False)
    print(f"\nSaved: {OUT_DIR}/q2_metrics.csv and the PNG images in {OUT_DIR}/")


if __name__ == "__main__":
    main()

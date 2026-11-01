# Question 2 - Notes for the report (based on the actual results)

## Setup
- Tool: PM4PY (Python). Algorithms: Alpha Miner, Heuristic Miner (dependency threshold 0.5),
  Inductive Miner (IM, noise 0) and, as an extra, Inductive Miner infrequent (IMf, noise 0.2).
- Two logs:
  - **Dirty log**: the XES as loaded by `pm4py.read_xes`, no cleaning. The 5 events with a missing
    timestamp must be dropped (cannot be ordered). The 5 malformed values (e.g. `2026-09-13T16`)
    are silently read as `16:00:00` and stay in the log, in the correct position.
    Result: 100 cases, 495 events, 5 variants.
  - **Clean reference log**: the 78 cases with no error from Question 1 (1 variant).
- Quality dimensions: fitness (token-based replay), precision (ETC), generalization, simplicity,
  soundness (workflow-net check).

## What the errors do to the input of the miners (variants of the dirty log, sorted by time)
| Cases | Variant | Caused by |
|---|---|---|
| 83 | Register → Verify → Evaluate → Approve → Close | normal (78 clean + 5 malformed, read as HH:00) |
| 12 | Register → Verify → **Approve → Evaluate** → Close | E3 (5) + E4 (7): Approve before Evaluate |
| 3 | Register → Verify → Approve → Close (no Evaluate) | E1: Evaluate missing (Case_34, 70, 84) |
| 1 | Register → Evaluate → Approve → Close (no Verify) | E1: Verify missing (Case_28) |
| 1 | Verify → Evaluate → Approve → Close (no Register) | E1: Register missing (Case_77) |

Key point: after sorting by time, E3 and E4 become the same thing (Approve before Evaluate in 12% of
cases), and every missing timestamp becomes a "skipped" activity.

## Results
| Log | Miner | Fitness | Fitting traces | Precision | Generalization | Simplicity | Sound | Places / Trans. (silent) |
|---|---|---|---|---|---|---|---|---|
| dirty | Alpha | 0.937 | **0%** | 1.000 | 0.899 | 0.765 | **No** | 8 / 5 (0) |
| dirty | Heuristic | 0.977 | 84% | 1.000 | 0.750 | 0.778 | Yes | 8 / 6 (1) |
| dirty | Inductive (IM) | **1.000** | 100% | **0.729** | 0.647 | 0.739 | Yes | 8 / 9 (4) |
| dirty | Inductive (IMf 0.2) | 0.992 | 95% | 1.000 | 0.899 | 1.000 | Yes | 7 / 5 (0) |
| clean78 | all four | 1.000 | 100% | 1.000 | 0.887 | 1.000 | Yes | 6 / 5 (0) |

## Comments per model

**Clean log (reference).** All algorithms give the same model: a simple sequence
Register → Verify → Evaluate → Approve → Close, with perfect fitness and precision.
So any difference in the dirty models is caused only by the timestamp errors.

**Alpha Miner (dirty).** The most affected model.
- Because Approve→Evaluate (12 cases) and Evaluate→Approve (83 cases) are both observed, Alpha
  declares the two activities **parallel** (A || B), so the model no longer says that a claim must
  be evaluated before it is approved.
- Because one trace starts with Verify (Register missing in Case_77), Verify gets an input arc
  from the start place AND from Register; one trace goes Register → Evaluate (Verify missing in
  Case_28), so an extra place Register → Evaluate is created.
- Result: the net is **not sound** (Verify needs two tokens that can never both be present), so
  **0% of the traces can be replayed perfectly**, even though the average token fitness looks high
  (0.937). Alpha has no notion of frequency, so a single wrong event changes the structure.

**Heuristic Miner (dirty).** Uses frequencies, so it is more robust.
- The 12 Approve-before-Evaluate cases are below the dependency threshold, so the main order
  Evaluate → Approve is kept.
- The missing Register event of Case_77 creates an invisible (silent) transition that lets the
  process skip Register; extra arcs Verify → Approve and Evaluate → Close come from the traces
  with a missing Evaluate.
- Sound, but 16% of the traces do not fit (the reordered/missing ones).

**Inductive Miner (IM, dirty).** Guarantees a sound model and 100% fitness, so it must explain
every trace, including the wrong ones.
- 4 silent transitions: Register, Verify and Evaluate become **optional** (skippable), and
  Evaluate || Approve become **parallel**.
- Fitness 1.0 but precision only 0.729: the model **over-generalizes** and allows behaviour that
  never happens (e.g. closing a claim without registering or evaluating it). This is the classic
  effect of noise on IM.

**Inductive Miner infrequent (IMf, noise 0.2, extra).** Filtering infrequent behaviour removes the
skips caused by the 5 missing timestamps, but the 12% Approve-before-Evaluate behaviour is still
frequent enough to survive, so **Evaluate || Approve remains parallel**. Timestamp errors in 12% of
cases are no longer "noise", they look like real process behaviour.

## Conclusions
1. The errors do not create loops; they create **false parallelism** (Evaluate || Approve) and
   **false optional activities** (skips).
2. Ranking of robustness on this log: IMf > Heuristic > IM > Alpha.
   Alpha even produces an unsound model; IM is sound but imprecise.
3. The malformed timestamps (E2) had no visible effect here only because PM4PY reads them as
   HH:00 and, in this log, the hour still falls between the neighbouring events (checked in Q1).
   With a strict parser they would be dropped and would add 5 more "skip" variants.
4. Simply filtering is not enough (IMf still keeps the parallelism). The timestamps must be
   corrected (Question 5) to recover the real sequential process. After Q5, rerun:
   `python q2_petri_nets.py insurance_claims_log.xes fixed_log.csv` and compare the "fixed" rows
   with the "clean78" rows.

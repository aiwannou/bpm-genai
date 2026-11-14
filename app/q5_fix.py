"""
Question 5 - Timestamp correction with two LLMs x two methods.

Method A (statistical): median duration of each step + typical (P5-P95) and observed (min-max) range,
                        learned from the internally consistent cases of the log.
Method B (machine learning): Random Forest regressor that predicts the duration of each step from
                        activity, users, cost and time of day (out-of-fold, so an event never "sees" itself).

1. First pass: the LLM receives the raw cases + the method's hints, decides which events are wrong
   and returns corrected values as JSON. It is NOT told which events are wrong.
2. Self-check (up to 2 rounds): every case is checked against the process rules only (valid value,
   order, same day, step lengths in the observed range, known hour kept, consistent events untouched).
   Cases with problems go back to the same LLM with the problems in words (Self-Refine idea,
   Madaan et al., 2023). An answer is accepted only if it is not worse than the previous one.

The Q1 ground truth is used ONLY for scoring (Q6), never to build hints or feedback.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ORDER = ["Register Claim", "Verify Documents", "Evaluate Claim", "Approve Claim", "Close Claim"]
PREV = {a: (ORDER[i - 1] if i else None) for i, a in enumerate(ORDER)}
NEXT = {a: (ORDER[i + 1] if i < len(ORDER) - 1 else None) for i, a in enumerate(ORDER)}
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
HOUR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})$")
FMT = "%Y-%m-%dT%H:%M:%S"

VERSION = "Q5 final · prompt v4 + self-check · v6 (GPT-OSS + Qwen)"   # shown in the GUI so you can see which version is running

MODELS = {
    "GPT-OSS 120B (OpenAI)": "openai/gpt-oss-120b",
    "Qwen 3.8 27B (Alibaba)": "qwen/qwen3.8-27b",   # Llama 3.3 70B was retired by Groq on 16 Aug 2026
}
METHODS = {"A": "Statistical (median durations)", "B": "Machine learning (Random Forest)"}


# ----------------------------------------------------------------------------- data
def load_log(path_or_df) -> pd.DataFrame:
    """Read the CSV keeping timestamps as raw text (same idea as Q1)."""
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df, dtype=str)
    df = df.rename(columns={"Timestamp": "ts_raw"}) if "Timestamp" in df.columns else df
    df["ts_raw"] = df["ts_raw"].fillna("").astype(str).str.strip()
    df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce")
    df["file_pos"] = df.groupby("CaseID", sort=False).cumcount()
    ok = df["ts_raw"].str.match(TS_RE)
    df["ts"] = pd.to_datetime(df["ts_raw"].where(ok), format=FMT, errors="coerce")
    df["event_id"] = df["CaseID"] + "|" + df["Activity"]
    return df.reset_index(drop=True)


def ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Q1 rules -> the 29 wrong events. Used ONLY for evaluation (Q6)."""
    rows = []
    for case, g in df.groupby("CaseID", sort=False):
        for _, r in g.iterrows():
            if r.ts_raw == "":
                rows.append((case, r.Activity, "E1 missing"))
            elif pd.isna(r.ts):
                rows.append((case, r.Activity, "E2 malformed"))
        last = None
        for _, r in g.iterrows():
            if pd.notna(r.ts):
                if last is not None and r.ts < last:
                    rows.append((case, r.Activity, "E3 negative duration"))
                last = r.ts
        acts = g.Activity.tolist()
        if acts != ORDER:
            for a in acts:
                if acts.index(a) != ORDER.index(a):
                    rows.append((case, a, "E4 out of order"))
    gt = pd.DataFrame(rows, columns=["CaseID", "Activity", "error_type"]).drop_duplicates(["CaseID", "Activity"])
    gt["event_id"] = gt.CaseID + "|" + gt.Activity
    return gt


def _gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Gap (minutes) from the EXPECTED predecessor / to the expected successor."""
    t = df.set_index("event_id")["ts"]
    out = df.copy()
    out["prev_act"] = out.Activity.map(PREV)
    out["next_act"] = out.Activity.map(NEXT)
    prev_ts = (out.CaseID + "|" + out.prev_act.fillna("")).map(t)
    next_ts = (out.CaseID + "|" + out.next_act.fillna("")).map(t)
    out["prev_ts"] = prev_ts
    out["gap_prev_min"] = (out.ts - prev_ts).dt.total_seconds() / 60
    out["gap_next_min"] = (next_ts - out.ts).dt.total_seconds() / 60
    return out


# ------------------------------------------------------------------ method hints
def consistent_cases(df: pd.DataFrame) -> set:
    """Cases that look internally consistent: every timestamp valid, rows in process order and
    time always moving forward. Hints are LEARNED only from these, so broken cases do not
    distort the 'normal' durations. (Generic data cleaning - the Q1 labels are not used.)"""
    ok = set()
    for case, g in df.groupby("CaseID", sort=False):
        t = g.ts
        if g.Activity.tolist() == ORDER and t.notna().all() and (t.diff().dropna() > pd.Timedelta(0)).all():
            ok.add(case)
    return ok


def stats_hints(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Method A: robust duration statistics per activity (median + P5-P95)."""
    g = _gaps(df)
    train = g[(g.gap_prev_min > 0) & g.CaseID.isin(consistent_cases(df))]
    st = train.groupby("Activity").gap_prev_min.agg(
        median="median", p5=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95),
        min="min", max="max", n="count")
    st = st.reindex(ORDER[1:]).round(0)
    g = g.join(st[["median", "p5", "p95", "min", "max"]], on="Activity")
    g["expected_gap_min"] = g["median"]
    g["expected_ts"] = g.prev_ts + pd.to_timedelta(g["median"], unit="m")
    return g, st


def ml_hints(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Method B: Random Forest predicts the step duration; out-of-fold for training rows."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    g = _gaps(df)
    users = df.set_index("event_id")["User"]
    g["prev_user"] = (g.CaseID + "|" + g.prev_act.fillna("")).map(users).fillna("none")
    g["prev_hour"] = g.prev_ts.dt.hour + g.prev_ts.dt.minute / 60
    g["weekday"] = g.prev_ts.dt.weekday
    X = pd.get_dummies(g[["Activity", "User", "prev_user"]]).astype(float)
    X["Cost"] = g.Cost
    X["prev_hour"] = g.prev_hour.fillna(12)
    X["weekday"] = g.weekday.fillna(0)
    train = ((g.gap_prev_min > 0) & g.CaseID.isin(consistent_cases(df))).values
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed)
    pred = np.full(len(g), np.nan)
    kf = KFold(5, shuffle=True, random_state=seed)
    pred[train] = cross_val_predict(rf, X[train], g.gap_prev_min[train], cv=kf)
    rf.fit(X[train], g.gap_prev_min[train])
    pred[~train] = rf.predict(X[~train])
    resid = g.gap_prev_min[train] - pred[train]
    info = {
        "train_rows": int(train.sum()),
        "MAE_min": float(np.mean(np.abs(resid))),
        "baseline_MAE_min": float(np.mean(np.abs(g.gap_prev_min[train] - g.gap_prev_min[train].median()))),
        "top_features": pd.Series(rf.feature_importances_, X.columns).nlargest(5).round(3).to_dict(),
        "residual_p95": float(np.quantile(np.abs(resid), 0.95)),
    }
    g["expected_gap_min"] = np.round(pred, 0)
    g.loc[g.prev_act.isna(), "expected_gap_min"] = np.nan
    g["expected_ts"] = g.prev_ts + pd.to_timedelta(g["expected_gap_min"], unit="m")
    lo = np.round(pred - info["residual_p95"], 0)
    g["p5"] = np.maximum(lo, 1)
    g["p95"] = np.round(pred + info["residual_p95"], 0)
    return g, info


_HINT_CACHE: dict = {}


def hints(df, method):
    """Cached: the Random Forest is trained once per log, not on every Streamlit rerun."""
    key = (method, hash("".join(df.event_id) + "".join(df.ts_raw)))
    if key not in _HINT_CACHE:
        _HINT_CACHE[key] = stats_hints(df)[0] if method == "A" else ml_hints(df)[0]
    return _HINT_CACHE[key]


# --------------------------------------------------------------------- prompting
SYSTEM = """You repair timestamps in an insurance-claims event log.
Every claim must follow: Register Claim -> Verify Documents -> Evaluate Claim -> Approve Claim -> Close Claim,
all on the same day, with strictly increasing timestamps.
A timestamp can be wrong because it is missing, malformed (not YYYY-MM-DDTHH:MM:SS),
earlier than its expected predecessor, or because the row order breaks the process order.
Rules:
- Only list events whose OWN timestamp is wrong. A valid timestamp that fits the process order is
  ground truth: never move it just to make a gap look closer to the median.
- Fix the wrong event to fit BETWEEN its correct neighbours (for a swapped pair, give each event
  the other's time). Do not shift later events to make room.
- When two events conflict, look at BOTH gaps (gap_from_... and gap_to_...): the wrong event is the
  one whose gaps are abnormal on both sides; its neighbours with normal gaps are correct.
- An event is wrong ONLY if (1) its value is missing, (2) its value is malformed, (3) it is earlier
  than its expected predecessor (negative gap), or (4) the row order breaks the process order.
  A positive gap of ANY length is never an error by itself, even if it is outside the typical range.
- Choose each repaired time so that the gaps on BOTH sides fall inside the observed range; if the
  neighbours are far apart, put it near the middle between them.
- Never invent cases or activities; use exactly the case ids and activity names given.
- Every corrected value must be YYYY-MM-DDTHH:MM:SS, on the claim's date, and keep the order above.
- If a malformed value still shows the hour (known_hour), the fix MUST stay inside that hour.
- Base each correction on the hints provided ({method_desc}); say briefly why.
Return JSON only: {{"corrections":[{{"case":"Case_1","activity":"Approve Claim","error_type":"missing|malformed|negative_duration|out_of_order","corrected":"2026-09-01T15:46:00","reason":"..."}}]}}
If a batch has no errors return {{"corrections":[]}}."""

METHOD_DESC = {
    "A": "statistical hints: the median duration of each step, its typical range (P5-P95) and observed range (min-max)",
    "B": "machine-learning hints: a Random Forest prediction of each step's duration and the predicted timestamp",
}


def _fmt(t):
    return "" if pd.isna(t) else pd.Timestamp(t).strftime(FMT)


def build_prompt(h: pd.DataFrame, cases: list[str], method: str) -> str:
    lines = []
    if method == "A":
        st = h.dropna(subset=["median"]).groupby("Activity")[["median", "p5", "p95", "min", "max"]] \
            .first().reindex(ORDER[1:])
        lines.append("Step duration statistics (minutes from the previous step, consistent cases only):")
        for a, r in st.iterrows():
            lines.append(f"  {a}: median {r['median']:.0f}, typical {r.p5:.0f}-{r.p95:.0f}, "
                         f"observed range {r['min']:.0f}-{r['max']:.0f}")
    else:
        lines.append("For each event, 'ml_gap' is the Random Forest's predicted minutes since the expected "
                     "previous step and 'ml_ts' is the resulting predicted timestamp.")
    lines.append("\nCases (rows in file order):")
    for c in cases:
        lines.append(f"\n{c}:")
        for _, r in h[h.CaseID == c].sort_values("file_pos").iterrows():
            s = f"  - {r.Activity} | ts='{r.ts_raw}' | user={r.User} | cost={r.Cost:.0f}"
            m = HOUR_RE.match(r.ts_raw)
            if m:
                s += f" | known_hour={m.group(2)} (fix must be {m.group(2)}:00-{m.group(2)}:59)"
            if pd.notna(r.gap_prev_min):
                s += f" | gap_from_{PREV[r.Activity].split()[0]}={r.gap_prev_min:.0f}"
            if pd.notna(r.gap_next_min):
                s += f" | gap_to_{NEXT[r.Activity].split()[0]}={r.gap_next_min:.0f}"
            if method == "B" and pd.notna(r.expected_gap_min):
                s += f" | ml_gap={r.expected_gap_min:.0f}"
                if pd.notna(r.expected_ts):
                    s += f" | ml_ts={_fmt(r.expected_ts)}"
            lines.append(s)
    return "\n".join(lines)


# -------------------------------------------------------------------- LLM calls
KEY_ALIASES = {
    "case": ("case", "caseid", "case_id", "case id"),
    "activity": ("activity", "activity_name", "event", "step"),
    "corrected": ("corrected", "corrected_timestamp", "new_timestamp", "timestamp", "fixed", "new"),
    "error_type": ("error_type", "error", "type", "issue"),
    "reason": ("reason", "justification", "explanation", "why"),
}


def _norm_ts(v):
    """Accept small format slips of the LLM ('2026-09-01 15:46', '...15:46:00Z') -> YYYY-MM-DDTHH:MM:SS.
    A value that is not a real date/time is returned unchanged (and counted as invalid)."""
    v = str(v).strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?Z?$", v)
    if not m:
        return v
    out = f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4) or '00'}"
    return out if pd.notna(pd.to_datetime(out, format=FMT, errors="coerce")) else v


def normalize_item(it):
    """Map the LLM's field names to ours ('CaseID' -> 'case', ...) and tidy the values."""
    if not isinstance(it, dict):
        return {"case": "", "activity": "", "corrected": str(it), "error_type": "", "reason": "not an object"}
    low = {str(k).strip().lower(): v for k, v in it.items()}
    out = {}
    for key, names in KEY_ALIASES.items():
        out[key] = next((str(low[n]).strip() for n in names if n in low and low[n] is not None), "")
    c = re.match(r"^(?:case[_ ]?)?0*(\d+)$", out["case"].strip(), re.I)
    if c:
        out["case"] = f"Case_{int(c.group(1))}"
    words = re.findall(r"[a-z]+", out["activity"].lower())
    match = [a for a in ORDER if a.split()[0].lower() in words or a.split()[0].lower()[:5] in
             [w[:5] for w in words]]
    if len(match) == 1:  # e.g. "Verify Document", "approve", "Evaluation" -> real name
        out["activity"] = match[0]
    out["corrected"] = _norm_ts(out["corrected"])
    return out


def is_rate_limit(err: str) -> bool:
    e = err.lower()
    return "429" in e or "rate_limit" in e or "rate limit" in e


def is_daily_limit(err: str) -> bool:
    e = err.lower()
    return "tokens per day" in e or "tpd" in e or "requests per day" in e or "rpd" in e


def _retry_after(err, default):
    """Groq says 'Please try again in 105ms' / '7.5s' / '1m2.5s' / '1h3m': wait exactly that long.
    A per-MINUTE limit never needs more than about a minute."""
    m = re.search(r"try again in\s+([\d.]+)\s*ms", err)
    if m:
        wait = float(m.group(1)) / 1000 + 1
    else:
        m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", err)
        if not m or not any(m.groups()):
            return default
        h, mi, se = (float(x) if x else 0.0 for x in m.groups())
        wait = h * 3600 + mi * 60 + se + 1
    if "per minute" in err.lower() or "tpm" in err.lower() or "rpm" in err.lower():
        wait = min(wait, 65)
    return wait


def parse_json(text):
    """Parse the LLM answer; tolerates ```json fences and text around the object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)  # JSON with text around it
        return json.loads(m.group(0) if m else text)


def _call(llm, system, prompt, json_mode=True):
    """Works with llm.generate returning a dict, a tuple or an object."""
    t0 = time.time()
    res = llm.generate(system, prompt, json_mode=json_mode)
    secs = time.time() - t0
    if isinstance(res, dict):
        text = res.get("text") or res.get("content") or ""
        return text, res.get("seconds", secs), res.get("input_tokens", 0), res.get("output_tokens", 0)
    if isinstance(res, (tuple, list)):
        text, *rest = res
        rest = list(rest) + [secs, 0, 0]
        return text, rest[0], rest[1], rest[2]
    return (getattr(res, "text", str(res)), getattr(res, "seconds", secs),
            getattr(res, "input_tokens", 0), getattr(res, "output_tokens", 0))


@dataclass
class RunResult:
    model: str
    method: str
    corrections: pd.DataFrame            # final answer (after the self-check rounds)
    calls: pd.DataFrame
    first: pd.DataFrame = None           # answer of the first pass only (blind LLM result)
    stopped: str = ""                    # set when the run had to stop (daily limit)
    dropped: int = 0                     # answers about non-existent / not-shown cases removed


class DailyLimit(Exception):
    """Groq's daily token/request budget for this model is used up: stop the run."""


class BadKey(Exception):
    """The API key is not accepted: stop the run."""


def _ask(llm, system, prompt, retries=6, rate_wait=20, max_wait=90):
    """One LLM request with retries. Returns (items, seconds, in_tokens, out_tokens, error)."""
    items, secs, tin, tout, err = [], 0.0, 0, 0, ""
    json_fails = other_fails = 0
    strict = True               # Groq's JSON mode; switched off if Groq rejects the model's JSON
    for attempt in range(retries + 1):
        try:
            p = prompt if strict else prompt + "\n\nAnswer with the JSON object only, no other text."
            text, s_, i_, o_ = _call(llm, system, p, json_mode=strict)
            secs, tin, tout = secs + (s_ or 0), tin + (i_ or 0), tout + (o_ or 0)
            if not (text or "").strip():
                raise ValueError("empty answer from the model (no JSON)")
            obj = parse_json(text)
            items = obj.get("corrections") if isinstance(obj, dict) else obj
            if items is None and isinstance(obj, dict):  # list under another name
                items = next((v for v in obj.values() if isinstance(v, list)), [])
            err = ""
            break
        except Exception as e:  # bad JSON or API error
            err = str(e)[:300]
            if is_daily_limit(err):
                raise DailyLimit(err)
            if "401" in err or "invalid api key" in err.lower():
                raise BadKey(err)
            if attempt >= retries:
                break
            if is_rate_limit(err):
                wait = _retry_after(err, rate_wait)
                if wait > max_wait:          # not a per-minute limit: do not hang for hours
                    raise DailyLimit(err)
                time.sleep(wait)
                continue
            if isinstance(e, ValueError) or "json" in err.lower():
                json_fails += 1
                strict = False   # e.g. Groq "Failed to validate JSON": ask again without JSON mode,
                if json_fails <= 2:  # and extract the JSON from the reply ourselves
                    continue
                break
            other_fails += 1     # network glitch, timeout, server error (5xx): try again
            if other_fails <= 2:
                time.sleep(5)
                continue
            break
    if isinstance(items, dict):  # a single correction instead of a list
        items = [items]
    items = [normalize_item(it) for it in (items if isinstance(items, list) else [])]
    return items, secs, tin, tout, err


# ------------------------------------------------------------ self-check (verifier)
def observed_range(df) -> dict:
    """Observed min/max minutes of each step in the internally consistent cases (no Q1 labels)."""
    st = stats_hints(df)[1]
    return {a: (float(st.loc[a, "min"]), float(st.loc[a, "max"])) for a in ORDER[1:]}


def check_case(df, case, props, rng, tol=5):
    """Check one case after applying the LLM's corrections. Only process rules are used
    (valid values, order, same day, normal step lengths, known hour, untouched consistent events):
    the Q1 answer key is NOT used. Returns a list of problems in plain words."""
    g = df[df.CaseID == case].set_index("Activity")
    if g.empty:
        return []
    probs, latest = [], {}
    for p in props:
        if p["activity"] not in g.index:
            probs.append(f"'{p['activity']}' is not an activity of {case}; use only "
                         + ", ".join(ORDER))
        elif not valid_ts(p["corrected"]):
            probs.append(f"{p['activity']}: '{p['corrected']}' is not a valid YYYY-MM-DDTHH:MM:SS value")
        else:
            latest[p["activity"]] = p["corrected"]
    orig = g.ts.reindex(ORDER)
    t = orig.copy()
    for a, v in latest.items():
        t[a] = pd.to_datetime(v, format=FMT)
    for a in ORDER:
        if pd.isna(t[a]):
            probs.append(f"{a} still has no valid timestamp (original '{g.ts_raw.get(a, '')}')")
    if t.notna().all() and t.dt.date.nunique() > 1:
        probs.append("the corrected timestamps are not all on the same day")
    gaps = t.diff().dt.total_seconds().div(60)
    for i in range(1, len(ORDER)):
        a, b = ORDER[i - 1], ORDER[i]
        gp = gaps[b]
        if pd.isna(gp):
            continue
        if gp <= 0:
            probs.append(f"{a} -> {b} = {gp:.0f} min: time must increase in the process order")
        else:  # every step must have a normal length (an untouched wrong event shows up here too)
            lo, hi = rng[b]
            if not lo - tol <= gp <= hi + tol:
                span = ""
                mid = a if a in latest else b
                pa, na = PREV[mid], NEXT[mid]
                if (pa and na and pd.notna(t[pa]) and pd.notna(t[na])
                        and pa not in latest and na not in latest):  # anchors must be untouched
                    sp = (t[na] - t[pa]).total_seconds() / 60
                    span = f"; {pa} and {na} are {sp:.0f} min apart, so put {mid} about half way"
                probs.append(f"{a} -> {b} = {gp:.0f} min is outside the observed range "
                             f"{lo:.0f}-{hi:.0f} min{span}")
    file_order = g.sort_values("file_pos").index.tolist()
    if file_order == ORDER:
        for i in range(len(ORDER) - 1):
            a, b = ORDER[i], ORDER[i + 1]
            if a in latest and b in latest and latest[a] == _fmt(orig[b]) and latest[b] == _fmt(orig[a]):
                keep = []
                for ev, nb, sign in ((a, PREV[a], 1), (b, NEXT[b], -1)):
                    if nb and pd.notna(orig[ev]) and pd.notna(orig[nb]):
                        gp = (orig[ev] - orig[nb]).total_seconds() / 60 * sign
                        lo, hi = rng[ev if sign == 1 else nb]
                        if lo - tol <= gp <= hi + tol:
                            keep.append(f"{ev} (its original {g.ts_raw[ev]} is {gp:.0f} min from {nb}, "
                                        f"which is normal)")
                probs.append(f"you swapped {a} and {b}, but the rows of {case} are already in process order, "
                             f"so this is NOT a swapped pair: only one of these two timestamps is wrong"
                             + (f". Keep {' and '.join(keep)} and fix only the other one" if keep else ""))
    if file_order != ORDER:
        moved = [a for a in file_order if file_order.index(a) != ORDER.index(a)]
        if moved and not all(a in latest for a in moved):
            probs.append(f"the rows of {case} are not in process order ({' and '.join(moved)} are swapped): "
                         f"both timestamps are suspicious: usually the fix is to give each of them the other's time")
    for a, v in latest.items():
        m = HOUR_RE.match(g.ts_raw[a])
        if m and v[:13] != g.ts_raw[a]:
            probs.append(f"{a}: the original '{g.ts_raw[a]}' still shows hour {m.group(2)}, "
                         f"so the fix must be between {m.group(2)}:00 and {m.group(2)}:59")
    # changed an event whose ORIGINAL value would still fit between its (corrected) neighbours?
    for a in latest:
        if pd.isna(orig[a]):
            continue
        ok, parts = True, []
        for nb, sign in ((PREV[a], 1), (NEXT[a], -1)):
            if nb is None:
                continue
            if pd.isna(t[nb]):
                ok = False
                break
            gp = (orig[a] - t[nb]).total_seconds() / 60 * sign
            lo, hi = rng[a if sign == 1 else nb]
            parts.append(f"{gp:.0f} min")
            ok &= lo - tol <= gp <= hi + tol
        if ok and parts:
            probs.append(f"you changed {a}, but its original value {g.ts_raw[a]} already fits between its "
                         f"neighbours (steps of {' and '.join(parts)}, within the normal range): keep the original")
    return probs


SEVERE = ("still has no valid timestamp", "time must increase", "not a valid", "not an activity",
          "not all on the same day")


def problem_score(problems):
    """Severe problems (broken order, missing/invalid value) weigh 3, mild ones (step length,
    known hour, unnecessary change) weigh 1."""
    return sum(3 if any(k in p for k in SEVERE) else 1 for p in problems)


REFINE = """Your corrections were checked automatically against the process rules, and the cases
below still have problems. For each case you see its rows (with the same hints as before),
your current corrections and the problems found.
Return the COMPLETE list of corrections for THESE cases only. It replaces your earlier answer for
them, so include every event you still want to change, and leave out an event to keep its original value.
"""


def run_llm(df, llm, model_name, method, batch_size=10, pause=0.0, progress=None,
            retries=6, rate_wait=20, rounds=2) -> RunResult:
    """First pass over all cases, then up to `rounds` self-check rounds on the cases with problems."""
    h = hints(df, method)
    system = SYSTEM.format(method_desc=METHOD_DESC[method])
    cases = df.CaseID.drop_duplicates().tolist()
    raw = df.set_index("event_id").ts_raw.to_dict()
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    props = {c: [] for c in cases}           # current corrections per case
    calls, stopped, dropped = [], "", 0
    steps_total = len(batches) + rounds      # for the progress bar

    def log(stage, n, b, secs, tin, tout, err):
        calls.append(dict(model=model_name, method=method, stage=stage, batch=n,
                          cases=f"{b[0]}..{b[-1]}" if b else "", seconds=round(secs, 2),
                          input_tokens=tin, output_tokens=tout, error=err))

    try:
        def ask_split(cs, n, stage):
            """Ask for cases cs; if the answer is empty/broken JSON, split the cases and ask again."""
            its, se, ti, to, er = _ask(llm, system, build_prompt(h, cs, method), retries, rate_wait)
            log(stage, n, cs, se, ti, to, er)
            if er and len(cs) > 1 and ("json" in er.lower() or "empty answer" in er.lower()):
                half = len(cs) // 2
                return (ask_split(cs[:half], n, "first pass (split)")
                        + ask_split(cs[half:], n, "first pass (split)"))
            return its

        for i, b in enumerate(batches):
            items = [it for it in ask_split(b, i + 1, "first pass") if _is_change(it, raw)]
            for it in items:
                if it["case"] in props and it["case"] in b:
                    props[it["case"]].append(it)
                else:
                    props.setdefault("_other", []).append(it)   # invented / outside the batch
            if progress:
                progress((i + 1) / steps_total)
            if pause:
                time.sleep(pause)
        first = _props_frame(props)
        rng = observed_range(df)
        if calls and all(c["error"] for c in calls):
            rounds = 0                     # nothing came back: no point in a self-check
        if rounds and "_other" in props:   # schema check: answers about cases not in the batch
            dropped = len(props.pop("_other"))
        for r in range(1, rounds + 1):
            bad = {c: check_case(df, c, props[c], rng) for c in cases}
            bad = {c: p for c, p in bad.items() if p}
            if not bad:
                break
            todo = list(bad)
            for k in range(0, len(todo), batch_size):
                b = todo[k:k + batch_size]
                prompt = REFINE + "\n" + build_prompt(h, b, method) + "\n\nCurrent corrections and problems:"
                for c in b:
                    cur = [{x: p[x] for x in ("activity", "corrected")} for p in props[c]]
                    prompt += f"\n\n{c}:\n  your corrections: {json.dumps(cur)}\n  problems:\n" + \
                              "\n".join(f"   - {p}" for p in bad[c])
                items, secs, tin, tout, err = _ask(llm, system, prompt, retries, rate_wait)
                log(f"self-check {r}", k // batch_size + 1, b, secs, tin, tout, err)
                if err and len(b) > 1 and ("json" in err.lower() or "empty answer" in err.lower()):
                    # empty / broken answer with several cases: ask again one case at a time
                    items = []
                    for c in b:
                        one = REFINE + "\n" + build_prompt(h, [c], method) + "\n\nCurrent corrections and problems:"
                        cur = [{x: p[x] for x in ("activity", "corrected")} for p in props[c]]
                        one += f"\n\n{c}:\n  your corrections: {json.dumps(cur)}\n  problems:\n" + \
                               "\n".join(f"   - {p}" for p in bad[c])
                        it1, s1, i1, o1, e1 = _ask(llm, system, one, retries, rate_wait)
                        log(f"self-check {r} (single case)", k // batch_size + 1, [c], s1, i1, o1, e1)
                        if not e1:
                            items += it1
                    err = "" if items else err
                if err:          # failed call: keep the previous answer for these cases
                    continue
                for c in b:
                    new = [it for it in items if it["case"] == c and _is_change(it, raw)]
                    # accept only if it is not worse than before (leaving a case out = keep originals)
                    if problem_score(check_case(df, c, new, rng)) <= problem_score(bad[c]):
                        props[c] = new
                if pause:
                    time.sleep(pause)
            if progress:
                progress((len(batches) + r) / steps_total)
    except BadKey as e:
        log("stopped (bad API key)", 0, [], 0.0, 0, 0, str(e)[:300])
        stopped = "The Groq API key was refused (401). Check the key in .streamlit/secrets.toml. " + str(e)[:120]
        first = locals().get("first", _props_frame(props))
    except DailyLimit as e:
        log("stopped (daily limit)", 0, [], 0.0, 0, 0, str(e)[:300])
        stopped = ("Groq's DAILY limit for this model is used up, so the run was stopped. "
                   "Try again later (the error says when) or use another key. " + str(e)[:160])
        first = locals().get("first", _props_frame(props))
    if progress:
        progress(1.0)
    res = RunResult(model_name, method, _props_frame(props), pd.DataFrame(calls, columns=[
        "model", "method", "stage", "batch", "cases", "seconds", "input_tokens", "output_tokens", "error"]))
    res.first = first
    res.stopped = stopped
    res.dropped = dropped
    return res


def _is_change(it, raw):
    """An answer that repeats the original value (or gives no value) is not a correction."""
    v = str(it.get("corrected", "")).strip()
    if v.lower() in ("", "none", "null", "nan", "n/a"):
        return False
    return v != raw.get(f"{it.get('case')}|{it.get('activity')}", None)


def _props_frame(props):
    rows = [dict(it, batch_cases=c) for c, items in props.items() for it in items]
    return pd.DataFrame(rows, columns=["case", "activity", "error_type", "corrected", "reason", "batch_cases"])


def run_rules(df, method) -> pd.DataFrame:
    """Non-LLM reference: the same hints applied by fixed rules (for comparison in Q6)."""
    h = hints(df, method).set_index("event_id")
    fixed = df.set_index("event_id")["ts"].copy()
    out = []
    for case, g in df.groupby("CaseID", sort=False):
        acts = g.Activity.tolist()
        if acts != ORDER:  # out of order -> swap the timestamps of the misplaced pair
            wrong = [a for a in acts if acts.index(a) != ORDER.index(a)]
            if len(wrong) == 2:
                e1, e2 = (f"{case}|{w}" for w in wrong)
                fixed[e1], fixed[e2] = fixed[e2], fixed[e1]
                out += [(case, w, "out_of_order") for w in wrong]
        for a in ORDER:
            eid = f"{case}|{a}"
            r = h.loc[eid]
            prev = fixed.get(f"{case}|{PREV[a]}") if PREV[a] else None
            nxt = fixed.get(f"{case}|{NEXT[a]}") if NEXT[a] else None
            bad = None
            if r.ts_raw == "":
                bad = "missing"
            elif pd.isna(r.ts):
                bad = "malformed"
            elif prev is not None and pd.notna(prev) and fixed[eid] <= prev:
                bad = "negative_duration"
            if not bad:
                continue
            gap = r.expected_gap_min if pd.notna(r.expected_gap_min) else 70
            if prev is not None and pd.notna(prev):
                est = prev + pd.Timedelta(minutes=gap)
            elif nxt is not None and pd.notna(nxt):  # first step missing: go back from the next one
                gnext = h.loc[f"{case}|{NEXT[a]}"].expected_gap_min
                est = nxt - pd.Timedelta(minutes=gnext if pd.notna(gnext) else 70)
            else:
                continue
            if bad == "malformed":  # keep the hour that survived
                m = HOUR_RE.match(r.ts_raw)
                if m:
                    lo = pd.Timestamp(f"{m.group(1)} {m.group(2)}:00")
                    est = min(max(est, lo), lo + pd.Timedelta(minutes=59))
            if (nxt is not None and pd.notna(nxt) and bad != "malformed"
                    and (nxt - est) > pd.Timedelta(minutes=120)):  # keep both steps normal
                est = est + ((nxt - est) - pd.Timedelta(minutes=gap)) / 2
            if nxt is not None and pd.notna(nxt) and est >= nxt:  # stay before the next step
                base = prev if prev is not None and pd.notna(prev) else nxt - pd.Timedelta(minutes=60)
                est = base + (nxt - base) / 2
            fixed[eid] = est.floor("min")
            out.append((case, a, bad))
    res = pd.DataFrame(out, columns=["case", "activity", "error_type"])
    res["corrected"] = [_fmt(fixed[f"{c}|{a}"]) for c, a in zip(res.case, res.activity)]
    res["reason"] = "rule"
    return res.drop_duplicates(["case", "activity"], keep="last")


# ------------------------------------------------------------------ evaluation
def valid_ts(v) -> bool:
    """Right format AND a real date/time (no 2026-02-30 or 25:00)."""
    v = str(v)
    return bool(TS_RE.match(v)) and pd.notna(pd.to_datetime(v, format=FMT, errors="coerce"))


def apply_corrections(df, corr) -> pd.DataFrame:
    fixed = df.copy()
    fixed["Timestamp"] = fixed.ts_raw
    key = fixed.event_id
    for _, c in corr.iterrows():
        m = key == f"{c.case}|{c.activity}"
        if m.any() and valid_ts(c.corrected):
            fixed.loc[m, "Timestamp"] = c.corrected
    return fixed


def _fixed_times(df, real):
    """Timestamps of every event after applying the (valid) corrections, as datetimes."""
    fixed = apply_corrections(df, real)
    return pd.to_datetime(fixed.Timestamp.where(fixed.Timestamp.map(valid_ts)), format=FMT,
                          errors="coerce").set_axis(fixed.event_id)


def _case_times(fixed_t, case):
    return pd.Series({a: fixed_t.get(f"{case}|{a}", pd.NaT) for a in ORDER})


def _case_ok(fixed_t, case):
    t = _case_times(fixed_t, case)
    gaps = t.diff().dt.total_seconds().div(60).iloc[1:]
    return bool(t.notna().all() and (gaps > 0).all() and t.dt.date.nunique() == 1)


def _repair_check(df, fixed_t, eid):
    """Is the repaired value of one wrong event plausible? -> (ok, reason)."""
    case, a = eid.split("|", 1)
    t = _case_times(fixed_t, case)
    if pd.isna(t[a]):
        return False, "no usable value"
    gaps = t.diff().dt.total_seconds().div(60)
    if t.notna().all() and t.dt.date.nunique() > 1:
        return False, "case now spans more than one day"
    if (gaps.iloc[1:] <= 0).any():
        bad = gaps.iloc[1:][gaps.iloc[1:] <= 0].index[0]
        return False, f"order still broken ({PREV[bad]} -> {bad} = {gaps[bad]:.0f} min)"
    i = ORDER.index(a)
    for b in ORDER[max(i, 1):i + 2]:   # gap into this event and into the next one
        if pd.notna(gaps.get(b)) and not 20 <= gaps[b] <= 150:
            why = f"unusual step length ({PREV[b]} -> {b} = {gaps[b]:.0f} min, normal 30-120)"
            lo, hi = PREV[a], NEXT[a]
            if lo and hi and pd.notna(t[lo]) and pd.notna(t[hi]):
                span = (t[hi] - t[lo]).total_seconds() / 60
                if span > 240 or span < 40:  # no value between the neighbours gives two normal steps
                    why += f"; unavoidable: {lo} and {hi} are {span:.0f} min apart"
            return False, why
    raw = df.set_index("event_id").ts_raw[eid]
    m = HOUR_RE.match(raw)
    if m and t[a].strftime("%Y-%m-%dT%H") != raw:
        return False, f"lost the known hour (original '{raw}', got {t[a].strftime('%H:%M')})"
    return True, ""


def diagnose(df, gt, corr) -> pd.DataFrame:
    """One row per LLM answer + one per missed error, with a verdict and the CAUSE."""
    ids = set(df.event_id)
    gt_type = dict(zip(gt.event_id, gt.error_type))
    raw = df.set_index("event_id").ts_raw
    g = _gaps(df).set_index("event_id")
    corr = corr.copy()
    corr["event_id"] = corr["case"].astype(str) + "|" + corr["activity"].astype(str)
    real = corr[corr.event_id.isin(ids)].drop_duplicates("event_id", keep="last")
    fixed_t = _fixed_times(df, real)
    rows, seen = [], set()
    for idx, c in corr.iloc[::-1].iterrows():        # last answer wins, like the scoring
        e = c.event_id
        base = {"case": c["case"], "activity": c["activity"], "original": raw.get(e, ""),
                "corrected": c["corrected"], "LLM reason": c.get("reason", "")}
        if e in seen:
            rows.append({**base, "verdict": "duplicate (ignored)", "cause": "same event answered twice"})
            continue
        seen.add(e)
        if e not in ids:
            known_case = c["case"] in set(df.CaseID)
            cause = (f"activity '{c['activity']}' does not exist" if known_case
                     else f"case '{c['case']}' does not exist")
            rows.append({**base, "verdict": "INVENTED event", "cause": cause})
        elif not valid_ts(c["corrected"]):
            rows.append({**base, "verdict": "INVALID value", "cause": f"'{c['corrected']}' is not a real YYYY-MM-DDTHH:MM:SS"})
        elif e not in gt_type:
            r = g.loc[e]
            parts = [f"gap from previous {r.gap_prev_min:.0f} min" if pd.notna(r.gap_prev_min) else "",
                     f"gap to next {r.gap_next_min:.0f} min" if pd.notna(r.gap_next_min) else ""]
            rows.append({**base, "verdict": "changed a CORRECT event",
                         "cause": "original was valid (" + ", ".join(p for p in parts if p) + ")"})
        else:
            ok, why = _repair_check(df, fixed_t, e)
            rows.append({**base, "verdict": "fixed correctly" if ok else "found, but bad repair",
                         "cause": gt_type[e] + ("" if ok else f": {why}")})
    for e, typ in gt_type.items():
        if e not in set(real.event_id):
            case, a = e.split("|", 1)
            rows.append({"case": case, "activity": a, "original": raw[e], "corrected": "",
                         "LLM reason": "", "verdict": "MISSED error", "cause": typ})
    out = pd.DataFrame(rows[::-1] if rows else [], columns=["verdict", "case", "activity", "original",
                                                             "corrected", "cause", "LLM reason"])
    order = {"fixed correctly": 0, "found, but bad repair": 1, "MISSED error": 2,
             "changed a CORRECT event": 3, "INVENTED event": 4, "INVALID value": 5, "duplicate (ignored)": 6}
    out["_o"] = out.verdict.map(order)
    out["_n"] = out.case.astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    return out.sort_values(["_o", "_n"]).drop(columns=["_o", "_n"]).reset_index(drop=True)


def evaluate(df, gt, corr) -> dict:
    """Detection metrics on all 500 events + quality of the repaired values."""
    ids = set(df.event_id)
    gt_ids = set(gt.event_id)
    corr = corr.copy()
    corr["event_id"] = corr["case"].astype(str) + "|" + corr["activity"].astype(str)
    halluc = corr[~corr.event_id.isin(ids)]                                    # invented events
    real = corr[corr.event_id.isin(ids)].drop_duplicates("event_id", keep="last")
    bad_fmt = real[~real.corrected.map(valid_ts)]               # unusable values
    changed = set(real.event_id)
    tp, fp = len(changed & gt_ids), len(changed - gt_ids)
    fn = len(gt_ids - changed)
    tn = len(ids) - tp - fp - fn
    acc = (tp + tn) / len(ids)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # repair quality: per case, after applying corrections
    fixed_t = _fixed_times(df, real)
    err_cases = set(gt.CaseID)
    good_cases = sum(_case_ok(fixed_t, c) for c in err_cases)
    good_events = sum(_repair_check(df, fixed_t, e)[0] for e in gt_ids & changed)
    n_out = max(len(corr), 1)
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "Correct repair rate": float(good_events / len(gt_ids)),
        "Valid cases after repair": float(good_cases / len(err_cases)),
        "Hallucination rate": (len(halluc) + len(bad_fmt) + fp) / n_out,
        "Invented events": len(halluc), "Invalid values": len(bad_fmt),
    }


def table_2x2(results: dict, metric: str) -> pd.DataFrame:
    """results[(method, model_label)] = metrics dict -> rows = method, columns = model."""
    rows = {METHODS[m]: {lab: results.get((m, lab), {}).get(metric) for lab in MODELS} for m in METHODS}
    return pd.DataFrame(rows).T.astype(float)


def usage_summary(calls: pd.DataFrame) -> pd.DataFrame:
    return calls.groupby(["method", "model"]).agg(
        calls=("batch", "count"), seconds=("seconds", "sum"),
        input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"),
        failed_calls=("error", lambda s: (s != "").sum())).reset_index()


def ml_info(df):
    """Model quality numbers for the GUI/report (cached)."""
    key = ("info", hash("".join(df.event_id) + "".join(df.ts_raw)))
    if key not in _HINT_CACHE:
        _HINT_CACHE[key] = ml_hints(df)[1]
    return _HINT_CACHE[key]

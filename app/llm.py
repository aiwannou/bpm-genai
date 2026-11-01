"""
Thin wrapper around the LLM APIs.

Every call returns an LLMResult with the text AND the cost of the call
(seconds, input tokens, output tokens), because Question 7 asks for a
comparison of execution time and token cost.

Provider: Groq (free tier, no credit card). It hosts several open models from
different companies, e.g. Llama 3.3 70B (Meta), GPT-OSS 120B (OpenAI),
Qwen3 32B (Alibaba), so the two LLMs of Question 5 can both come from it.
"""

import re
import time
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    seconds: float
    input_tokens: int
    output_tokens: int
    model: str


def _strip_thinking(text):
    """Some reasoning models put their reasoning in <think>...</think>; drop it."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


# ----------------------------------------------------------------------
# Groq (OpenAI-compatible API)
# ----------------------------------------------------------------------
class GroqLLM:
    provider = "Groq"
    key_name = "GROQ_API_KEY"
    BASE_URL = "https://api.groq.com/openai/v1"
    SKIP = ("whisper", "tts", "guard", "playai", "orpheus", "compound", "allam", "distil")
    PREFERRED = ("llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b")

    def __init__(self, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL, max_retries=1)
        self.model = model

    @classmethod
    def list_models(cls, api_key):
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=cls.BASE_URL)
        names = [m.id for m in client.models.list()
                 if not any(s in m.id.lower() for s in cls.SKIP)]
        return sorted(set(names))

    @classmethod
    def default_model(cls, names):
        for p in cls.PREFERRED:
            if p in names:
                return p
        return names[0] if names else None

    @classmethod
    def priority(cls, names):
        pref = [p for p in cls.PREFERRED if p in names]
        return pref + [n for n in names if n not in pref]

    def generate(self, system, prompt, json_mode=False):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            **kwargs)
        seconds = time.perf_counter() - t0
        u = resp.usage
        return LLMResult(text=_strip_thinking(resp.choices[0].message.content),
                         seconds=seconds,
                         input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                         output_tokens=getattr(u, "completion_tokens", 0) or 0,
                         model=self.model)


PROVIDERS = {"Groq": GroqLLM}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def friendly_error(e):
    """Turn long API errors into a short message a user can act on."""
    s = str(e)
    if "429" in s or "rate_limit" in s.lower():
        m = re.search(r"(?:retry|try again) in ([\d.]+)\s*s", s, re.I)
        wait = f" Try again in about {int(float(m.group(1))) + 1} seconds." \
            if m and float(m.group(1)) < 300 else " Wait a minute and try again, or choose another model."
        return "Rate limit reached (too many requests for the free tier)." + wait
    if "Invalid API Key" in s or "401" in s:
        return "The API key is not valid. Check the key in .streamlit/secrets.toml."
    if "decommissioned" in s or "404" in s or "does not exist" in s:
        return "This model is not available any more. Choose another model."
    if "403" in s:
        return "Access denied for this model/key. Choose another model."
    if "json_validate_failed" in s or "Failed to generate JSON" in s:
        return "The model did not return valid JSON. Try again or choose another model."
    return s[:300]


def find_working_model(cls, api_key, names, max_tries=8):
    """Send a tiny request to each model until one answers. Returns (model, log)."""
    log = []
    for name in cls.priority(names)[:max_tries]:
        try:
            cls(api_key, name).generate("Reply with OK.", "ping")
            log.append((name, "works"))
            return name, log
        except Exception as e:                      # noqa: BLE001
            log.append((name, friendly_error(e)))
    return None, log

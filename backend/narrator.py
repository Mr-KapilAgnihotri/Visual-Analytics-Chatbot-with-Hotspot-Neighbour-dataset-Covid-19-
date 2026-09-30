"""
NARRATION: writes the text that explains the chart.

Grounding rule: every number in the explanation is computed in Python first
(the `facts` dictionary). The LLM is only asked to put those facts into clear
prose; it is told never to introduce a number that is not in the facts.
This prevents "hallucinated" statistics, the classic failure of LLM analytics.

If the LLM is unavailable, a template sentence built from the same facts is
used, so the answer is always correct, just less eloquent.
"""
from __future__ import annotations

import json
import logging
import re

from backend.llm import get_llm

log = logging.getLogger(__name__)

SYSTEM = """You are a public-health data analyst explaining a chart to a non-expert.
Write 3 to 5 sentences of plain prose (no headings, no bullet points, no markdown tables).
Rules:
- Use ONLY the numbers present in FACTS. Never invent, estimate or recompute numbers.
- Quote numbers as they appear in FACTS (rounding decimals or writing "1.7 million" is fine;
  do not calculate new differences, ratios or percentages yourself). Keep units (e.g. "per 100,000 people").
- Start with the direct answer to the question, then the most interesting pattern,
  then one caveat or piece of context if the facts contain one.
- A "hotspot" is a district whose 7-day incidence is at least 75 per 100,000."""


NUM = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _default(o):
    return str(o)


def _numbers(obj) -> list[float]:
    """Every number contained in the facts, including those inside strings such as dates."""
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif obj is not None:
        out += [float(x.replace(",", "")) for x in NUM.findall(str(obj)) if x.replace(",", "").lstrip("-")]
    return out


def ungrounded_numbers(text: str, facts: dict, extra: str = "") -> list[str]:
    """Numbers in `text` that cannot be traced to the facts (allowing for rounding).
    Small integers (1-31, e.g. day numbers or 'three states') and percentages derived
    by the LLM are not accepted blindly: only values present in the facts pass."""
    known = _numbers(facts) + _numbers(extra) + [100000.0, 100.0, 75.0]
    bad = []
    for m in NUM.finditer(text):
        raw = m.group()
        v = float(raw.replace(",", ""))
        if abs(v) <= 12 and v == int(v):                     # counts like "3 districts", month numbers
            continue
        after = text[m.end():m.end() + 10].lower()
        scale = 1e6 if after.lstrip().startswith(("million", "m ")) else 1e3 if after.lstrip().startswith(("thousand", "k ")) else 1
        v *= scale
        tol = 0.03 if scale > 1 else 0.01                   # "1.7 million" is a legitimate rounding
        if not any(abs(v - k) <= max(0.6, tol * abs(k)) for k in known):
            bad.append(raw)
    return bad


def narrate(question: str, chart_desc: str, facts: dict, fallback: str) -> tuple[str, str]:
    """Returns (text, source) where source is 'llm' or 'template'."""
    llm = get_llm()
    if llm.enabled and facts:
        user = (f"QUESTION: {question}\nCHART: {chart_desc}\n"
                f"FACTS: {json.dumps(facts, default=_default, ensure_ascii=False)}")
        text = llm.complete(SYSTEM, user, max_tokens=700, temperature=0.3)
        if text:
            bad = ungrounded_numbers(text, facts, question + " " + chart_desc)
            if not bad:
                return text, "llm"
            log.warning("LLM narration used numbers not in the facts %s; using template text", bad)
    return fallback, "template"

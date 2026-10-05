"""Deterministic text matching shared by the graph builder and the reasoner.

No statistical NLP: aliases become regular expressions, stems come from a tiny
suffix stripper, sentences are split on punctuation. All of it is readable.
"""
from __future__ import annotations

import re
from functools import lru_cache

_BOUND_L = r"(?<![A-Za-z0-9²])"
_BOUND_R = r"(?![A-Za-z0-9²])"


@lru_cache(maxsize=4096)
def alias_regex(alias: str) -> re.Pattern[str]:
    """Compile an alias. Space/hyphen are interchangeable; trailing * = any word ending."""
    wildcard = alias.endswith("*")
    core = alias[:-1] if wildcard else alias
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", core.strip()) if p]
    body = r"[\s\-]*".join(parts)
    tail = r"[A-Za-z0-9]*" if wildcard else _BOUND_R
    return re.compile(_BOUND_L + body + tail, re.IGNORECASE)


def find_alias(text: str, alias: str) -> bool:
    return bool(alias_regex(alias).search(text or ""))


@lru_cache(maxsize=256)
def cue_regex(name_or_pattern: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + name_or_pattern + r")\b", re.IGNORECASE)


def sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])", text.strip())
    return [p.strip() for p in parts if p.strip()]


STOPWORDS = frozenset("""a an and are as at be been being by can could do does for from has have had how i in into is it
its of on or our that the their them these this those to was we were what when where which while who why will with would
you your using use used want need build building idea system systems new approach method methods paper propose proposed
present based such also more most than then there they very via within across between over under both each other some
any all not only own same so too just like make made makes much many well""".split())

GENERIC = frozenset("""retrieval augmented generation graph knowledge question answering llm llms language large model models rag
reasoning task tasks data information work results study framework""".split())


def stem(token: str) -> str:
    t = token.lower()
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ied", "ed", "es", "ly", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            t = t[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
            break
    return t


def stems(text: str, drop_stopwords: bool = True) -> list[str]:
    out = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9²]+", text or ""):
        if drop_stopwords and tok.lower() in STOPWORDS:
            continue
        out.append(stem(tok))
    return out

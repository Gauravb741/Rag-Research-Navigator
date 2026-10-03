"""Small dependency-free lexical retrieval baseline.

This is deliberately only a baseline and candidate locator. The final system
uses the explicit graph for evidence paths and relation-aware ranking.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to", "using", "with", "where", "will", "would", "i", "want", "we", "our",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()) if token not in STOPWORDS]


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def flat_retrieve(idea: str, papers: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    query = Counter(tokenize(idea))
    documents = [Counter(tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}")) for paper in papers]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    total = max(len(documents), 1)
    query_tfidf = Counter({
        token: count * math.log((1 + total) / (1 + document_frequency[token]))
        for token, count in query.items()
    })
    results = []
    for paper, document in zip(papers, documents):
        weighted = Counter({
            token: count * math.log((1 + total) / (1 + document_frequency[token]))
            for token, count in document.items()
        })
        score = cosine(query_tfidf, weighted)
        results.append({
            "paper_id": paper.get("paper_id", paper.get("id")),
            "title": paper["title"],
            "score": round(score, 6),
            "source_url": paper.get("source_url"),
        })
    results.sort(key=lambda item: (-item["score"], item["title"].lower()))
    return results[:limit]

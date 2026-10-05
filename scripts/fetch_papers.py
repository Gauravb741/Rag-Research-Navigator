#!/usr/bin/env python3
"""Fetch and select a reproducible focused corpus for structured/graph RAG.

The script uses the public OpenAlex Works API. Search results are cached locally,
and the final selection is deterministic: focused lexical scoring plus a small
set of foundational RAG papers needed to interpret the graph-RAG literature.
No API key is required. An optional OPENALEX_MAILTO environment variable can be
used for the polite pool.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = RAW_DIR / "openalex_cache"
TODAY = date.today().isoformat()

QUERIES = [
    "graph retrieval augmented generation",
    "knowledge graph retrieval augmented generation",
    "multi-hop retrieval augmented generation",
    "subgraph retrieval augmented generation",
    "citation graph retrieval language model",
    "document graph retrieval augmented generation",
    "textual graph question answering retrieval",
    "research paper graph retrieval generation",
    "GraphRAG RAG knowledge graph",
    "multi-hop RAG benchmark",
    "research question answering citation graph",
    "academic paper retrieval graph",
    "paper citation recommendation graph neural language model",
    "graph RAG query-focused summarization",
]

CURATED_TITLES = [
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Dense Passage Retrieval for Open-Domain Question Answering",
    "Retrieval-Augmented Generation for Large Language Models: A Survey",
    "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
    "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
    "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    "Active Retrieval Augmented Generation",
    "G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering",
    "GRAG: Graph Retrieval-Augmented Generation",
    "Knowledge Graph-Guided Retrieval Augmented Generation",
    "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models",
    "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries",
    "CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs",
    "Global Discovery: A Global Graph-RAG Approach for Query-Focused Multimodal Summarization Across Multiple PDF Papers",
    "Graph Retrieval-Augmented Generation: A Survey",
    "Knowledge Graph Prompting for Multi-Document Question Answering",
    "GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs",
    "Retrieval-Augmented Generation with Graphs (GraphRAG)",
    "GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation",
    "MAGIC: A Multi-Hop and Graph-Based Benchmark for Inter-Context Conflicts in Retrieval-Augmented Generation",
]

FOCUS_TERMS: dict[str, tuple[int, ...]] = {
    "graph retrieval augmented generation": (12, 8),
    "graph rag": (12, 8),
    "knowledge graph": (10, 6),
    "subgraph": (8, 5),
    "document graph": (10, 6),
    "textual graph": (8, 5),
    "citation graph": (10, 6),
    "citation-aware": (8, 5),
    "multi-hop": (8, 5),
    "multi hop": (8, 5),
    "reasoning path": (5, 3),
    "entity": (3, 2),
    "entities": (3, 2),
    "relation": (3, 2),
    "relations": (3, 2),
    "structured": (3, 2),
    "retrieval augmented": (4, 3),
    "retrieval-augmented": (4, 3),
    "rag": (2, 1),
    "question answering": (2, 1),
    "research paper": (5, 3),
    "academic": (4, 2),
}

FOUNDATIONAL_HINTS = [
    "retrieval-augmented generation for knowledge-intensive nlp tasks",
    "dense passage retrieval for open-domain question answering",
    "retrieval augmented generation for large language models",
    "raptor: recursive abstractive processing",
    "from local to global",
    "g-retriever",
    "self-rag",
    "active retrieval augmented generation",
    "multi-hop-rag",
    "kg^2rag",
    "kg2rag",
    "hipporag",
    "graphrag",
]


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions:
            words.append((position, word))
    return " ".join(word for _, word in sorted(words))


def get_json(url: str) -> Any:
    headers = {
        "User-Agent": "RAGResearchOnboarding/0.1 (OpenAlex corpus research; mailto:research@example.com)",
        "Accept": "application/json",
    }
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}mailto={mailto}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def compact_work(work: dict[str, Any], query: str) -> dict[str, Any]:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    ids = work.get("ids") or {}
    external_ids = {
        "doi": work.get("doi") or ids.get("doi"),
        "openalex": work.get("id") or ids.get("openalex"),
        "mag": ids.get("mag"),
        "pmid": ids.get("pmid"),
    }
    arxiv_id = None
    for candidate in [work.get("doi"), primary_location.get("landing_page_url"), work.get("id")]:
        if candidate:
            match = re.search(r"arxiv(?:\.org)?/(?:abs|pdf)/([0-9.]+)(?:v\d+)?", candidate, re.I)
            if match:
                arxiv_id = match.group(1)
                break
    authors = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("id") or author.get("display_name"):
            authors.append({
                "id": author.get("id"),
                "name": author.get("display_name"),
                "orcid": author.get("orcid"),
            })
    return {
        "paper_id": f"oa_{(work.get('id') or '').rstrip('/').split('/')[-1]}",
        "openalex_id": work.get("id"),
        "title": (work.get("title") or "").strip(),
        "authors": authors,
        "year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "doi": external_ids.get("doi"),
        "arxiv_id": arxiv_id,
        "semantic_scholar_id": None,
        "source_url": primary_location.get("landing_page_url") or work.get("doi") or work.get("id"),
        "openalex_url": work.get("id"),
        "venue": source.get("display_name"),
        "is_oa": bool(work.get("open_access", {}).get("is_oa")),
        "citation_count": work.get("cited_by_count", 0),
        "reference_count": work.get("referenced_works_count", 0),
        "referenced_work_ids": work.get("referenced_works") or [],
        "search_queries": [query],
        "type": work.get("type"),
    }


def score_work(work: dict[str, Any]) -> tuple[int, list[str]]:
    title = work["title"].lower()
    abstract = work["abstract"].lower()
    haystack = f"{title} {abstract}"
    score = 0
    matched: list[str] = []
    for term, (title_weight, abstract_weight) in FOCUS_TERMS.items():
        title_hit = term in title
        abstract_hit = term in abstract
        if title_hit:
            score += title_weight
            matched.append(f"title:{term}")
        elif abstract_hit:
            score += abstract_weight
            matched.append(f"abstract:{term}")
    for hint in FOUNDATIONAL_HINTS:
        if hint in title:
            score += 20
            matched.append(f"foundational:{hint}")
            break
    has_rag = any(term in haystack for term in ("retrieval augmented", "retrieval-augmented", " r a g ", "rag"))
    has_structure = any(term in haystack for term in ("graph", "knowledge", "multi-hop", "multi hop", "citation", "entity", "relation", "structured"))
    if has_rag and has_structure:
        score += 4
    if work.get("abstract"):
        score += 1
    return score, matched


def search_query(query: str, per_page: int = 50) -> dict[str, Any]:
    params = {
        "search": query,
        "filter": "from_publication_date:2020-01-01,to_publication_date:2026-10-02",
        "sort": "relevance_score:desc",
        "per-page": str(per_page),
        "page": "1",
        "select": "id,doi,title,publication_year,publication_date,authorships,abstract_inverted_index,primary_location,open_access,cited_by_count,referenced_works_count,referenced_works,type,ids",
    }
    url = "https://api.openalex.org/works?" + urlencode(params)
    return get_json(url)


def normalize_title(title: str) -> str:
    """Normalize a title for edition/version deduplication."""
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower())
    normalized = re.sub(r"\b(v\d+|version \d+)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=72)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if not 50 <= args.limit <= 100:
        parser.error("--limit must be between 50 and 100")

    for directory in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "OpenAlex Works API",
        "api": "https://api.openalex.org/works",
        "queries": QUERIES,
        "date_range": ["2020-01-01", "2026-10-02"],
        "retrieved_on": TODAY,
        "selection_limit": args.limit,
        "inclusion_strategy": "deterministic lexical focus score over title/abstract plus explicit foundational RAG hints",
    }
    (RAW_DIR / "query_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    candidates: dict[str, dict[str, Any]] = {}
    def add_payload(payload: dict[str, Any], query: str) -> None:
        for raw in payload.get("results", []):
            if not raw.get("id"):
                continue
            compact = compact_work(raw, query)
            key = compact["openalex_id"]
            if key in candidates:
                candidates[key]["search_queries"] = sorted(set(candidates[key]["search_queries"] + [query]))
            else:
                candidates[key] = compact

    for query in QUERIES:
        safe = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
        cache_path = CACHE_DIR / f"{safe}.json"
        if cache_path.exists() and not args.refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = search_query(query)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            time.sleep(0.2)
        add_payload(payload, query)

    for title_query in CURATED_TITLES:
        safe = "title_" + re.sub(r"[^a-z0-9]+", "_", title_query.lower()).strip("_")
        cache_path = CACHE_DIR / f"{safe}.json"
        if cache_path.exists() and not args.refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = search_query(title_query, per_page=5)
            cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            time.sleep(0.15)
        exact = []
        target = normalize_title(title_query)
        for raw in payload.get("results", []):
            title = normalize_title(raw.get("title") or "")
            if title == target or target in title or title in target:
                exact.append(raw)
        add_payload({"results": exact[:1]}, f"curated:{title_query}")

    by_title: dict[str, dict[str, Any]] = {}
    for work in candidates.values():
        key = normalize_title(work["title"])
        prior = by_title.get(key)
        if prior is None or (bool(work["abstract"]), work["citation_count"]) > (bool(prior["abstract"]), prior["citation_count"]):
            by_title[key] = work
    candidates = {work["openalex_id"]: work for work in by_title.values()}

    scored = []
    for work in candidates.values():
        score, matched = score_work(work)
        work["focus_score"] = score
        work["focus_matches"] = matched
        if any(q.startswith("curated:") for q in work["search_queries"]):
            work["focus_score"] += 15
            work["focus_matches"].append("curated_title_match")
        scored.append(work)

    eligible = [work for work in scored if work["title"] and work["focus_score"] >= 12]
    eligible.sort(key=lambda work: (-work["focus_score"], -work["citation_count"], work["title"].lower()))
    selected = eligible[: args.limit]
    if len(selected) < 50:
        raise RuntimeError(f"Only {len(selected)} eligible papers found; broaden queries or inspect API data.")

    for index, paper in enumerate(selected, start=1):
        paper["corpus_index"] = index
        paper["corpus_role"] = "focused_structured_graph_rag"
    corpus = {
        "metadata": {
            **manifest,
            "selected_count": len(selected),
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "exclusion_note": "Papers outside structured/graph/multi-hop/citation-aware RAG were excluded, except foundational RAG baselines needed to explain the selected scope.",
        },
        "papers": selected,
    }
    (PROCESSED_DIR / "papers.json").write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    (RAW_DIR / "all_candidates.json").write_text(json.dumps({"papers": scored}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selected {len(selected)} papers from {len(candidates)} unique OpenAlex candidates ({len(eligible)} eligible).")
    for paper in selected[:10]:
        print(f"{paper['corpus_index']:>2}. {paper['title']} [{paper['year']}] score={paper['focus_score']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

#!/usr/bin/env python3
"""Build the inspectable knowledge state from the selected paper corpus."""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.ontology import TERMS, TermSpec

ROOT = Path(__file__).resolve().parents[1]
PAPERS_PATH = ROOT / "data" / "processed" / "papers.json"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:80]


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def find_evidence(paper: dict[str, Any], alias: str) -> dict[str, str]:
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    alias_lower = alias.lower()
    if alias_lower in title.lower():
        return {
            "evidence": title,
            "source_location": "title",
            "evidence_type": "explicit_title_match",
        }
    for sentence in sentences(abstract):
        if alias_lower in sentence.lower():
            return {
                "evidence": sentence,
                "source_location": "abstract",
                "evidence_type": "explicit_abstract_match",
            }
    return {
        "evidence": abstract[:500] if abstract else title,
        "source_location": "abstract_or_title",
        "evidence_type": "paper_metadata_match",
    }


def contains_alias(text: str, alias: str) -> bool:
    lower = text.lower()
    alias_lower = alias.lower()
    if len(alias_lower) <= 5 and alias_lower.replace("-", "").isalnum():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias_lower)}(?![a-z0-9])", lower))
    return alias_lower in lower


def matching_terms(paper: dict[str, Any]) -> list[tuple[TermSpec, str]]:
    text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}"
    matches: list[tuple[TermSpec, str]] = []
    for spec in TERMS:
        for alias in spec.aliases:
            if contains_alias(text, alias):
                matches.append((spec, alias))
                break
    return matches


def entity(spec: TermSpec) -> dict[str, Any]:
    return {
        "id": spec.entity_id,
        "type": spec.entity_type,
        "name": spec.name,
        "description": spec.description,
        "controlled_vocabulary": True,
        "mapping_rule": "hand-authored alias match in src/modeling/ontology.py",
    }


def relation(source: str, relation_type: str, target: str, evidence: dict[str, Any], source_paper: str | None = None) -> dict[str, Any]:
    return {
        "id": f"rel_{len(evidence.get('_counter', []))}_{source}_{relation_type}_{target}",
        "source": source,
        "relation": relation_type,
        "target": target,
        "evidence": evidence.get("evidence", ""),
        "source_location": evidence.get("source_location", ""),
        "evidence_type": evidence.get("evidence_type", ""),
        "source_paper": source_paper,
    }


def main() -> int:
    corpus = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    papers = corpus["papers"]
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    paper_by_openalex = {paper.get("openalex_id"): paper for paper in papers}
    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    entities: dict[str, dict[str, Any]] = {spec.entity_id: entity(spec) for spec in TERMS}
    relationships: list[dict[str, Any]] = []
    relation_index: set[tuple[str, str, str]] = set()
    relation_counter = 0

    def add_relation(source: str, rel_type: str, target: str, evidence: dict[str, Any], paper_id: str | None = None) -> None:
        nonlocal relation_counter
        key = (source, rel_type, target)
        if key in relation_index:
            return
        relation_index.add(key)
        relation_counter += 1
        relationships.append({
            "id": f"rel_{relation_counter:05d}",
            "source": source,
            "relation": rel_type,
            "target": target,
            "evidence": evidence.get("evidence", ""),
            "source_location": evidence.get("source_location", ""),
            "evidence_type": evidence.get("evidence_type", ""),
            "source_paper": paper_id,
        })

    for paper in papers:
        paper_id = paper["paper_id"]
        title = paper["title"]
        paper_entity = {
            "id": paper_id,
            "type": "Paper",
            "name": title,
            "title": title,
            "year": paper.get("year"),
            "abstract": paper.get("abstract", ""),
            "doi": paper.get("doi"),
            "arxiv_id": paper.get("arxiv_id"),
            "openalex_id": paper.get("openalex_id"),
            "source_url": paper.get("source_url"),
            "citation_count": paper.get("citation_count", 0),
            "corpus_index": paper.get("corpus_index"),
            "corpus_role": paper.get("corpus_role"),
        }
        entities[paper_id] = paper_entity
        for author in paper.get("authors", []):
            if not author.get("name"):
                continue
            author_id = "author_" + slug(author["name"])
            if author_id not in entities:
                entities[author_id] = {
                    "id": author_id,
                    "type": "Author",
                    "name": author["name"],
                    "orcid": author.get("orcid"),
                    "mapping_rule": "OpenAlex authorship metadata",
                }
            add_relation(paper_id, "AUTHORED_BY", author_id, {
                "evidence": f"OpenAlex authorship metadata lists {author['name']}.",
                "source_location": "metadata.authorships",
                "evidence_type": "direct_metadata",
            }, paper_id)

        matches = matching_terms(paper)
        for spec, alias in matches:
            if spec.entity_id.startswith("method_"):
                rel_type = "INTRODUCES_METHOD" if contains_alias(title, alias) else "USES_METHOD"
            elif spec.entity_type == "ResearchProblem":
                rel_type = "ADDRESSES_PROBLEM"
            elif spec.entity_type == "Dataset":
                rel_type = "EVALUATES_ON"
            elif spec.entity_type == "Metric":
                rel_type = "USES_METRIC"
            elif spec.entity_type == "Architecture":
                rel_type = "IMPLEMENTS_ARCHITECTURE"
            elif spec.entity_type == "Limitation":
                rel_type = "HAS_LIMITATION"
            elif spec.entity_type == "ResearchDirection":
                rel_type = "SUGGESTS_DIRECTION"
            else:
                rel_type = "RELATES_TO_CONCEPT"
            evidence = find_evidence(paper, alias)
            evidence["matched_alias"] = alias
            add_relation(paper_id, rel_type, spec.entity_id, evidence, paper_id)

        for referenced in paper.get("referenced_work_ids", []):
            target = paper_by_openalex.get(referenced)
            if target:
                add_relation(paper_id, "CITES", target["paper_id"], {
                    "evidence": f"OpenAlex lists {target['title']} among the references of {title}.",
                    "source_location": "metadata.referenced_works",
                    "evidence_type": "citation_metadata",
                }, paper_id)

    for rel in list(relationships):
        if rel["relation"] != "CITES":
            continue
        citing = paper_by_id.get(rel["source"])
        cited = paper_by_id.get(rel["target"])
        if not citing or not cited:
            continue
        later_text = f"{citing.get('title', '')} {citing.get('abstract', '')}".lower()
        cue = next((term for term in ("extend", "enhanc", "improv", "build on", "based on", "inspired by", "follow-up") if term in later_text), None)
        if cue:
            add_relation(rel["source"], "BUILDS_ON", rel["target"], {
                "evidence": f"{citing['title']} contains the improvement/extension cue '{cue}' and cites the earlier paper.",
                "source_location": "title_or_abstract_plus_citation_metadata",
                "evidence_type": "rule_derived_citation_and_text",
            }, rel["source"])

    state = {
        "metadata": {
            "schema_version": "1.0",
            "generated_on": date.today().isoformat(),
            "source_corpus": "data/processed/papers.json",
            "ontology_source": "src/modeling/ontology.py",
            "construction_method": "controlled-vocabulary lexical mapping plus in-corpus OpenAlex citation edges",
            "human_auditable": True,
            "automatic_kg_library_used": False,
            "entity_type_counts": {},
            "relationship_type_counts": {},
            "corpus_size": len(papers),
        },
        "entities": list(entities.values()),
        "relationships": relationships,
        "papers": [paper_by_id[paper["paper_id"]] for paper in papers],
    }
    for item in state["entities"]:
        state["metadata"]["entity_type_counts"][item["type"]] = state["metadata"]["entity_type_counts"].get(item["type"], 0) + 1
    for item in relationships:
        state["metadata"]["relationship_type_counts"][item["relation"]] = state["metadata"]["relationship_type_counts"].get(item["relation"], 0) + 1

    (KNOWLEDGE_DIR / "knowledge_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    (KNOWLEDGE_DIR / "entities.json").write_text(json.dumps(state["entities"], indent=2, ensure_ascii=False), encoding="utf-8")
    (KNOWLEDGE_DIR / "relationships.json").write_text(json.dumps(state["relationships"], indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "corpus_size": len(papers),
        "entity_count": len(state["entities"]),
        "relationship_count": len(relationships),
        "entity_type_counts": state["metadata"]["entity_type_counts"],
        "relationship_type_counts": state["metadata"]["relationship_type_counts"],
    }
    (KNOWLEDGE_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

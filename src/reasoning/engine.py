"""Graph-aware reasoning for unseen RAG research ideas."""
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.retrieval.lexical import flat_retrieve
from src.modeling.ontology import TERMS, TermSpec


def contains_alias(text: str, alias: str) -> bool:
    lower = text.lower()
    alias_lower = alias.lower()
    if len(alias_lower) <= 5 and alias_lower.replace("-", "").isalnum():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(alias_lower)}(?![a-z0-9])", lower))
    return alias_lower in lower


class GraphReasoner:
    """Reason over a serialized graph, not just a vector or text index."""

    def __init__(self, state: dict[str, Any]):
        self.state = state
        self.entities = {item["id"]: item for item in state.get("entities", [])}
        self.relationships = state.get("relationships", [])
        self.papers = {item["id"]: item for item in state.get("entities", []) if item.get("type") == "Paper"}
        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relationship in self.relationships:
            self.outgoing[relationship["source"]].append(relationship)
            self.incoming[relationship["target"]].append(relationship)
        self.relationship_by_id = {item["id"]: item for item in self.relationships}
        self.term_by_id = {term.entity_id: term for term in TERMS}

    def match_terms(self, idea: str) -> list[dict[str, Any]]:
        matches = []
        for term in TERMS:
            aliases = [alias for alias in term.aliases if contains_alias(idea, alias)]
            if aliases:
                matches.append({
                    "entity_id": term.entity_id,
                    "type": term.entity_type,
                    "name": term.name,
                    "matched_aliases": aliases,
                })
        return matches

    def _paper_path_from_seed(self, seed_id: str, paper_id: str) -> list[dict[str, Any]] | None:
        for rel in self.incoming.get(seed_id, []):
            if rel["source"] == paper_id:
                return [{
                    "from": seed_id,
                    "from_name": self.entities.get(seed_id, {}).get("name", seed_id),
                    "relation": f"{rel['relation']} (reverse)",
                    "to": paper_id,
                    "to_name": self.papers.get(paper_id, {}).get("name", paper_id),
                    "evidence_id": rel["id"],
                }]
        return None

    def _relation_step(self, rel: dict[str, Any], reverse: bool = False) -> dict[str, Any]:
        source, target = (rel["target"], rel["source"]) if reverse else (rel["source"], rel["target"])
        return {
            "from": source,
            "from_name": self.entities.get(source, {}).get("name", source),
            "relation": f"{rel['relation']} (reverse)" if reverse else rel["relation"],
            "to": target,
            "to_name": self.entities.get(target, {}).get("name", target),
            "evidence_id": rel["id"],
        }

    def _candidate_papers(self, matched: list[dict[str, Any]], idea: str) -> list[dict[str, Any]]:
        seed_ids = {item["entity_id"] for item in matched}
        scores: dict[str, float] = defaultdict(float)
        paths: dict[str, list[dict[str, Any]]] = {}
        matched_for_paper: dict[str, list[str]] = defaultdict(list)

        for seed_id in seed_ids:
            for rel in self.incoming.get(seed_id, []):
                if rel["source"] not in self.papers:
                    continue
                paper_id = rel["source"]
                scores[paper_id] += 5.0 if self.entities.get(seed_id, {}).get("type") in {"ResearchProblem", "Method"} else 3.0
                matched_for_paper[paper_id].append(seed_id)
                paths.setdefault(paper_id, self._paper_path_from_seed(seed_id, paper_id) or [])

        for paper_id in list(scores):
            for rel in self.outgoing.get(paper_id, []):
                if rel["relation"] not in {"CITES", "BUILDS_ON"}:
                    continue
                if rel["target"] in self.papers:
                    scores[rel["target"]] += 1.5
                    if rel["target"] not in paths and paths.get(paper_id):
                        paths[rel["target"]] = paths[paper_id] + [self._relation_step(rel)]

        lexical = {item["paper_id"]: item["score"] for item in flat_retrieve(idea, list(self.papers.values()), limit=len(self.papers))}
        ranked = []
        for paper_id, score in scores.items():
            paper = self.papers[paper_id]
            total = score + lexical.get(paper_id, 0.0) * 2.0 + min(float(paper.get("citation_count", 0)), 100) / 1000
            ranked.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "year": paper.get("year"),
                "source_url": paper.get("source_url"),
                "graph_score": round(score, 4),
                "lexical_tiebreak": round(lexical.get(paper_id, 0.0), 4),
                "score": round(total, 4),
                "matched_entities": [self.entities[item]["name"] for item in dict.fromkeys(matched_for_paper[paper_id])],
                "seed_path": paths.get(paper_id, []),
            })
        ranked.sort(key=lambda item: (-item["score"], -(item.get("year") or 0), item["title"].lower()))
        return ranked

    def _paths_from_paper(self, paper_id: str, limit: int = 4) -> list[dict[str, Any]]:
        paths: list[dict[str, Any]] = []
        for rel in self.outgoing.get(paper_id, []):
            target = self.entities.get(rel["target"], {})
            if target.get("type") in {"Limitation", "ResearchDirection", "ResearchProblem"}:
                paths.append({
                    "path_type": f"paper_to_{target['type'].lower()}",
                    "nodes": [self.papers[paper_id]["name"], target.get("name", rel["target"])],
                    "steps": [self._relation_step(rel)],
                    "evidence_ids": [rel["id"]],
                })
            if rel["relation"] in {"CITES", "BUILDS_ON"} and rel["target"] in self.papers:
                cited_id = rel["target"]
                for second in self.outgoing.get(cited_id, []):
                    second_target = self.entities.get(second["target"], {})
                    if second_target.get("type") in {"Method", "ResearchProblem", "Limitation", "ResearchDirection", "Dataset"}:
                        paths.append({
                            "path_type": "paper_to_related_paper_to_entity",
                            "nodes": [self.papers[paper_id]["name"], self.papers[cited_id]["name"], second_target.get("name", second["target"])],
                            "steps": [self._relation_step(rel), self._relation_step(second)],
                            "evidence_ids": [rel["id"], second["id"]],
                        })
        paths.sort(key=lambda path: (
            0 if len(path["steps"]) > 1 else 1,
            0 if "limitation" in path["path_type"] or "direction" in path["path_type"] else 1,
            len(path["steps"]),
        ))
        return paths[:limit]

    def _unique_relation_targets(self, paper_ids: list[str], types: set[str], relation_types: set[str]) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for paper_id in paper_ids:
            for rel in self.outgoing.get(paper_id, []):
                target = self.entities.get(rel["target"])
                if not target or target.get("type") not in types or rel["relation"] not in relation_types:
                    continue
                found.setdefault(target["id"], {
                    "id": target["id"],
                    "name": target["name"],
                    "type": target["type"],
                    "supporting_papers": [],
                    "evidence_ids": [],
                })
                found[target["id"]]["supporting_papers"].append(self.papers[paper_id]["name"])
                found[target["id"]]["evidence_ids"].append(rel["id"])
        return sorted(found.values(), key=lambda item: (-len(item["supporting_papers"]), item["name"]))

    def reason(self, idea: str, limit: int = 8) -> dict[str, Any]:
        matched = self.match_terms(idea)
        candidates = self._candidate_papers(matched, idea)
        top = candidates[:limit]
        top_ids = [item["paper_id"] for item in top]
        paths = []
        for candidate in top[:5]:
            for path in self._paths_from_paper(candidate["paper_id"], limit=4):
                path["source_paper_id"] = candidate["paper_id"]
                paths.append(path)
        limitations = self._unique_relation_targets(top_ids, {"Limitation"}, {"HAS_LIMITATION"})
        directions = self._unique_relation_targets(top_ids, {"ResearchDirection"}, {"SUGGESTS_DIRECTION"})
        problems = self._unique_relation_targets(top_ids, {"ResearchProblem"}, {"ADDRESSES_PROBLEM"})
        methods = self._unique_relation_targets(top_ids, {"Method"}, {"USES_METHOD", "INTRODUCES_METHOD"})
        datasets = self._unique_relation_targets(top_ids, {"Dataset"}, {"EVALUATES_ON"})
        architectures = self._unique_relation_targets(top_ids, {"Architecture"}, {"IMPLEMENTS_ARCHITECTURE"})

        cited_papers = []
        seen = set()
        for paper_id in top_ids:
            for rel in self.outgoing.get(paper_id, []):
                if rel["relation"] not in {"CITES", "BUILDS_ON"} or rel["target"] not in self.papers or rel["target"] in seen:
                    continue
                seen.add(rel["target"])
                cited = self.papers[rel["target"]]
                cited_papers.append({
                    "paper_id": rel["target"],
                    "title": cited["title"],
                    "relationship": rel["relation"],
                    "evidence_ids": [rel["id"]],
                    "source_url": cited.get("source_url"),
                })
        recommended = []
        for item in top[:6]:
            recommended.append({
                "order": len(recommended) + 1,
                "paper_id": item["paper_id"],
                "title": item["title"],
                "reason": "Directly connected to matched concepts/problems/methods through graph edges.",
                "evidence_ids": [step["evidence_id"] for step in item.get("seed_path", [])],
            })
        for cited in cited_papers[: max(0, 6 - len(recommended))]:
            recommended.append({
                "order": len(recommended) + 1,
                "paper_id": cited["paper_id"],
                "title": cited["title"],
                "reason": f"{cited['relationship']} connection from a directly related paper.",
                "evidence_ids": cited["evidence_ids"],
            })

        uncertainty = [
            "The system reports overlap within the indexed corpus; it does not prove or disprove absolute novelty.",
            "Controlled-vocabulary matches are conservative and may miss terminology not present in the ontology.",
            "Relationship evidence is derived from paper metadata and abstract/title lexical matches; inspect the evidence IDs before relying on a claim.",
        ]
        if not matched:
            uncertainty.insert(0, "No controlled-vocabulary entity matched directly; results rely on the lexical baseline only and should be treated as exploratory.")
        if not candidates:
            uncertainty.insert(0, "No graph-connected papers were found for the input; broaden the ontology or corpus before interpreting the result.")

        evidence_ids = []
        for candidate in top:
            evidence_ids.extend(step["evidence_id"] for step in candidate.get("seed_path", []))
        for path in paths:
            evidence_ids.extend(path["evidence_ids"])
        for item in limitations + directions + problems + methods + datasets + architectures:
            evidence_ids.extend(item["evidence_ids"])
        evidence = []
        for evidence_id in dict.fromkeys(evidence_ids):
            rel = self.relationship_by_id.get(evidence_id)
            if rel:
                evidence.append({
                    "evidence_id": evidence_id,
                    "source_paper_id": rel.get("source_paper"),
                    "relation": rel["relation"],
                    "source": rel["source"],
                    "target": rel["target"],
                    "quote": rel.get("evidence"),
                    "source_location": rel.get("source_location"),
                    "evidence_type": rel.get("evidence_type"),
                })

        return {
            "input_summary": idea.strip(),
            "matched_concepts": matched,
            "related_methods": methods,
            "related_architectures": architectures,
            "related_problems": problems,
            "related_datasets": datasets,
            "related_papers": top,
            "foundational_connections": cited_papers,
            "multi_hop_evidence_paths": paths,
            "known_limitations": limitations,
            "research_directions": directions,
            "recommended_reading_order": recommended,
            "evidence": evidence,
            "flat_retrieval_baseline": flat_retrieve(idea, list(self.papers.values()), limit=5),
            "uncertainty_notes": uncertainty,
            "reasoning_summary": {
                "method": "controlled-vocabulary seed matching -> graph relation traversal -> citation/path expansion -> evidence attachment",
                "graph_used": bool(matched and candidates),
                "candidate_count": len(candidates),
                "returned_paper_count": len(top),
            },
        }

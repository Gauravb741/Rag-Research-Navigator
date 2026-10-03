#!/usr/bin/env python3
"""Evaluate the graph reasoner against manually prepared unseen ideas.

The test cases contain expected useful entities, not gold answers. Scores are
coverage diagnostics for this project and should not be read as universal
benchmark results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.engine import GraphReasoner
from src.retrieval.lexical import flat_retrieve

STATE = ROOT / "data" / "knowledge" / "knowledge_state.json"
CASES = ROOT / "data" / "evaluation" / "test_cases.json"
OUT = ROOT / "data" / "evaluation" / "evaluation_results.json"


def output_entity_ids(report: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    ids.update(item["entity_id"] for item in report.get("matched_concepts", []))
    for key in ("related_methods", "related_architectures", "related_problems", "related_datasets", "known_limitations", "research_directions"):
        ids.update(item["id"] for item in report.get(key, []))
    return ids


def baseline_entity_ids(reasoner: GraphReasoner, idea: str) -> set[str]:
    ids: set[str] = set()
    for item in flat_retrieve(idea, list(reasoner.papers.values()), limit=8):
        for rel in reasoner.outgoing.get(item["paper_id"], []):
            target = reasoner.entities.get(rel["target"])
            if target and target.get("type") != "Author":
                ids.add(target["id"])
    return ids


def main() -> int:
    reasoner = GraphReasoner(json.loads(STATE.read_text(encoding="utf-8")))
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        report = reasoner.reason(case["input"], limit=8)
        expected = set(case["expected_entities"])
        graph_ids = output_entity_ids(report)
        baseline_ids = baseline_entity_ids(reasoner, case["input"])
        graph_hits = sorted(expected & graph_ids)
        baseline_hits = sorted(expected & baseline_ids)
        results.append({
            "id": case["id"],
            "expected_entity_count": len(expected),
            "graph_entity_hits": graph_hits,
            "baseline_entity_hits": baseline_hits,
            "graph_entity_recall": round(len(graph_hits) / len(expected), 4) if expected else 0.0,
            "flat_baseline_entity_recall": round(len(baseline_hits) / len(expected), 4) if expected else 0.0,
            "path_count": len(report.get("multi_hop_evidence_paths", [])),
            "graph_used": report.get("reasoning_summary", {}).get("graph_used", False),
            "evidence_count": len(report.get("evidence", [])),
            "evidence_grounded": all(item.get("source_paper_id") and item.get("quote") for item in report.get("evidence", [])),
            "top_papers": [item["title"] for item in report.get("related_papers", [])[:5]],
        })
    summary = {
        "test_case_count": len(results),
        "average_graph_entity_recall": round(sum(item["graph_entity_recall"] for item in results) / len(results), 4) if results else 0.0,
        "average_flat_baseline_entity_recall": round(sum(item["flat_baseline_entity_recall"] for item in results) / len(results), 4) if results else 0.0,
        "cases_with_graph_paths": sum(bool(item["path_count"]) for item in results),
        "cases_with_grounded_evidence": sum(item["evidence_grounded"] for item in results),
    }
    output = {"method_note": "Coverage over manually specified expected useful connections; not a universal benchmark.", "summary": summary, "cases": results}
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

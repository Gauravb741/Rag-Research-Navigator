import json
from pathlib import Path

from src.reasoning.engine import GraphReasoner
from src.retrieval.lexical import flat_retrieve

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / "data/knowledge/knowledge_state.json").read_text(encoding="utf-8"))


def reasoner():
    return GraphReasoner(STATE)


def test_knowledge_state_is_inspectable_and_focused():
    assert STATE["metadata"]["corpus_size"] >= 50
    assert STATE["metadata"]["corpus_size"] <= 100
    assert STATE["metadata"]["human_auditable"] is True
    assert STATE["metadata"]["automatic_kg_library_used"] is False
    assert len(STATE["entities"]) >= STATE["metadata"]["corpus_size"]
    assert len(STATE["relationships"]) >= 100


def test_relationships_have_provenance():
    relationships = STATE["relationships"]
    assert relationships
    assert all(item["id"] and item["source"] and item["relation"] and item["target"] for item in relationships)
    assert sum(bool(item.get("evidence")) for item in relationships) / len(relationships) > 0.9
    assert {item["relation"] for item in relationships} >= {"CITES", "ADDRESSES_PROBLEM", "MOTIVATED_BY_LIMITATION"}


def test_graph_reasoner_returns_paths_and_grounded_evidence():
    report = reasoner().reason(
        "I want an explainable RAG system for multi-hop research question answering "
        "that follows citation networks and entity-relation paths across papers."
    )
    assert report["reasoning_summary"]["graph_used"] is True
    assert report["related_papers"]
    assert report["multi_hop_evidence_paths"]
    assert report["evidence"]
    assert all(item["evidence_id"] for item in report["evidence"])
    assert any(item["entity_id"] == "concept_citation_network" for item in report["matched_concepts"])


def test_unrelated_input_does_not_invent_graph_connections():
    report = reasoner().reason("I want to design a low-cost hiking recipe planner for weekends.")
    assert report["reasoning_summary"]["graph_used"] is False
    assert report["related_papers"] == []
    assert any("No controlled-vocabulary" in note for note in report["uncertainty_notes"])


def test_flat_retrieval_baseline_is_sorted_and_separate():
    papers = [item for item in STATE["entities"] if item.get("type") == "Paper"]
    results = flat_retrieve("graph based multi-hop retrieval", papers, limit=5)
    assert len(results) == 5
    assert all(results[i]["score"] >= results[i + 1]["score"] for i in range(len(results) - 1))
    assert all("paper_id" in item and "title" in item for item in results)

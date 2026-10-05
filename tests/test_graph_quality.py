import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reasoning.engine import GraphReasoner  # noqa: E402

STATE_PATH = ROOT / "data" / "knowledge" / "knowledge_state.json"
STATE = json.loads(STATE_PATH.read_text(encoding="utf-8"))
REASONER = GraphReasoner(STATE)
ENT = {e["id"]: e for e in STATE["entities"]}
IDEA = ("An explainable RAG system for research question answering that follows citation graphs "
        "for multi-hop questions and keeps the graph updated incrementally.")


def test_every_edge_has_rule_confidence_and_evidence():
    for r in STATE["relationships"]:
        assert r["rule"] in STATE["rules"], r["id"]
        assert r["confidence"] in {"high", "medium", "low"}
        assert r["evidence"], r["id"]
        assert r["relation"] in STATE["schema"]["relation_types"], r["relation"]


def test_graph_has_entity_to_entity_edges():
    assert STATE["metadata"]["entity_to_entity_edge_count"] >= 100
    types = {r["relation"] for r in STATE["relationships"]}
    assert {"IMPROVES_ON", "INSTANCE_OF", "TARGETS_PROBLEM", "MOTIVATED_BY", "MOTIVATES"} <= types


def test_paradigms_are_never_introduced_by_a_paper():
    for r in STATE["relationships"]:
        if r["relation"] == "INTRODUCES_METHOD":
            assert ENT[r["target"]].get("kind") != "paradigm"
    # the old rule made dozens of papers "introduce" standard RAG / GraphRAG
    assert sum(1 for r in STATE["relationships"] if r["relation"] == "INTRODUCES_METHOD") <= 45


def test_limitation_edges_need_a_limitation_cue_sentence():
    lims = [r for r in STATE["relationships"] if r["relation"] == "MOTIVATED_BY_LIMITATION"]
    assert lims and all(r["rule"] == "limitation_cue_sentence" and r["source_location"] == "abstract" for r in lims)


def test_report_contains_typed_entity_chains_and_gap_analysis():
    report = REASONER.reason(IDEA)
    chains = [p for p in report["multi_hop_evidence_paths"] if p["path_type"] == "entity_chain"]
    assert chains and all(len(c["steps"]) >= 1 and all("relation" in s for s in c["steps"]) for c in chains)
    og = report["overlap_and_gap"]
    for key in ("idea_facets", "closest_prior_work", "covered_combinations", "uncovered_combinations", "novelty_signal"):
        assert key in og
    assert report["recommended_reading_order"] and all(item["reason"] for item in report["recommended_reading_order"])
    assert "unmatched_terms" in report


def test_off_topic_input_gets_no_graph_and_no_novelty_claim():
    report = REASONER.reason("Best hiking trails and a sourdough recipe for the weekend.")
    assert report["related_papers"] == []
    assert report["overlap_and_gap"]["novelty_signal"] == "insufficient_ontology_coverage"


def test_near_duplicate_is_not_called_novel():
    report = REASONER.reason("A RAG system that represents knowledge as a hypergraph with n-ary relations instead of "
                             "binary-relation graphs, evaluated across medicine, agriculture and law.")
    assert report["overlap_and_gap"]["novelty_signal"] in {"close_match_found", "partial_overlap"}
    assert any("HyperGraphRAG" in p["title"] for p in report["related_papers"][:3])


def test_unfamiliar_wording_is_reported_not_hidden():
    report = REASONER.reason("Temporal knowledge graphs with validity intervals on every edge for evolving facts.")
    assert any(u["term"].startswith("tempor") for u in report["unmatched_terms"])


def test_excluded_papers_never_appear_anywhere():
    pid = next(p["id"] for p in STATE["entities"] if p["type"] == "Paper" and "CG-RAG" in p["title"])
    report = REASONER.reason(IDEA, exclude_paper_ids={pid})
    assert pid not in {p["paper_id"] for p in report["related_papers"]}
    assert pid not in {p["paper_id"] for p in report["foundational_connections"]}
    assert pid not in {p["paper_id"] for p in report["recommended_reading_order"]}


def test_ablation_flags_change_scores():
    full = REASONER.reason(IDEA, limit=10)
    plain = REASONER.reason(IDEA, limit=10, use_citation_expansion=False, use_propagation=False)
    assert [p["score"] for p in full["related_papers"]] != [p["score"] for p in plain["related_papers"]]


def test_build_is_deterministic():
    before = hashlib.sha256(STATE_PATH.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_knowledge.py")], check=True, capture_output=True, cwd=ROOT)
    assert hashlib.sha256(STATE_PATH.read_bytes()).hexdigest() == before

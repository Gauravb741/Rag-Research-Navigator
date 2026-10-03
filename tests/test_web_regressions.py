import copy
import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from src.interface.web import GraphReasoner, INDEX_HTML, SecureHandler, public_graph

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / "data/knowledge/knowledge_state.json").read_text(encoding="utf-8"))

REASON_LIST_FIELDS = (
    "matched_concepts",
    "related_methods",
    "related_architectures",
    "related_problems",
    "related_datasets",
    "known_limitations",
    "research_directions",
    "related_papers",
    "multi_hop_evidence_paths",
    "evidence",
    "recommended_reading_order",
    "uncertainty_notes",
)


def request_reason(idea: str) -> tuple[int, dict]:
    SecureHandler.reasoner = GraphReasoner(STATE)
    server = ThreadingHTTPServer(("127.0.0.1", 0), SecureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        connection = http.client.HTTPConnection(host, port, timeout=10)
        connection.request(
            "POST",
            "/api/reason",
            body=json.dumps({"idea": idea}),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        return response.status, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_reason_input_matrix_returns_structured_safe_states():
    queries = {
        "simple": "GraphRAG",
        "medium": "How can graph-based retrieval improve RAG for multi-hop questions where information is distributed across several documents?",
        "complex": "I want to develop a graph-based retrieval-augmented generation system that models relationships between documents, entities, methods, and citations so that a researcher can answer multi-hop questions requiring evidence distributed across several research papers.",
        "long": "I want to develop and evaluate an explainable graph-based retrieval-augmented generation system for scientific research assistance. The system should model relationships between papers, documents, entities, methods, datasets, limitations, research directions, and citations, then retrieve compact subgraphs for multi-hop questions whose supporting evidence is distributed across several research papers. I also want to compare graph traversal with ordinary chunk retrieval, measure evidence provenance, citation faithfulness, context relevance, answer completeness, and latency, and study how the system handles noisy, conflicting, or incomplete relationships.",
    }
    successful = []
    for name, idea in queries.items():
        status, payload = request_reason(idea)
        if name == "simple":
            assert status == 400
            assert payload.get("error")
            continue
        assert status == 200
        successful.append(payload)
        for field in REASON_LIST_FIELDS:
            assert isinstance(payload.get(field), list), (name, field)
        assert isinstance(payload.get("reasoning_summary"), dict)
        assert isinstance(payload["related_papers"], list)
        assert isinstance(payload["multi_hop_evidence_paths"], list)
        assert isinstance(payload["evidence"], list)
    assert len(successful) == 3


def test_complex_response_ids_are_valid_graph_contracts():
    status, report = request_reason(
        "I want to develop a graph-based retrieval-augmented generation system that models relationships between documents, entities, methods, and citations so that a researcher can answer multi-hop questions requiring evidence distributed across several research papers."
    )
    assert status == 200
    graph = public_graph(STATE)
    entity_ids = {item["id"] for item in graph["entities"]}
    relation_ids = {item["id"] for item in graph["relationships"]}
    report_entity_ids = {
        item["entity_id"] for item in report["matched_concepts"]
    }
    for key in ("related_methods", "related_architectures", "related_problems", "related_datasets", "known_limitations", "research_directions"):
        report_entity_ids.update(item["id"] for item in report[key])
    report_entity_ids.update(item["paper_id"] for item in report["related_papers"])
    assert report_entity_ids <= entity_ids
    assert all(item["evidence_id"] in relation_ids for item in report["evidence"])
    assert all(
        evidence_id in relation_ids
        for path in report["multi_hop_evidence_paths"]
        for evidence_id in path["evidence_ids"]
    )


def test_unrelated_query_has_explicit_empty_result_contract():
    status, report = request_reason("quantum banana orchard robotics")
    assert status == 200
    assert report["reasoning_summary"]["graph_used"] is False
    assert report["related_papers"] == []
    assert report["multi_hop_evidence_paths"] == []
    assert report["evidence"] == []
    assert "No controlled-vocabulary" in " ".join(report["uncertainty_notes"])
    assert "No strong graph connections were found" in INDEX_HTML


def test_frontend_has_normalization_error_and_selection_guards():
    assert "normalizeGraphPayload" in INDEX_HTML
    assert "normalizeReportPayload" in INDEX_HTML
    assert "renderGraphMessage" in INDEX_HTML
    assert "The graph could not be rendered." in INDEX_HTML
    assert "entityById(state.selectedId)" in INDEX_HTML
    assert "text.style.pointerEvents = \"auto\"" in INDEX_HTML
    assert "requestId !== state.requestId" in INDEX_HTML
    assert "The request could not be completed." in INDEX_HTML
    assert "This node is no longer available." in INDEX_HTML
    assert "This relationship is no longer available." in INDEX_HTML


def test_all_supported_entity_types_have_safe_inspector_path():
    graph = public_graph(STATE)
    types = {item["type"] for item in graph["entities"]}
    expected = {"Paper", "Method", "ResearchProblem", "Dataset", "Metric", "Architecture", "Limitation", "ResearchDirection", "Concept", "Author"}
    assert expected <= types
    assert "function renderInspector()" in INDEX_HTML
    assert "if (item.description)" in INDEX_HTML
    assert "if (item.abstract)" in INDEX_HTML
    assert "No connected entities" in INDEX_HTML


def test_public_graph_tolerates_missing_optional_entity_fields():
    altered = copy.deepcopy(STATE)
    altered["entities"][0].pop("abstract", None)
    altered["entities"][0].pop("description", None)
    result = public_graph(altered)
    assert len(result["entities"]) == len(STATE["entities"])
    assert result["entities"][0]["id"]

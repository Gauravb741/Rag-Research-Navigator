#!/usr/bin/env python3
"""Fast final compliance checks for the repository."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.engine import GraphReasoner


def main() -> int:
    required = [
        "README.md", "approach.md", "requirements.txt", "main.py",
        "docs/DECISIONS.md", "docs/PROJECT_DECISION_ROADMAP.md", "docs/SECURITY_REVIEW.md",
        "src/interface/web.py", "scripts/serve.py",
        "data/processed/papers.json", "data/knowledge/knowledge_state.json",
        "data/evaluation/test_cases.json", "tests/test_system.py", "tests/test_web_security.py",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    corpus = json.loads((ROOT / "data/processed/papers.json").read_text(encoding="utf-8"))
    state = json.loads((ROOT / "data/knowledge/knowledge_state.json").read_text(encoding="utf-8"))
    papers = corpus["papers"]
    titles = [paper["title"].strip().lower() for paper in papers]
    assert 50 <= len(papers) <= 100
    assert len(titles) == len(set(titles)), "Duplicate paper titles remain"
    assert all(paper.get("title") and paper.get("openalex_id") and paper.get("source_url") for paper in papers)
    assert all(item.get("evidence") for item in state["relationships"])
    assert state["metadata"]["human_auditable"] is True
    assert state["metadata"]["automatic_kg_library_used"] is False
    web_source = (ROOT / "src/interface/web.py").read_text(encoding="utf-8")
    assert not any(sink in web_source for sink in ("innerHTML", "dangerouslySetInnerHTML", "eval(", "Function("))
    assert not (ROOT / ".env").exists()

    new_idea = "I propose a citation-aware multi-hop RAG method using explicit entity-relation paths and evidence provenance."
    corpus_text = " ".join(f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in papers).lower()
    assert new_idea.lower() not in corpus_text
    report = GraphReasoner(state).reason(new_idea)
    assert report["reasoning_summary"]["graph_used"] is True
    assert report["multi_hop_evidence_paths"]

    secret_pattern = re.compile(r"(?:sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY\s*=\s*[^\s#]+)")
    scanned = []
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            try:
                scanned.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                pass
    assert not any(secret_pattern.search(text) for text in scanned)
    print(json.dumps({
        "required_files": len(required),
        "corpus_papers": len(papers),
        "knowledge_entities": len(state["entities"]),
        "knowledge_relationships": len(state["relationships"]),
        "new_input_graph_used": report["reasoning_summary"]["graph_used"],
        "new_input_path_count": len(report["multi_hop_evidence_paths"]),
        "secrets_detected": False,
        "status": "PASS",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

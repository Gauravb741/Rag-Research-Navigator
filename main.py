#!/usr/bin/env python3
"""CLI for RAG Research Onboarding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.engine import GraphReasoner

STATE_PATH = ROOT / "data" / "knowledge" / "knowledge_state.json"
DEFAULT_IDEA = (
    "I want to build a RAG system for multi-hop question answering where evidence "
    "may be distributed across several documents. Instead of retrieving isolated "
    "chunks, I want to exploit explicit relationships between documents and entities."
)


def load_reasoner() -> GraphReasoner:
    if not STATE_PATH.exists():
        raise FileNotFoundError("Knowledge state is missing. Run: python scripts/fetch_papers.py && python scripts/build_knowledge.py")
    return GraphReasoner(json.loads(STATE_PATH.read_text(encoding="utf-8")))


def print_report(report: dict) -> None:
    print("\n=========================================")
    print("RAG RESEARCH ONBOARDING SYSTEM")
    print("=========================================\n")
    print("1. Research problem detected")
    for item in report["related_problems"][:5]:
        print(f"   - {item['name']}")
    if not report["related_problems"]:
        print("   - No controlled problem match")

    print("\n2. Related concepts")
    for item in report["matched_concepts"][:10]:
        print(f"   - {item['name']} ({item['type']})")
    if not report["matched_concepts"]:
        print("   - None")

    print("\n3. Closest related methods")
    for item in report["related_methods"][:8]:
        print(f"   - {item['name']} — supported by {len(item['supporting_papers'])} paper(s)")
    if not report["related_methods"]:
        print("   - None")

    print("\n4. Prior papers")
    for item in report["related_papers"]:
        print(f"   - {item['title']} ({item.get('year', 'n.d.')}) [graph score {item['graph_score']}]\n     {item['source_url']}")

    print("\n5. Multi-hop evidence paths")
    for path in report["multi_hop_evidence_paths"][:8]:
        hops = " ".join(f"-[{st['relation']}{'(rev)' if st.get('reverse') else ''}]-> {st['to']}" for st in path["steps"])
        print(f"   - ({path['path_type']}) {path['nodes'][0]} {hops}")
        print(f"     evidence: {', '.join(path['evidence_ids'])}")
    if not report["multi_hop_evidence_paths"]:
        print("   - No graph path beyond direct candidate links")

    print("\n6. Related limitations")
    for item in report["known_limitations"][:8]:
        print(f"   - {item['name']}")

    print("\n7. Research directions")
    for item in report["research_directions"][:8]:
        print(f"   - {item['name']}")

    print("\n8. Recommended reading order")
    for item in report["recommended_reading_order"]:
        print(f"   {item['order']}. {item['title']} — {item['reason']}")

    og = report["overlap_and_gap"]
    print(f"\n8b. Overlap and gap analysis  [signal: {og['novelty_signal']}, best weighted overlap {og['best_weighted_overlap']}]")
    print(f"   {og['novelty_note']}")
    for item in og["closest_prior_work"][:3]:
        have = ", ".join(f["name"] for f in item["overlapping_facets"]) or "-"
        miss = ", ".join(f["name"] for f in item["missing_facets"]) or "-"
        print(f"   - {item['title'][:70]} (overlap {item['weighted_overlap']})\n     shares: {have}\n     lacks:  {miss}")
    for combo in og["uncovered_combinations"][:4]:
        print("   - NOT combined by any indexed paper: " + " + ".join(combo["facets"]))
    for name in og["facets_without_corpus_support"]:
        print(f"   - known concept with no paper in the corpus: {name}")
    if report["unmatched_terms"]:
        print("\n8c. Terms outside the ontology: " + ", ".join(u["term"] for u in report["unmatched_terms"][:10]))
    if report["method_lineage"]:
        print("\n8d. Method lineage")
        for link in report["method_lineage"][:6]:
            print(f"   - {link['from']} --{link['relation']}--> {link['to']}  [{link['evidence_id']}]")

    print("\n9. Evidence sources")
    for item in report["evidence"][:10]:
        print(f"   - {item['evidence_id']}: {item['relation']} | {item['quote'][:220]}")

    print("\nUncertainty notes")
    for note in report["uncertainty_notes"]:
        print(f"   - {note}")
    print(f"\nGraph used: {report['reasoning_summary']['graph_used']}; candidates traversed: {report['reasoning_summary']['candidate_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Position a new RAG research idea using an inspectable graph.")
    parser.add_argument("--idea", help="New RAG research idea or short abstract.")
    parser.add_argument("--json", action="store_true", help="Emit the structured report as JSON.")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    try:
        idea = args.idea
        if idea is None:
            print("\nEnter a new RAG research idea (not part of the indexed corpus):\n")
            idea = input("> ").strip() or DEFAULT_IDEA
        report = load_reasoner().reason(idea, limit=max(1, min(args.limit, 20)))
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

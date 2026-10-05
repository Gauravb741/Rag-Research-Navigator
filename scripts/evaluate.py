#!/usr/bin/env python3
"""Evaluate the graph reasoner against a flat TF-IDF baseline.

PRIMARY (automatic, no hand labels, not circular):
  Leave-one-out. For every corpus paper P that has an OpenAlex reference list, feed P's *abstract*
  (not its title) to each system as if it were a new idea, with P excluded from the corpus.
  Ground truth = the other corpus papers that share >= 1 reference with P or are cited by / cite P.
  This is read straight from OpenAlex metadata, never from the graph's derived edges.
  Metrics: precision@5, recall@10, MRR. Systems: baseline, graph, and two ablations.

SECONDARY (diagnostics only):
  * entity coverage on hand-written unseen ideas (the baseline does not output entities, so this
    can only describe the graph system, not compare it)
  * novelty-signal check on ideas whose expected signal is obvious
  * manual labels: if data/evaluation/test_cases.json has relevant_paper_ids filled in
    (see scripts/labeling.py), precision/recall/MRR are reported on those too.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reasoning.engine import GraphReasoner  # noqa: E402
from src.retrieval.lexical import flat_retrieve  # noqa: E402

STATE = ROOT / "data" / "knowledge" / "knowledge_state.json"
PAPERS = ROOT / "data" / "processed" / "papers.json"
CASES = ROOT / "data" / "evaluation" / "test_cases.json"
OUT = ROOT / "data" / "evaluation" / "evaluation_results.json"


def metrics(rankings: list[list[str]], truths: list[set[str]]) -> dict:
    p5 = r10 = mrr = 0.0
    for ranked, truth in zip(rankings, truths):
        hits5 = sum(1 for x in ranked[:5] if x in truth)
        hits10 = sum(1 for x in ranked[:10] if x in truth)
        p5 += hits5 / 5
        r10 += hits10 / len(truth)
        mrr += next((1 / (i + 1) for i, x in enumerate(ranked) if x in truth), 0.0)
    n = max(len(rankings), 1)
    return {"precision@5": round(p5 / n, 4), "recall@10": round(r10 / n, 4), "mrr": round(mrr / n, 4)}


def leave_one_out(reasoner: GraphReasoner) -> dict:
    corpus = json.loads(PAPERS.read_text(encoding="utf-8"))["papers"]
    oa_to_id = {p["openalex_id"]: p["paper_id"] for p in corpus}
    refs = {p["paper_id"]: set(p.get("referenced_work_ids", [])) for p in corpus}
    cites = {p["paper_id"]: {oa_to_id[r] for r in refs[p["paper_id"]] if r in oa_to_id} for p in corpus}
    queries = []
    for p in corpus:
        pid = p["paper_id"]
        if len(p.get("abstract") or "") < 200 or not refs[pid]:
            continue
        truth = set(cites[pid]) | {q for q in cites if pid in cites[q]}
        truth |= {q["paper_id"] for q in corpus if q["paper_id"] != pid and refs[pid] & refs[q["paper_id"]]}
        truth.discard(pid)
        if truth:
            queries.append((pid, p["abstract"], truth))
    systems = {
        "flat_tfidf_baseline": lambda pid, text: [h["paper_id"] for h in flat_retrieve(text, list(reasoner.papers.values()), limit=10, exclude_ids={pid})],
        "graph_full": lambda pid, text: [r["paper_id"] for r in reasoner.reason(text, limit=10, exclude_paper_ids={pid})["related_papers"]],
        "graph_no_citation_expansion": lambda pid, text: [r["paper_id"] for r in reasoner.reason(text, limit=10, exclude_paper_ids={pid}, use_citation_expansion=False)["related_papers"]],
        "graph_no_entity_propagation": lambda pid, text: [r["paper_id"] for r in reasoner.reason(text, limit=10, exclude_paper_ids={pid}, use_propagation=False)["related_papers"]],
    }
    results = {}
    for name, fn in systems.items():
        rankings = [fn(pid, text) for pid, text, _ in queries]
        results[name] = metrics(rankings, [t for _, _, t in queries])
    return {"n_queries": len(queries), "mean_truth_size": round(sum(len(t) for _, _, t in queries) / max(len(queries), 1), 2),
            "systems": results,
            "caveats": ["Ground truth is citation/coupling relatedness, a noisy proxy for topical relevance.",
                        "Only papers with OpenAlex reference lists can be queries (recent papers are excluded).",
                        "Graph nodes derived from the held-out paper's own title (e.g. its method name) still exist; the paper itself is excluded from candidates and from every traversed edge."]}


def case_checks(reasoner: GraphReasoner) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    rows, labeled_rank, labeled_truth, base_rank = [], [], [], []
    for case in cases:
        report = reasoner.reason(case["input"], limit=10)
        matched = {m["entity_id"] for m in report["matched_concepts"]}
        expected = set(case.get("expected_entities", []))
        row = {"id": case["id"], "kind": case.get("kind"), "matched_count": len(matched),
               "entity_coverage": round(len(expected & matched) / len(expected), 3) if expected else None,
               "novelty_signal": report["overlap_and_gap"]["novelty_signal"],
               "expected_signal": case.get("expected_novelty_signal"),
               "signal_ok": (report["overlap_and_gap"]["novelty_signal"] in case["expected_novelty_signal"]) if case.get("expected_novelty_signal") else None,
               "top_paper": report["related_papers"][0]["title"] if report["related_papers"] else None}
        rows.append(row)
        truth = set(case.get("relevant_paper_ids") or [])
        if truth:
            labeled_rank.append([r["paper_id"] for r in report["related_papers"]])
            base_rank.append([h["paper_id"] for h in flat_retrieve(case["input"], list(reasoner.papers.values()), limit=10)])
            labeled_truth.append(truth)
    cov = [r["entity_coverage"] for r in rows if r["entity_coverage"] is not None]
    sig = [r["signal_ok"] for r in rows if r["signal_ok"] is not None]
    out = {"cases": rows,
           "mean_entity_coverage": round(sum(cov) / len(cov), 3) if cov else None,
           "novelty_signal_accuracy": f"{sum(sig)}/{len(sig)}" if sig else None,
           "note": "Secondary diagnostics only: expected entities are my own labels from the same ontology and the flat baseline cannot output entities."}
    if labeled_truth:
        out["manual_labels"] = {"n_cases": len(labeled_truth), "graph": metrics(labeled_rank, labeled_truth), "baseline": metrics(base_rank, labeled_truth)}
    else:
        out["manual_labels"] = "No relevant_paper_ids filled in yet. Run scripts/labeling.py sheet, label, then scripts/labeling.py apply."
    return out


def main() -> int:
    reasoner = GraphReasoner(json.loads(STATE.read_text(encoding="utf-8")))
    result = {"primary_leave_one_out": leave_one_out(reasoner), "secondary_case_checks": case_checks(reasoner)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    p = result["primary_leave_one_out"]
    print(f"Leave-one-out queries: {p['n_queries']} (mean relevant papers per query: {p['mean_truth_size']})")
    for name, m in p["systems"].items():
        print(f"  {name:32s} P@5={m['precision@5']:.3f}  R@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}")
    s = result["secondary_case_checks"]
    print(f"Case checks: mean entity coverage={s['mean_entity_coverage']}, novelty-signal accuracy={s['novelty_signal_accuracy']}")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

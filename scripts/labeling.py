#!/usr/bin/env python3
"""Hand-labelling helper for the manual evaluation.

  python scripts/labeling.py sheet   -> data/evaluation/labeling_sheet.csv
      Pools the top-10 papers of BOTH systems for every test idea, shuffled and without saying
      which system returned what (standard pooled-relevance practice), with a blank 'relevant' column.
  python scripts/labeling.py apply   -> writes the 1-marked papers into test_cases.json
      Mark relevant papers with 1 (leave others blank or 0), then apply.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.reasoning.engine import GraphReasoner  # noqa: E402
from src.retrieval.lexical import flat_retrieve  # noqa: E402

CASES = ROOT / "data" / "evaluation" / "test_cases.json"
SHEET = ROOT / "data" / "evaluation" / "labeling_sheet.csv"


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    data = json.loads(CASES.read_text(encoding="utf-8"))
    if mode == "sheet":
        reasoner = GraphReasoner(json.loads((ROOT / "data" / "knowledge" / "knowledge_state.json").read_text(encoding="utf-8")))
        rng = random.Random(11)
        rows = []
        for case in data["cases"]:
            pool = {r["paper_id"] for r in reasoner.reason(case["input"], limit=10)["related_papers"]}
            pool |= {h["paper_id"] for h in flat_retrieve(case["input"], list(reasoner.papers.values()), limit=10)}
            ids = sorted(pool)
            rng.shuffle(ids)
            for pid in ids:
                p = reasoner.papers[pid]
                rows.append({"case_id": case["id"], "idea": case["input"], "paper_id": pid, "title": p["title"],
                             "abstract_start": (p.get("abstract") or "")[:300], "relevant": ""})
        with SHEET.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"Wrote {SHEET} with {len(rows)} rows. Put 1 in 'relevant' for papers that genuinely bear on the idea.")
    elif mode == "apply":
        labels: dict[str, list[str]] = {}
        with SHEET.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["relevant"].strip() == "1":
                    labels.setdefault(row["case_id"], []).append(row["paper_id"])
        for case in data["cases"]:
            case["relevant_paper_ids"] = sorted(labels.get(case["id"], []))
        CASES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Applied labels for", sum(1 for c in data["cases"] if c["relevant_paper_ids"]), "ideas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

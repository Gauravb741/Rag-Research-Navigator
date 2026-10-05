# RAG Research Navigator

Position a **new research idea** inside the literature on structured / graph-based Retrieval-Augmented Generation (RAG). The system turns 72 real papers into a hand-modelled knowledge graph, reasons over it when you describe an idea, and returns a structured report: closest prior work, what overlaps and what nobody has combined, known limitations and directions, method lineage, and a reading order.

Domain B (Research Paper Onboarding), Calyb AI engineering assignment. The design reasoning is in **[approach.md](approach.md)**; the inspectable knowledge state is **`data/knowledge/knowledge_state.json`**.

## What it does

1. **Knowledge model** — 590 entities (Paper, Author, Method, ResearchProblem, Dataset, Metric, Architecture, Limitation, ResearchDirection, ApplicationDomain, Concept) and 1,203 relationships. 131 of them link two non-paper entities (e.g. `Method -IMPROVES_ON-> Method`, `Limitation -MOTIVATES-> ResearchDirection`), so reasoning is more than "paper has tag".
2. **Provenance** — every edge stores the rule that produced it, a `high/medium/low` confidence, a quote and where the quote came from.
3. **Reasoning on new input** — weighted facet matching (exact alias, stem, hand-written synonym, description overlap), propagation over entity-to-entity edges, citation expansion, typed multi-hop chains, overlap-and-gap analysis, topologically ordered reading list.
4. **No automatic extraction tools.** Vocabulary, cue lists and rules are in `src/modeling/ontology.py` and `scripts/build_knowledge.py`; matching is regular expressions and a tiny suffix stemmer (`src/modeling/matching.py`). No NLP/KG library and no model call at build or query time.

## Structure

```
src/modeling/ontology.py    vocabulary, cue lists, schema, synonyms, curated priors (hand-written)
src/modeling/matching.py    alias regex, stemmer, sentence splitter shared by builder and reasoner
scripts/fetch_papers.py     OpenAlex fetch + deterministic corpus selection (cached under data/raw)
scripts/build_knowledge.py  corpus -> knowledge state (passes A-G, see approach.md)
src/reasoning/engine.py     GraphReasoner: the new-input pipeline; all weights are constants at the top
src/retrieval/lexical.py    flat TF-IDF baseline (used for comparison and as a small blended signal)
scripts/evaluate.py         leave-one-out evaluation + diagnostics
scripts/labeling.py         build / apply a hand-labelling sheet for manual evaluation
scripts/edge_quality_report.py  review sheet of sampled edges for hand-checking
main.py                     CLI        scripts/serve.py + src/interface/web.py   local web atlas
data/knowledge/             knowledge_state.json (+ entities.json, relationships.json, build_summary.json)
data/evaluation/            test_cases.json, evaluation_results.json
docs/                       corpus selection, decisions log, edge review sheet, v2 change notes
```

## Install

Python 3.10+. The runtime needs only the standard library; `pytest` is for tests.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Environment (optional): `OPENALEX_MAILTO` — an email for the OpenAlex polite pool, used only by `fetch_papers.py`. It is read from the process environment (a `.env` file is not auto-loaded). See `.env.example`.

## Load or regenerate the knowledge state

The committed state already exists. To rebuild it from the cached corpus (offline, deterministic, byte-identical on every run):

```bash
python scripts/build_knowledge.py
```

To re-download the corpus first (needs network): `python scripts/fetch_papers.py --limit 72 --refresh`, then rebuild.

## Run it on a new input

```bash
python main.py --idea "A RAG system that follows citation graphs to answer multi-hop research questions, with provenance, and updates incrementally as papers arrive."
python main.py --json --idea "..."     # full structured report as JSON
python main.py                         # interactive prompt
python scripts/serve.py --port 8000    # web atlas at http://localhost:8000 (binds 127.0.0.1 by default; pass --host to expose)
```

The report includes: `matched_concepts` (with match type and confidence), `unmatched_terms`, `related_papers` (with facet overlap), `method_lineage`, `multi_hop_evidence_paths` (typed hops with evidence IDs), `known_limitations`, `research_directions`, **`overlap_and_gap`** (closest prior work, covered and uncovered facet combinations, a cautious novelty signal), `recommended_reading_order` (with reasons), `external_foundations`, `evidence` (quote, rule, confidence per ID), `uncertainty_notes`, and the `flat_retrieval_baseline` for comparison.

## Evaluate and test

```bash
python scripts/evaluate.py      # writes data/evaluation/evaluation_results.json
python -m pytest                # 24 tests
python scripts/edge_quality_report.py   # docs/EDGE_QUALITY_REPORT.md: sample edges for you to mark correct/incorrect
python scripts/labeling.py sheet        # then label by hand, then: python scripts/labeling.py apply
```

Honest headline from the leave-one-out evaluation (33 queries): the flat TF-IDF baseline is **slightly better at ranking papers** (P@5 0.248 vs 0.212) than the graph reasoner. The graph's value is the structured output (typed chains, overlap/gap, reading order) that retrieval cannot give. Details and caveats are in `approach.md`.

## Known limitations

- 32 of 72 papers have no OpenAlex reference list (recent work), so citation evidence is sparse (23 `CITES`, 57 `SHARES_REFERENCES`). Textual method mentions partly compensate.
- Edges come from titles and abstracts only; abstracts rarely state limitations, so only 20 `MOTIVATED_BY_LIMITATION` edges exist.
- Wording outside the ontology is only caught by stems, a small synonym table and description overlap; `unmatched_terms` shows what was missed.
- Edge precision has been spot-checked by sampling, not measured; fill in `docs/EDGE_QUALITY_REPORT.md` to measure it.

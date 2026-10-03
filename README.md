# RAG Research Onboarding System

A reproducible, inspectable research-paper onboarding prototype for **structured / graph-based Retrieval-Augmented Generation (RAG)**.

The system accepts a **new RAG research idea or short abstract that is not part of the indexed corpus** and positions it in the prior-work landscape using explicit entities, relationships, multi-hop paths, and source evidence.

It does **not** claim to prove absolute novelty. It reports overlap and connections found in the indexed corpus using cautious language such as “related prior work,” “similar methodological direction,” and “no close match was found within the indexed corpus.”

## Problem and objective

Flat retrieval can find semantically similar papers, but it does not by itself explain how a new idea connects to:

- existing RAG methods
- research problems
- datasets and metrics
- limitations
- architectures
- citations and foundational work
- possible research directions

This project represents those connections explicitly and traverses them when a new idea is provided.

## Selected scope and use case

- **Domain:** Research Paper Onboarding
- **Topic:** Retrieval-Augmented Generation
- **Research scope:** Structured / graph-based RAG for multi-hop research synthesis
- **Use case:** Position a new RAG research idea in the prior-work landscape

The system's core flow is:

```text
72 real research papers
        ↓
controlled-vocabulary entity and relationship mapping
        ↓
inspectable JSON knowledge state
        ↓
new unseen RAG idea
        ↓
seed matching + graph traversal + citation/path expansion
        ↓
evidence-backed positioning report
```

## What is included

- A focused corpus of **72 papers** from OpenAlex Works API searches from 2020–2026.
- Cached API responses and a query manifest.
- A small hand-authored ontology in `src/modeling/ontology.py`.
- `knowledge_state.json` with **537 entities** and **914 relationships**.
- Evidence/provenance attached to derived relationships.
- A graph-aware reasoner that is distinct from flat paper retrieval.
- A flat lexical retrieval baseline for comparison.
- A CLI and structured JSON output.
- A dependency-free same-origin web atlas with the graph as its visual centerpiece.
- Four manually specified unseen research ideas for evaluation.
- Unit tests for ingestion artifacts, provenance, graph traversal, evidence, unrelated inputs, and the baseline.

## Knowledge model

### Entity types

- `Paper`
- `Author`
- `Method`
- `ResearchProblem`
- `Dataset`
- `Metric`
- `Architecture`
- `Limitation`
- `ResearchDirection`
- `Concept`

### Relationship types

- `AUTHORED_BY`
- `CITES`
- `BUILDS_ON`
- `INTRODUCES_METHOD`
- `USES_METHOD`
- `ADDRESSES_PROBLEM`
- `EVALUATES_ON`
- `USES_METRIC`
- `IMPLEMENTS_ARCHITECTURE`
- `HAS_LIMITATION`
- `SUGGESTS_DIRECTION`
- `RELATES_TO_CONCEPT`

Every relationship stores its source paper where applicable, a quote or metadata explanation, source location, evidence type, and relationship identifier.

## Architecture

```text
OpenAlex Works API
        ↓
scripts/fetch_papers.py
        ↓
data/raw/openalex_cache + data/processed/papers.json
        ↓
scripts/build_knowledge.py
        ↓
data/knowledge/knowledge_state.json
        ↓
main.py / src.reasoning.engine.GraphReasoner
        ↓
structured report + readable CLI + same-origin web atlas
```

The graph is stored as JSON so it can be inspected without running the application. A vector database is not used as the central representation. No automatic knowledge-graph-construction library is used.

## Requirements

- Python 3.10+
- Internet access only when reproducing the OpenAlex corpus
- No API key is required
- `pytest` for tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

Optional polite-pool configuration:

```bash
cp .env.example .env
# Set OPENALEX_MAILTO to an email address if desired.
```

The scripts read `OPENALEX_MAILTO` from the process environment; a `.env` file is not automatically loaded.

## Reproduce the corpus and knowledge state

```bash
python scripts/fetch_papers.py --limit 72
python scripts/build_knowledge.py
```

To refresh the public API cache:

```bash
python scripts/fetch_papers.py --limit 72 --refresh
python scripts/build_knowledge.py
```

The corpus procedure is documented in [`docs/CORPUS_SELECTION.md`](docs/CORPUS_SELECTION.md). The decision history is in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Run the CLI

Interactive mode:

```bash
python main.py
```

## Run the web UI

The project also includes a secure, same-origin local web interface. It uses the existing graph reasoner and does not expose API keys or external services to the browser.

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` locally, or use the live preview supplied by the development environment. The atlas loads the actual serialized knowledge state through `GET /api/graph`, shows a subdued overview of the 537 entities and 914 relationships, and lets a researcher search, filter, zoom, pan, drag, inspect, and follow evidence-bearing connections. Entering an idea calls the existing `POST /api/reason` route; the returned report synchronizes with graph modes, real multi-hop evidence paths, paper details, relationship evidence, and the corpus timeline. The API accepts only a bounded JSON body containing an `idea` string. The frontend renders external paper metadata with safe DOM text nodes rather than arbitrary HTML.

Security review and hardening details are documented in [`docs/SECURITY_REVIEW.md`](docs/SECURITY_REVIEW.md).

Scripted input:

```bash
python main.py --idea "I want to build an explainable RAG system for research question answering that follows citation networks and entity-relation paths across papers instead of retrieving isolated chunks."
```

Structured JSON output:

```bash
python main.py \
  --idea "I want to evaluate GraphRAG on multi-hop questions with inter-context conflicts, measuring faithfulness and context relevance." \
  --json
```

The report includes:

- matched concepts
- related methods and architectures
- related problems, datasets, and papers
- citation/foundation connections
- multi-hop evidence paths
- known limitations
- research directions
- recommended reading order
- evidence IDs and source quotes
- uncertainty notes
- flat retrieval baseline results

## Example reasoning behavior

For an unseen idea about citation-aware multi-hop synthesis, the system can produce paths such as:

```text
Citation network
  → CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs
  → Citation-aware research synthesis
```

or:

```text
Multi-hop reasoning
  → GraphRAG paper
  → Limitation / research direction
```

The actual paths and evidence are generated from `data/knowledge/knowledge_state.json`; the CLI does not fabricate citations.

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

This evaluates four manually specified ideas that are not papers in the indexed corpus. It reports:

- expected-entity coverage for the graph reasoner
- expected-entity coverage for a flat retrieval baseline
- number of multi-hop paths
- whether graph reasoning was used
- evidence count and provenance completeness

These are project diagnostics over a small manually specified set, not a universal benchmark or a claim of research-level performance.

Run the tests:

```bash
pytest
```

## Repository structure

```text
.
├── README.md
├── approach.md
├── requirements.txt
├── pytest.ini
├── main.py
├── docs/
│   ├── DECISIONS.md
│   ├── CORPUS_SELECTION.md
│   ├── PROJECT_DECISION_ROADMAP.md
│   ├── RESEARCH_SCOPE_REVIEW.md
│   └── SECURITY_REVIEW.md
├── data/
│   ├── raw/
│   │   ├── openalex_cache/
│   │   └── query_manifest.json
│   ├── processed/
│   │   └── papers.json
│   ├── knowledge/
│   │   ├── knowledge_state.json
│   │   ├── entities.json
│   │   ├── relationships.json
│   │   └── build_summary.json
│   └── evaluation/
│       ├── test_cases.json
│       └── evaluation_results.json
├── scripts/
│   ├── fetch_papers.py
│   ├── build_knowledge.py
│   ├── evaluate.py
│   ├── validate_project.py
│   └── serve.py
├── src/
│   ├── modeling/ontology.py
│   ├── reasoning/engine.py
│   └── retrieval/lexical.py
└── tests/test_system.py
```

## Limitations

- The ontology is intentionally small and hand-authored; terminology outside its aliases may be missed.
- Relationship mapping is conservative lexical mapping over title/abstract text plus OpenAlex metadata. It is not a complete scholarly ontology.
- OpenAlex citation edges are retained only when both papers are in the selected 72-paper corpus.
- The corpus includes broad and domain-specific GraphRAG studies because the selected scope covers methods, benchmarks, and applications; domain transfer should not be assumed.
- The system reports corpus-relative overlap and does not establish absolute novelty.
- The web atlas is a local, dependency-free research interface rather than a multi-user production application.
- Evaluation uses four manually specified ideas and requires a larger expert-annotated benchmark for stronger conclusions.

## Future work

- Add expert-reviewed relation annotations and inter-annotator agreement.
- Add citation-aware academic sources beyond OpenAlex.
- Expand the ontology while preserving provenance and human review.
- Add temporal reasoning for limitation-to-direction chains.
- Extend the atlas with expert-reviewed relation annotations and richer temporal path views.
- Compare graph traversal against stronger dense and hybrid retrieval baselines.

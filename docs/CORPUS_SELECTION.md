# Corpus Selection and Reproducibility

## Scope

The corpus is for **structured / graph-based RAG for multi-hop research synthesis**, with emphasis on methods and benchmarks that represent or retrieve entities, relations, subgraphs, citation networks, graph neighborhoods, or multi-hop evidence.

The approved use case is **positioning a new RAG research idea in the prior-work landscape**.

## Source

- API: [OpenAlex Works API](https://api.openalex.org/works)
- Metadata fields: title, authors, year, publication date, abstract, DOI, OpenAlex ID, landing/source URL, citation count, reference count, referenced works, open-access flag, and publication type.
- Retrieval date: 2026-10-02
- Date filter: 2020-01-01 through 2026-10-02
- No API key is required.
- The `semantic_scholar_id` field is retained in normalized metadata for future enrichment; OpenAlex does not provide it for the selected records.

## Search queries

The fetch script uses these focused free-text queries:

1. `graph retrieval augmented generation`
2. `knowledge graph retrieval augmented generation`
3. `multi-hop retrieval augmented generation`
4. `subgraph retrieval augmented generation`
5. `citation graph retrieval language model`
6. `document graph retrieval augmented generation`
7. `textual graph question answering retrieval`
8. `research paper graph retrieval generation`
9. `GraphRAG RAG knowledge graph`
10. `multi-hop RAG benchmark`
11. `research question answering citation graph`
12. `academic paper retrieval graph`
13. `paper citation recommendation graph neural language model`
14. `graph RAG query-focused summarization`

A second set of close-title searches protects the inclusion of foundational baselines and directly relevant academic/citation-graph studies, such as the original RAG paper, DPR, RAPTOR, G-Retriever, GRAG, KG²RAG, HippoRAG, MultiHop-RAG, CG-RAG, and GraphRAG-Bench.

The complete manifest is in `data/raw/query_manifest.json`; raw API responses are cached in `data/raw/openalex_cache/`.

## Inclusion criteria

A paper is eligible when its title or abstract contains RAG language and at least one selected-scope signal:

- graph RAG or knowledge-graph RAG
- subgraph or textual-graph retrieval
- multi-hop retrieval or reasoning
- citation-aware or research-question answering
- graph-enhanced document retrieval
- explicit entities, relations, graph paths, or structured evidence
- a foundational RAG/Retrieval baseline needed to explain the selected subset

The deterministic scoring script gives greater weight to focused title terms, then to focused abstract terms. Curated close-title matches receive an explicit reproducibility boost.

## Exclusion criteria

- Non-RAG graph papers without a retrieval-augmented generation or equivalent language-model retrieval connection.
- Papers outside the 2020–2026 date window.
- Duplicate OpenAlex records for the same normalized title.
- Records without a usable title.
- Generic LLM application papers whose title and abstract do not contain a structural, graph, multi-hop, citation, or evidence-retrieval connection.
- Unrelated uses of the acronym “RAG,” such as biological RNA-as-graphs records.

Domain-specific applications are retained only when their abstract describes a graph-RAG method, a graph-retrieval problem, a multi-hop benchmark, or a structural evaluation contribution. They are not treated as universally generalizable evidence.

## Deduplication

Titles are lowercased, punctuation is removed, and version markers are stripped. If more than one OpenAlex record has the same normalized title, the record with an abstract is preferred; citation count breaks remaining ties.

## Final corpus

- Unique API candidates before title deduplication: 405
- Eligible candidates after focus scoring: 192
- Selected papers: **72**
- Selection limit: 72
- Paper metadata: `data/processed/papers.json`

The final graph contains:

- 72 Paper entities
- 389 Author entities
- 19 Method entities
- 8 ResearchProblem entities
- 13 Dataset entities
- 9 Metric entities
- 5 Architecture entities
- 8 Limitation entities
- 6 ResearchDirection entities
- 8 Concept entities
- 914 relationships with evidence/provenance fields

## Reproduction

```bash
python scripts/fetch_papers.py --limit 72
python scripts/build_knowledge.py
```

Use `--refresh` to ignore cached search responses. The public API may evolve, so a future refresh can produce a different candidate ranking; the checked-in normalized corpus and raw caches preserve the run used for the current demonstration.

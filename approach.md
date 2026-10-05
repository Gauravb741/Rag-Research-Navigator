# Technical Approach

> **Author's note on authorship.** AI coding assistants were used for implementation. The sections below describe the design as built, including why each choice was made. The parts of the v2 rebuild (rules, aliases, relation types, weights) were drafted with an AI assistant and **must be read, understood and, where you disagree, changed by the author** before submission. `docs/V2_CHANGES.md` lists every decision that needs that review. Edit this paragraph so it states exactly what was and was not AI-assisted.

## 1. Subset of data and why

**Topic:** structured / graph-based RAG (GraphRAG, knowledge-graph RAG, multi-hop RAG over graphs) and its direct foundations (dense retrieval, RAPTOR, GNN-based retrievers).
**Size:** 72 papers from OpenAlex, selected deterministically from a 405-paper candidate pool (`data/raw/all_candidates.json`; selection rules in `docs/CORPUS_SELECTION.md`).
**Why this slice:** the field is young and fast moving, so a newcomer genuinely lacks a map; it is narrow enough that relationships between papers matter more than coverage; and I can judge the output myself because I know the topic.

**Data problems that shaped the design** (measured, not assumed):
- 32 of 72 papers have *zero* OpenAlex references (2025–26 papers not yet indexed). Citation edges alone cannot connect them.
- 5 papers have no usable abstract.
- 53 of 72 papers are from 2025–26, so "who cites whom" is thin; text-level evidence had to carry the model.

## 2. Entities and relationships, and why

**Use case:** a researcher with a new idea wants to know *what is closest, what overlaps, what nobody has combined, what to read first*. That is the third example in the assignment (is my contribution novel, what prior work is closest) and it needs structure, because "novel" means a *combination* of facets that is absent, which keyword search cannot see.

**Entity types (11):** Paper, Author, Method (`kind=paradigm` such as GraphRAG, or `kind=named` such as HyperGraphRAG), ResearchProblem, Dataset, Metric, Architecture, Limitation, ResearchDirection, ApplicationDomain, Concept. Facets (problem, method, architecture, dataset, domain) are separate entities because overlap is computed per facet.

**Two kinds of relationship:**
- *Paper → entity* (what a paper does): `INTRODUCES_METHOD`, `USES_METHOD`, `EVALUATES_METHOD`, `MENTIONS_METHOD`, `INTRODUCES_DATASET`, `ADDRESSES_PROBLEM`, `EVALUATES_ON`, `USES_METRIC`, `IMPLEMENTS_ARCHITECTURE`, `MOTIVATED_BY_LIMITATION`, `PURSUES_DIRECTION`, `APPLIED_IN_DOMAIN`, `RELATES_TO_CONCEPT`; paper → paper: `CITES`, `SHARES_REFERENCES`, `BUILDS_ON`, `COMPARES_AGAINST`.
- *Entity → entity* (what the field knows, derived): `Method -IMPROVES_ON / BUILDS_ON / COMPARES_AGAINST -> Method`, `Method -INSTANCE_OF-> paradigm`, `Method -TARGETS_PROBLEM-> Problem`, `Method -MOTIVATED_BY-> Limitation`, `Method -USES_ARCHITECTURE-> Architecture`, `Method -EVALUATED_ON-> Dataset`, `Limitation -MOTIVATES-> ResearchDirection`. These 131 edges are what let the reasoner chain facts across papers instead of only tagging papers.

**Deliberate modelling choices**
- *Paradigm vs named method.* In the first version every paper with "Retrieval-Augmented Generation" in its title "introduced" standard RAG (35 papers). A method can now only be *introduced* if it is named: via a `Name:` title prefix or a title that begins with a known method alias. Paradigms get `USES_METHOD` only, and are down-weighted in scoring.
- *Limitations are limitations of **prior** approaches.* Abstracts rarely state a paper's own limitations; they state what existing work gets wrong. The relation is therefore `MOTIVATED_BY_LIMITATION`, and it requires a limitation cue in the same sentence as the alias, no result/improvement claim in that sentence, and no negation ("negligible overhead"). This cut a vocabulary-only edge set (which produced e.g. "scalability" from "large-scale") down to 20 edges I can defend.
- *Directions are what a paper pursues.* `PURSUES_DIRECTION`, not "suggests"; future-work sentences get a `qualifier`.
- *Confidence is explicit.* `high` (title match or a cue sentence), `medium`, `low`; the scorer weights 1.0 / 0.7 / 0.4.

## 3. How the knowledge representation was built, and the tradeoffs

`scripts/build_knowledge.py` runs seven passes, every edge tagged with a named rule (all rules are listed in `knowledge_state.json → rules`):
A. discover method/dataset names from `Name:` title prefixes (18 methods found beyond the 21 seeded in the ontology); B. paper→entity edges from alias and cue rules; C. method dependencies from sentences that name another method with an improve/extend/compare cue; D. aggregate the introducing paper's facts onto the method; E. limitation→direction edges from co-occurrence in one paper plus 8 curated priors with rationales; F. citation metadata (`CITES`, `SHARES_REFERENCES` for ≥2 shared references, external works cited by ≥3 corpus papers); G. foundation scores.

| Tradeoff | Choice | Cost |
|---|---|---|
| Extraction | Hand-written aliases/cues over title+abstract | Misses unusual wording; precision only spot-checked. Required by the assignment, and every edge is auditable |
| Storage | One JSON file (+ entities/relationships/build_summary) | No graph DB or query language; fine for ~1.2k edges, readable without code |
| Evidence | Abstract only, no full text | Limitations/future work are under-captured |
| Stemming | A ~15-line suffix stripper | Crude, but readable and deterministic |
| Determinism | No timestamps, stable ordering | Verified by a test that rebuilds and compares hashes |

## 4. What happens when a new input arrives

`GraphReasoner.reason(idea)` (all weights are constants at the top of `src/reasoning/engine.py`):
1. **Match** the text to ontology entities: exact alias (1.0) → stem (0.85) → hand-written synonym (0.7) → description overlap (0.4). A lone weak match never switches the graph on. Words not covered are returned as `unmatched_terms`.
2. **Weight** each matched facet by IDF over the corpus × type weight × match confidence (paradigms ×0.4; known concepts with no supporting paper are flagged, not dropped).
3. **Score papers** by weighted facet overlap; add propagation through entity→entity edges (a matched problem reaches the papers that introduced methods targeting it); add citation/build-on/coupling expansion from the top papers; add a small lexical signal.
4. **Chain**: BFS over entity→entity edges (≤3 hops) from matched entities, ending at a limitation, direction or method. Direction, problem, architecture, dataset and paradigm nodes are terminals so a path cannot hop through a hub to an unrelated sibling.
5. **Overlap and gap**: for the idea's top supported facets, enumerate 2–4-facet combinations (≥2 facet types). *Covered* = at least one paper has all of them; *uncovered minimal* = no paper has them though every smaller sub-combination is covered. These are candidate gaps. `novelty_signal` ∈ {close_match_found ≥0.75 weighted overlap, partial_overlap ≥0.40, no_close_match_in_indexed_corpus, insufficient_ontology_coverage}, worded as "not proof of novelty".
6. **Reading order**: candidate papers + their in-corpus prerequisites + up to 2 overlapping surveys, ordered topologically on `CITES`/`BUILDS_ON`, with surveys and high foundation-score papers first, then by year. Each entry states why.

Output is JSON with evidence IDs that resolve to a quote, rule and confidence in the knowledge state.

## 5. Evaluation (and what it does not show)

**Primary — leave-one-out, no hand labels.** For each of 33 papers with a reference list, its abstract is the "new idea" and the paper itself is excluded; relevant = other corpus papers it cites, that cite it, or that share ≥1 reference (read from OpenAlex metadata, not from the graph).

| System | P@5 | R@10 | MRR |
|---|---|---|---|
| Flat TF-IDF baseline | 0.248 | 0.238 | 0.449 |
| Graph reasoner (full) | 0.212 | 0.179 | 0.404 |
| − citation expansion | 0.182 | 0.147 | 0.369 |
| − entity propagation | 0.164 | 0.164 | 0.346 |

Reading: **the baseline ranks papers a little better than the graph.** The ablations show the structure helps (propagation +0.05 P@5, citation expansion +0.03) but does not beat lexical similarity for plain retrieval. A sensitivity run of the lexical blend weight (2→25) brought the graph to roughly parity, never clearly above; I did not tune the shipped weight on this set. Caveats: citation relatedness is a noisy proxy; only papers with reference lists can be queries; nodes derived from the held-out paper's own title still exist (the paper is excluded from candidates and from every traversed edge).

**What the graph is for** is output retrieval cannot produce: typed chains, overlap-and-gap, method lineage, reading order. Those are checked by tests (24 tests: provenance on every edge, no paradigm "introductions", typed chains, near-duplicate not called novel, off-topic input rejected, exclusion works, build is deterministic) and 13 hand-written unseen ideas (entity coverage 0.80; novelty signal correct on the 2 cases with an obvious answer). Weak spot found: a paraphrase of the citation idea scored `insufficient_ontology_coverage`, showing the synonym fallback is thin.

**Not yet measured:** edge precision. `docs/EDGE_QUALITY_REPORT.md` has 315 sampled edges with a blank *correct?* column to fill in, and `scripts/labeling.py` builds a pooled sheet for relevance labels.

## 6. What I would build next, and why

1. **Measure and fix edge precision** (fill the review sheet): the rules are the product, and marked-wrong edges point at the exact alias to tighten.
2. **Full text for limitations/future work** (arXiv): the biggest information loss is abstract-only evidence.
3. **Resolve external foundations**: 27 works are cited by ≥3 corpus papers but only 4 have titles offline; resolving them would show the field's true foundations.
4. **Learned or embedding fallback for unfamiliar wording**, kept *outside* the graph builder so the schema stays hand-defined.
5. **A labelled retrieval set** (the labelling script exists) to replace the citation proxy.

## 7. What was intentionally not built

A graph database, an LLM-written summary layer, automatic entity/relation extraction, and any novelty claim stronger than "overlap within the indexed corpus".

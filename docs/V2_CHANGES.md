# Version 2 changes and decisions that need the author's review

**Why:** an audit of v1 (measured against the code) found: 0 entity→entity edges; 85 `INTRODUCES_METHOD` edges, 62 of them pointing at "Standard RAG" or "GraphRAG"; `HAS_LIMITATION` edges created from words like "scale" and "domain-specific tasks"; only 23 `CITES` edges; `matched_alias` computed but never stored; 8 vocabulary terms that never matched; an evaluation whose expected answers came from the system's own ontology.

## What changed
| Area | v1 | v2 |
|---|---|---|
| Entities / relationships | 537 / 914 | 590 / 1,203 (131 entity→entity) |
| Edge metadata | quote only | quote, location, rule, confidence, matched alias |
| Method introduction | alias in title ⇒ introduces | needs `Name:` prefix, or title starting with a named-method alias; paradigms never introduced |
| Limitations | any alias match | alias + limitation cue in same sentence, no result/negation words (20 edges) |
| Citations | `CITES` + loose `BUILDS_ON` | + `SHARES_REFERENCES`, textual `BUILDS_ON`/`COMPARES_AGAINST`, foundation scores, external foundations |
| Reasoner | lexical seed + paper expansion | weighted facets, 4 match types, propagation, typed chains, overlap-and-gap, explained reading order |
| Evaluation | 4 ideas, entity coverage vs a baseline that cannot output entities | leave-one-out retrieval with ablations + 13 ideas + labelling tools |
| Tests | 13 | 24 |
| New entity type | – | ApplicationDomain (also registered in the web UI) |
| `serve.py` default host | 0.0.0.0 | 127.0.0.1 |

## Decisions you must be able to explain (all were proposed by the assistant)
1. Relation set and the paradigm/named split (`ontology.py`, `RELATION_TYPES`).
2. Every alias and cue list; especially the limitation cues and `PRIOR_MARKER` in `build_knowledge.py`.
3. The `Name:` title-prefix rule for discovering methods (≤2 tokens, internal capital or digit).
4. The 8 curated limitation→direction priors and their rationales (`CURATED_LIMITATION_DIRECTIONS`).
5. Scoring weights, thresholds (0.75 / 0.40 / 0.25), `SHARED_REFERENCE_THRESHOLD = 2`.
6. The evaluation protocol (citation-relatedness proxy) and the decision *not* to tune weights on it.
7. Expected entities / signals in `test_cases.json` are the assistant's judgement; re-check them.

## Kept as-is on purpose
`docs/DECISIONS.md`, `docs/PROJECT_DECISION_ROADMAP.md` and the v1 corpus selection record how the project was actually built. They were not edited or removed.

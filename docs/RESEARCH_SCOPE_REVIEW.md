# Preliminary RAG Research-Scope Review

**Date:** 2026-10-02  
**Stage:** Before Decision #1  
**Purpose:** Compare plausible focused research subsets before selecting the project scope.

## Literature anchor

The original RAG paper frames the problem as combining a parametric language model with non-parametric external memory. Its experiments use a neural retriever over a dense Wikipedia index and compare sequence-level and token-level retrieval conditioning. The paper motivates RAG with knowledge access, provenance, and updateability problems.

Representative follow-on work makes the main design space concrete:

- **Retrieval:** DPR demonstrates learned dense passage retrieval against BM25-style sparse retrieval; HyDE uses a generated hypothetical document as a zero-shot dense-retrieval query.
- **Adaptive / multi-stage RAG:** FLARE retrieves during generation when predicted content is low-confidence; Self-RAG trains a model to decide when to retrieve and to critique retrieved passages and its own generations.
- **Long-context / document-intensive RAG:** LongRAG changes the retrieval unit from short passages to roughly 4K-token units; RAPTOR builds a recursive tree of clustered summaries and retrieves at multiple abstraction levels.
- **Structured / graph RAG:** GraphRAG-style work adds entities, relations, communities, and local/global retrieval to support multi-hop or corpus-level synthesis.
- **Evaluation:** FRAMES evaluates factuality, retrieval, and reasoning together; ARES and RagChecker emphasize component-level and fine-grained diagnosis rather than a single end-to-end score.

## Candidate scopes

### A. Evolution of RAG architectures

**Question shape:** How did RAG move from a fixed retrieve-then-generate pipeline toward advanced and modular systems?

**Useful corpus:** Foundational RAG; survey taxonomies; DPR/FiD-style baselines; HyDE; FLARE; Self-RAG; RAPTOR/GraphRAG as later branches.

**Strengths**

- Gives the clearest historical narrative for a research-paper onboarding system.
- Naturally supports relations such as `EXTENDS`, `INTRODUCES`, `USES_RETRIEVER`, and `ADDRESSES_LIMITATION`.
- Makes it easy to explain where a new paper fits in the evolution.

**Risks**

- Broad and potentially descriptive rather than experimentally sharp.
- A small corpus may oversimplify the field; a large corpus may exceed a manageable project scope.
- Evaluation must test the quality of a synthesis/positioning output, not only retrieval.

**Best fit:** A system that helps a researcher understand the lineage and position of a new RAG paper or idea.

### B. Retrieval techniques in RAG

**Question shape:** Which retrieval choices—sparse, dense, hybrid, query expansion, reranking, or metadata-aware retrieval—are useful under which conditions?

**Useful corpus:** BM25/DPR; ColBERT-style late interaction; HyDE; multi-query/decomposition; reranking and hybrid-retrieval studies.

**Strengths**

- Technically focused and experimentally measurable with Recall@k, MRR, nDCG, latency, and cost.
- A manageable paper set can be selected around one retrieval bottleneck.
- The knowledge model can connect techniques to assumptions, datasets, failure modes, and metrics.

**Risks**

- The project can drift into a retrieval benchmark or analytics tool.
- It says less about the generator, reasoning, provenance, or research evolution.
- The chosen paper corpus must preserve fair comparisons across datasets and settings.

**Best fit:** A system that recommends or explains a retrieval design for a specific RAG research problem.

### C. RAG evaluation and factuality

**Question shape:** How should RAG systems be evaluated for retrieval quality, groundedness, factuality, completeness, reasoning, and efficiency?

**Useful corpus:** RAGAS; ARES; RagChecker; FRAMES; RAGTruth and related factuality/robustness benchmarks.

**Strengths**

- Directly addresses reliability and the distinction between good retrieval and good answers.
- Supports a strong final evaluation story for our own prototype.
- Naturally yields actionable output such as “retrieval failure,” “unsupported claim,” or “reasoning failure.”

**Risks**

- Evaluation methods themselves require careful annotation, judge validation, and metric interpretation.
- The system could become an evaluation dashboard rather than a research-paper onboarding assistant.
- Some metrics depend on LLM judges or reference answers, complicating reproducibility.

**Best fit:** A system that diagnoses why a RAG method or paper is unreliable and recommends what to evaluate next.

### D. Advanced / multi-stage / adaptive RAG

**Question shape:** When should a RAG system retrieve, reformulate, iterate, critique, or stop?

**Useful corpus:** FLARE; Self-RAG; corrective and iterative retrieval; query decomposition; agentic or modular RAG methods.

**Strengths**

- Strong research problem with explicit reasoning and decision points.
- Rich relationships: `TRIGGERS_RETRIEVAL`, `REFLECTS_ON`, `ITERATES_OVER`, `REDUCES_NOISE`, and `REQUIRES_EVIDENCE`.
- Makes a compelling demonstration on multi-hop or evolving research questions.

**Risks**

- Harder to implement and compare fairly because methods differ in training, inference budget, and model access.
- More difficult to keep the knowledge model compact and human-auditable.
- Evaluation must separate gains from extra retrieval calls, model size, and compute.

**Best fit:** A system that recommends a retrieval/reasoning path for a new RAG research question.

### E. RAG for long-context and document-intensive tasks

**Question shape:** How should RAG preserve document-level context and answer questions requiring synthesis across long or multiple documents?

**Useful corpus:** LongRAG; RAPTOR; GraphRAG; long-context versus RAG comparison studies; multi-hop/document QA benchmarks.

**Strengths**

- Clear practical motivation: short chunks can lose context, while long contexts can be costly or difficult to navigate.
- Supports a compelling onboarding output about which structural strategy fits a new paper or document collection.
- Naturally supports a knowledge structure with papers, document granularity, hierarchy, communities, evidence spans, and task types.

**Risks**

- “Long-context” and “RAG” overlap, so the scope needs a precise boundary.
- Building faithful hierarchical or graph representations can be labor-intensive.
- Long-context experiments may require more compute and model-specific infrastructure.

**Best fit:** A system that helps a researcher choose between chunk retrieval, hierarchical retrieval, graph retrieval, or long-context reading for a document-intensive RAG task.

### F. Structured / graph-based RAG for multi-hop research synthesis

**Question shape:** When does explicit structure—entities, relations, communities, or citation graphs—improve RAG over flat chunk retrieval?

**Useful corpus:** GraphRAG and GraphRAG surveys; knowledge-graph QA; RAPTOR as a non-graph structural comparator; multi-hop RAG evaluations.

**Strengths**

- Best alignment with the assignment's required inspectable entities, relationships, and reasoning over a knowledge structure.
- Makes the prototype demonstrably more than a search engine: it can traverse paper–method–problem–limitation paths.
- Supports clear human approval of the knowledge model before implementation.

**Risks**

- The project must carefully distinguish a human-approved research knowledge state from an automatic KG construction tool.
- Entity/relation extraction errors and provenance management become central concerns.
- GraphRAG is a large and fast-moving family; the corpus boundary must be narrow.

**Best fit:** A system that explains how a new RAG idea connects to prior methods through explicit multi-hop evidence.

## Preliminary comparison

| Scope | Main output | Main evaluation | Knowledge-structure fit | Implementation risk |
|---|---|---|---|---|
| A. Evolution | Positioning and lineage | Coverage, relation correctness, expert judgment | High | Medium |
| B. Retrieval | Technique recommendation/comparison | IR metrics, cost, latency | Medium | Medium |
| C. Evaluation | Diagnosis and evaluation plan | Correlation, calibration, human agreement | High | High |
| D. Advanced/multi-stage | Strategy selection and reasoning path | Answer quality vs. retrieval/compute budget | High | High |
| E. Long-context | Document strategy recommendation | Multi-hop QA, context coverage, cost | High | Medium-high |
| F. Structured/graph | Multi-hop evidence-based connection map | Path/evidence correctness, expert judgment | **Very high** | Medium-high |

## Scope-selection guidance

The options are not equally broad. A, B, and C are relatively conventional literature-review scopes. D and E focus on specific technical failure modes. F is the most naturally compatible with the required inspectable knowledge state and structural reasoning, but it also creates the strongest requirement to keep graph construction human-auditable and narrowly scoped.

Decision #1 is now approved: **F. Structured / graph-based RAG for multi-hop research synthesis**.

## Focused-scope literature implications

The selected scope is supported by several complementary lines of work:

- GraphRAG surveys organize the workflow into graph-based indexing, graph-guided retrieval, and graph-enhanced generation, with explicit attention to multi-hop retrieval and evaluation [1](https://arxiv.org/html/2408.08921v2).
- SG-RAG formulates multi-hop questions as subgraph retrieval problems and exposes subject–relation–object structure to the generator, making paths inspectable rather than returning only isolated chunks [2](https://aclanthology.org/2024.icnlsp-1.45.pdf).
- GRAG explicitly discusses networked documents such as citation graphs and the need to use mutual-reference links for broader technical-evolution understanding [3](https://aclanthology.org/2025.findings-naacl.232.pdf).
- Academic graph retrieval work has explored citation recommendation from a seed paper using subgraph expansion and verbalized graph triplets, which is close to a research-paper onboarding workflow [4](https://arxiv.org/html/2512.16661).
- Document GraphRAG work identifies dispersed evidence and multi-hop question answering as settings where graph structure may improve over isolated vector retrieval, while also emphasizing the need for meaningful edges and multiple informed retrieval steps [5](https://www.mdpi.com/2079-9292/14/11/2102).

These findings suggest that the project should not define the task as “retrieve similar papers.” The stronger research-onboarding task is to accept a new idea or paper, traverse a human-approved graph of scholarly concepts and evidence, and produce an explanation of how the input connects to prior work.

## Decision #2 candidate use cases

### A. Position a new RAG research idea in the prior-work landscape

- **User input:** A new RAG idea or short abstract that is not in the original paper corpus.
- **Graph reasoning:** Traverse `Idea -> Method -> Problem`, `Method -> Paper`, `Paper -> Dataset/Metric`, `Paper -> Limitation`, and `Limitation -> Research Direction` paths.
- **Output:** A structured positioning brief containing closest prior methods, supporting multi-hop paths, overlap versus difference, unresolved limitations, evidence citations, and recommended next readings.
- **Calyb fit:** The unseen input is not merely searched; it is mapped into an inspectable knowledge structure and used to produce an actionable research-positioning output.
- **Implementation difficulty:** Medium-high; requires robust entity normalization, path ranking, provenance, and careful handling of uncertainty.
- **Evaluation challenge:** Experts must judge whether the proposed connections, differences, and cited evidence are correct and useful without treating the system as an oracle for novelty.

### B. Trace the foundations of a new RAG paper

- **User input:** A new paper's title, abstract, and optionally its reference list or DOI; the paper itself is excluded from the original corpus.
- **Graph reasoning:** Follow citation and method-dependency paths backward from the new paper's claimed contribution to foundational papers, retrievers, architectures, datasets, and problem definitions.
- **Output:** A foundation map with ranked ancestor papers, path explanations, contribution-to-foundation links, and a reading order.
- **Calyb fit:** It demonstrates multi-hop reasoning over explicit scholarly relationships and produces a useful onboarding artifact rather than a flat search-result list.
- **Implementation difficulty:** Medium; citation edges are relatively concrete, but semantic method-dependency edges require evidence extraction and human review.
- **Evaluation challenge:** Citation proximity is not the same as conceptual importance; experts must validate whether the selected ancestors genuinely support the claimed lineage.

### C. Connect a known limitation to later research directions

- **User input:** One limitation statement from a new or existing RAG paper, such as “retrieval misses evidence spread across documents.”
- **Graph reasoning:** Traverse `Limitation -> Method`, `Method -> Paper`, `Paper -> Dataset/Task`, and temporal `Paper -> Later Paper -> Addressed Limitation` paths.
- **Output:** A limitation-to-direction map showing which later methods address the limitation, what they change, what evidence they use, and which gaps remain.
- **Calyb fit:** The system turns an input limitation into an evidence-backed next-step research plan using explicit relations and multi-hop paths.
- **Implementation difficulty:** High; requires temporal reasoning, limitation normalization, and distinguishing “mentions a limitation” from “actually addresses it.”
- **Evaluation challenge:** Later work may only partially address a limitation, so evaluation requires fine-grained expert labels for addressability, evidence strength, and remaining gap.

### D. Select an evidence-backed benchmark and comparison set for a new idea

- **User input:** A new RAG idea plus its target task, anticipated failure mode, and constraints such as domain or available compute.
- **Graph reasoning:** Traverse `Idea -> Problem -> Task`, `Task -> Dataset`, `Method -> Dataset`, `Method -> Metric`, and `Dataset -> Known Limitation` paths.
- **Output:** A structured experiment-starting plan: recommended datasets, baseline methods, metrics, expected failure cases, and reasons for each recommendation.
- **Calyb fit:** It produces an actionable research plan from graph evidence rather than merely reporting dataset statistics or retrieving papers.
- **Implementation difficulty:** Medium-high; requires normalized metadata and reliable method–dataset–metric links.
- **Evaluation challenge:** There may be several defensible benchmark choices; evaluation needs expert agreement or rubric-based assessment rather than one exact answer.

### E. Compare two RAG approaches through multi-hop evidence

- **User input:** Two candidate RAG methods or papers and one comparison question, such as “Which approach is better for multi-hop research-paper synthesis under a small compute budget?”
- **Graph reasoning:** Traverse each method to its problem assumptions, graph/context representation, datasets, metrics, limitations, and later extensions, then compare matched paths.
- **Output:** A structured comparison containing aligned criteria, evidence paths for each claim, trade-offs, unresolved risks, and a recommendation conditioned on the stated constraints.
- **Calyb fit:** It requires reasoning over interconnected evidence and produces a decision-support artifact, not an ungrounded LLM opinion or a similarity ranking.
- **Implementation difficulty:** High; comparison requires schema alignment and normalization across papers with different terminology and experimental protocols.
- **Evaluation challenge:** Recommendations are conditional; evaluators must judge both factual support and whether the stated constraints were applied consistently.

### Candidate-use-case guidance

Option A is the broadest direct match to research-paper onboarding and the user's stated goal. Option B is the most concrete and likely easiest to evaluate. Option C provides the strongest research-direction story but has the highest semantic and temporal complexity. Option D is highly actionable for research planning. Option E is useful for decision support but risks becoming a comparison or analytics tool if its evidence paths are not central.

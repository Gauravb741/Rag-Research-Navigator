# Project Decision Roadmap

## Project context

- **Domain:** Domain B — Research Paper Onboarding
- **Overall topic:** Retrieval-Augmented Generation (RAG)
- **Repository state at project start:** Empty; no assignment brief, source code, tests, or configuration files were present in the repository. The governing requirements are the user-provided interactive decision-making brief and its Calyb AI compliance checklist.
- **Working rule:** Major research, knowledge-modeling, architecture, scope, UX, and evaluation decisions remain pending until explicitly approved by the user.

## Major decisions to make

| # | Decision | Why it matters | Status |
|---:|---|---|---|
| 1 | Exact RAG research scope | Determines the literature corpus, research question, entities, relationships, and the kind of onboarding advice the system can produce. | **Approved — F: Structured / graph-based RAG for multi-hop research synthesis** |
| 2 | One concrete use case | Converts the broad topic into a single actionable research-onboarding problem and defines the user journey. | **Approved — A: Position a new RAG research idea in the prior-work landscape** |
| 3 | Paper selection and corpus boundaries | Determines which real papers are included, how inclusion/exclusion is justified, and what counts as an unseen new input. | **Implemented autonomously — OpenAlex, deterministic focused selection, 72 papers** |
| 4 | Knowledge entities and relationships | Defines the inspectable knowledge model and what structural reasoning is possible. | **Approved — controlled-vocabulary typed graph with provenance** |
| 5 | Knowledge-state representation and provenance | Determines how facts, claims, citations, confidence, and source spans remain inspectable and reproducible. | **Implemented autonomously — JSON knowledge state with evidence-bearing relationships** |
| 6 | New-input format and boundary condition | Defines the paper/idea/query that is intentionally absent from the original corpus and how it enters the system. | **Implemented autonomously — free-text unseen RAG idea/abstract** |
| 7 | Reasoning strategy | Determines whether outputs rely on rules, graph traversal, semantic retrieval, LLM explanation, or a hybrid. | **Implemented autonomously — lexical candidate location plus explicit graph traversal and evidence attachment** |
| 8 | Structured actionable output | Defines what the system returns: e.g., related prior work, novelty links, limitations, evidence, and recommended next steps. | **Implemented autonomously — JSON positioning report plus readable CLI** |
| 9 | Evaluation methodology | Determines tasks, baselines, metrics, human checks, and how we demonstrate reasoning rather than search alone. | **Implemented autonomously — four unseen ideas, graph-vs-flat diagnostics, path/evidence checks** |
| 10 | Implementation architecture and storage | Determines whether the inspectable knowledge state uses JSON, SQLite, a graph database, vector retrieval, or a hybrid. | **Implemented autonomously — Python + inspectable JSON; no graph database required** |
| 11 | Testable interface | Determines whether the reviewer uses a CLI, web UI, notebook, or a combination. | **Implemented autonomously — CLI plus secure same-origin web UI with JSON mode** |
| 12 | Reproducibility and demonstration strategy | Determines setup, data provenance, offline/online dependencies, demo script, documentation, and final deliverables. | **Complete — security review, documentation, tests, build, and validation passed** |

## Decision protocol

1. Present only the next meaningful decision.
2. Explain practical alternatives, advantages, disadvantages, and a recommendation.
3. Wait for the user's explicit choice.
4. Record the selected option and rationale in `docs/DECISIONS.md`.
5. Implement the approved decision, then proceed to the next decision.

## Current gate

**Decisions #1, #2, and the final human approval of the knowledge model are approved. The autonomous engineering phase and final validation are complete.

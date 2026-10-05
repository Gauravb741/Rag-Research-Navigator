"""Hand-authored ontology: vocabulary, schema, cue lists and curated priors.

Nothing here is produced by an extraction library. Every alias, cue word and
relationship definition is written by hand so the mapping can be audited and
argued with. ``scripts/build_knowledge.py`` applies these rules to the corpus.

Alias syntax (see ``src/modeling/matching.py``):
  * aliases are matched case-insensitively on word boundaries
  * a space or hyphen inside an alias matches either ("multi-hop" == "multi hop" == "multihop")
  * a trailing ``*`` allows any word ending ("diagnos*" matches "diagnosis", "diagnostic")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TermSpec:
    entity_id: str
    entity_type: str
    name: str
    description: str
    aliases: tuple[str, ...]
    # Method only. "paradigm" = a family of approaches (GraphRAG, standard RAG);
    # "named" = one specific published method that a paper can introduce.
    kind: str = ""


def _t(eid, etype, name, desc, aliases, kind=""):
    return TermSpec(eid, etype, name, desc, tuple(aliases), kind)


TERMS: tuple[TermSpec, ...] = (
    # ------------------------------------------------------------------ methods
    _t("method_standard_rag", "Method", "Standard RAG", "Retrieve text chunks and condition generation on them.", ["standard rag", "vanilla rag", "naive rag", "conventional rag", "traditional rag", "retrieval-augmented generation", "retrieval augmented generation"], "paradigm"),
    _t("method_graphrag", "Method", "GraphRAG", "Family of RAG approaches that index or retrieve through a graph.", ["graphrag", "graph rag", "graph-based rag", "graph retrieval-augmented generation", "graph retrieval augmented generation", "graph-based retrieval-augmented generation", "graph-enhanced rag", "kg-rag"], "paradigm"),
    _t("method_dpr", "Method", "DPR", "Dense Passage Retrieval for open-domain question answering.", ["dense passage retrieval", "dpr"], "named"),
    _t("method_g_retriever", "Method", "G-Retriever", "RAG over textual graphs using a retrieved subgraph.", ["g-retriever"], "named"),
    _t("method_grag", "Method", "GRAG", "Graph RAG for networked textual documents (subgraph retrieval + soft prompting).", ["grag"], "named"),
    _t("method_kg2rag", "Method", "KG²RAG", "Knowledge-graph-guided chunk expansion and organisation for RAG.", ["kg²rag", "kg2rag"], "named"),
    _t("method_hipporag", "Method", "HippoRAG", "Neurobiologically inspired long-term memory for retrieval.", ["hipporag"], "named"),
    _t("method_raptor", "Method", "RAPTOR", "Recursive abstractive tree-organised retrieval.", ["raptor", "recursive abstractive processing"], "named"),
    _t("method_gnn_rag", "Method", "GNN-RAG", "GNN reasons over a KG subgraph, shortest paths are verbalised for the LLM.", ["gnn-rag"], "named"),
    _t("method_sg_rag", "Method", "SG-RAG", "Subgraph retrieval-augmented generation for multi-hop QA.", ["sg-rag"], "named"),
    _t("method_structugraphrag", "Method", "StructuGraphRAG", "Document-structure-informed knowledge graphs for RAG.", ["structugraphrag"], "named"),
    _t("method_document_graphrag", "Method", "Document GraphRAG", "Document-oriented knowledge-graph-enhanced RAG.", ["document graphrag"], "named"),
    _t("method_hypergraphrag", "Method", "HyperGraphRAG", "Hypergraph (n-ary relation) knowledge representation for RAG.", ["hypergraphrag"], "named"),
    _t("method_cg_rag", "Method", "CG-RAG", "Research question answering via citation-graph retrieval.", ["cg-rag"], "named"),
    _t("method_graph_prompting", "Method", "Knowledge Graph Prompting", "Graph structure prompts multi-document question answering.", ["knowledge graph prompting"], "named"),
    _t("method_active_rag", "Method", "Active RAG (FLARE)", "Retrieval triggered during generation when the model is uncertain.", ["active retrieval augmented generation", "active rag", "flare"], "named"),
    # Named methods that the corpus does not cover; kept so an idea that
    # mentions them can be told "known method, not in the indexed corpus".
    _t("method_self_rag", "Method", "Self-RAG", "Adaptive retrieval with self-critique tokens.", ["self-rag"], "named"),
    _t("method_lightrag", "Method", "LightRAG", "Dual-level graph retrieval with incremental updates.", ["lightrag"], "named"),
    _t("method_think_on_graph", "Method", "Think-on-Graph", "LLM agent explores a knowledge graph step by step.", ["think-on-graph"], "named"),
    _t("method_ircot", "Method", "IRCoT", "Interleaves retrieval with chain-of-thought for multi-step QA.", ["ircot"], "named"),
    _t("method_hyde", "Method", "HyDE", "Hypothetical document embeddings for zero-shot dense retrieval.", ["hyde"], "named"),
    # ----------------------------------------------------------------- problems
    _t("problem_multihop_reasoning", "ResearchProblem", "Multi-hop reasoning", "Answering requires chaining evidence across several relations or documents.", ["multi-hop reasoning", "multi-hop question answering", "multi-hop queries", "multi-hop qa", "multi-hop questions", "multi-hop retrieval", "multi-step reasoning", "multi-step inference", "mhqa"]),
    _t("problem_evidence_fragmentation", "ResearchProblem", "Evidence fragmentation", "Relevant evidence is scattered across chunks, documents or graph neighbourhoods.", ["fragmented", "fragmentation", "dispersed evidence", "distributed evidence", "evidence distributed", "evidence across", "information silo*", "scattered"]),
    _t("problem_hallucination", "ResearchProblem", "Hallucination and factuality", "Generated claims may be unsupported or factually wrong.", ["hallucinat*", "factuality", "factual consistency", "factual accuracy", "factually incorrect", "unreliab*"]),
    _t("problem_long_context", "ResearchProblem", "Long-context document understanding", "Reasoning over long or document-intensive inputs without losing structure.", ["long-context", "long document*", "long papers", "full papers", "document-intensive", "lost in the middle"]),
    _t("problem_graph_qa", "ResearchProblem", "Knowledge-graph question answering", "Answering questions by querying or reasoning over entity-relation graphs.", ["knowledge graph question answering", "graph question answering", "kgqa", "kbqa", "knowledge base question answering"]),
    _t("problem_citation_research_qa", "ResearchProblem", "Citation-aware research question answering", "Answering research questions using citation and scholarly structure.", ["research question answering", "citation graph", "citation-aware", "academic literature", "scientific literature", "scholarly"]),
    _t("problem_noisy_retrieval", "ResearchProblem", "Noisy or conflicting retrieval", "Retrieved context can be irrelevant, redundant, misleading or contradictory.", ["noisy", "irrelevant information", "irrelevant context", "retrieval noise", "inter-context conflict*", "conflicting evidence", "conflicting documents", "distractor*", "redundan*"]),
    _t("problem_provenance", "ResearchProblem", "Provenance and explainability", "Users must see why a claim was produced and which source supports it.", ["provenance", "explainab*", "interpretab*", "traceab*", "transparen*", "trustworth*", "verifiab*", "black-box"]),
    _t("problem_kg_construction", "ResearchProblem", "Knowledge-graph construction from text", "Building entity-relation graphs from unstructured documents.", ["knowledge graph construction", "kg construction", "knowledge graph building", "triplet extraction", "triple extraction", "entity and relation extraction", "entity extraction"]),
    _t("problem_domain_adaptation", "ResearchProblem", "Domain adaptation of RAG", "Making RAG work in specialised, terminology-heavy domains.", ["domain-specific", "domain specific", "specialized domain*", "specialised domain*", "vertical-domain", "domain adaptation", "domain-adapted", "domain-authentic"]),
    _t("problem_knowledge_gap", "ResearchProblem", "LLM knowledge gaps", "LLMs lack up-to-date or private knowledge.", ["knowledge gap*", "outdated knowledge", "lack of knowledge", "knowledge-intensive", "beyond their knowledge", "private knowledge"]),
    _t("problem_security", "ResearchProblem", "Security and robustness of RAG", "Poisoning and adversarial manipulation of retrieved knowledge.", ["poisoning", "adversarial", "attack*", "vulnerab*", "security implications"]),
    _t("problem_efficiency_cost", "ResearchProblem", "Retrieval efficiency and token cost", "Keeping retrieval fast and token usage low.", ["token cost", "token consumption", "token overhead", "retrieval efficiency", "efficiency-accuracy", "computational overhead"]),
    _t("problem_query_complexity", "ResearchProblem", "Complex query understanding", "Decomposing or planning for queries needing several steps.", ["complex quer*", "complex question*", "complex reasoning", "sub-question*", "subproblem*", "decompos*"]),
    # ----------------------------------------------------------------- datasets
    _t("dataset_hotpotqa", "Dataset", "HotpotQA", "Multi-hop QA over linked Wikipedia passages.", ["hotpotqa", "hotpot qa"]),
    _t("dataset_2wikimultihopqa", "Dataset", "2WikiMultiHopQA", "Multi-hop QA benchmark built from Wikipedia and Wikidata.", ["2wikimultihopqa", "2wiki"]),
    _t("dataset_musique", "Dataset", "MuSiQue", "Compositional multi-hop QA benchmark.", ["musique"]),
    _t("dataset_webqsp", "Dataset", "WebQSP", "Question answering over Freebase.", ["webqsp"]),
    _t("dataset_cwq", "Dataset", "CWQ", "ComplexWebQuestions over Freebase.", ["cwq", "complexwebquestions"]),
    _t("dataset_metaqa", "Dataset", "MetaQA", "Multi-hop QA over a movie knowledge graph.", ["metaqa"]),
    _t("dataset_graphqa", "Dataset", "GraphQA", "QA over textual graphs.", ["graphqa"]),
    _t("dataset_multihop_rag", "Dataset", "MultiHop-RAG", "Benchmark for RAG on multi-hop queries.", ["multihop-rag"]),
    _t("dataset_graphrag_bench", "Dataset", "GraphRAG-Bench", "Benchmark for domain-specific graph-RAG reasoning.", ["graphrag-bench"]),
    _t("dataset_magic", "Dataset", "MAGIC", "Multi-hop graph benchmark for inter-context conflicts.", ["magic"]),
    _t("dataset_qasper", "Dataset", "QASPER", "QA over scientific papers.", ["qasper"]),
    _t("dataset_nq", "Dataset", "Natural Questions", "Open-domain QA benchmark.", ["natural questions", "nq dataset"]),
    _t("dataset_triviaqa", "Dataset", "TriviaQA", "Open-domain QA benchmark.", ["triviaqa"]),
    _t("dataset_pubmedqa", "Dataset", "PubMedQA", "Biomedical QA benchmark.", ["pubmedqa"]),
    _t("dataset_medqa", "Dataset", "MedQA", "Medical licensing exam QA.", ["medqa"]),
    _t("dataset_squad", "Dataset", "SQuAD", "Reading-comprehension QA.", ["squad"]),
    # ------------------------------------------------------------------ metrics
    _t("metric_exact_match", "Metric", "Exact Match", "Exact string match of an answer.", ["exact match", "em score"]),
    _t("metric_f1", "Metric", "F1", "Token- or span-level F1.", ["f1", "f1-score", "f1 score"]),
    _t("metric_accuracy", "Metric", "Accuracy", "Fraction of correct predictions.", ["accuracy"]),
    _t("metric_precision_recall", "Metric", "Precision / recall", "Retrieval or answer precision and recall.", ["precision", "answer recall", "context recall", "recall@k", "recall at k"]),
    _t("metric_hit_at_1", "Metric", "Hit@1", "Top-ranked result is correct.", ["hit@1", "hit at 1"]),
    _t("metric_rouge", "Metric", "ROUGE", "N-gram overlap for generated text.", ["rouge"]),
    _t("metric_bertscore", "Metric", "BERTScore", "Embedding-based semantic similarity.", ["bertscore"]),
    _t("metric_faithfulness", "Metric", "Faithfulness", "Consistency of a response with its evidence.", ["faithfulness", "groundedness"]),
    _t("metric_context_relevance", "Metric", "Context relevance", "Relevance of retrieved context to the input.", ["context relevance", "context relevancy", "answer relevan*"]),
    # ------------------------------------------------------------ architectures
    _t("arch_vector_rag", "Architecture", "Vector retrieval", "Flat chunk retrieval by embedding similarity.", ["vector retrieval", "vector database", "vector search", "vector-based", "dense retrieval", "embedding-based retrieval", "chunk-based retrieval"]),
    _t("arch_graph_index", "Architecture", "Graph-indexed retrieval", "Graph indexing and graph-guided retrieval.", ["graph-based retrieval", "graph-guided retrieval", "graph-enhanced generation", "graph indexing", "graph index", "graph traversal"]),
    _t("arch_subgraph_retrieval", "Architecture", "Subgraph retrieval", "Retrieve a connected neighbourhood or path as evidence.", ["subgraph*", "shortest path*", "path retrieval", "reasoning path*"]),
    _t("arch_hierarchical", "Architecture", "Hierarchical retrieval", "Retrieve at several levels of abstraction.", ["hierarchical", "tree-organized", "tree-organised", "community summar*", "multi-granularity", "cross-granularity"]),
    _t("arch_hybrid_graph_vector", "Architecture", "Hybrid graph + vector retrieval", "Combine semantic chunk retrieval with graph structure.", ["hybrid graph", "graph and vector", "vector and graph", "dual pathway", "hybrid retriev*", "hybrid search", "dual-graph", "dual graph", "dual knowledge"]),
    _t("arch_hypergraph", "Architecture", "Hypergraph representation", "N-ary relations as hyperedges.", ["hypergraph*", "hyperedge*", "n-ary", "hyper-relational"]),
    _t("arch_gnn", "Architecture", "Graph neural network retriever", "A GNN scores or reasons over the retrieved graph.", ["graph neural network*", "gnn", "graph foundation model"]),
    _t("arch_agentic", "Architecture", "Agentic / iterative retrieval", "Agents or loops retrieve, evaluate and refine repeatedly.", ["agentic", "multi-agent", "iterative", "self-correct*", "retrieve-evaluate-refine", "reflect*"]),
    _t("arch_reinforcement_learning", "Architecture", "Reinforcement-learned retrieval", "Retrieval or reasoning policy trained with RL.", ["reinforcement learning", "reward"]),
    _t("arch_query_decomposition", "Architecture", "Query decomposition / logic trees", "Break a query into sub-questions or a dependency graph.", ["query decomposition", "question decomposition", "logic tree", "logic-aware", "dependency graph", "topological sort"]),
    # -------------------------------------------------------------- limitations
    # Limitations *of prior approaches*, as stated in abstracts (see MOTIVATED_BY_LIMITATION).
    _t("limitation_graph_construction_cost", "Limitation", "Graph construction cost", "Building and maintaining the graph is expensive.", ["construction cost", "costly process", "costly graph", "expensive to construct", "graph construction", "extraction cost", "pre-built graph*", "pre-constructed graph*", "pre-built", "prebuilt"]),
    _t("limitation_noisy_extraction", "Limitation", "Noisy or incomplete extraction", "Wrong or missing triples create misleading paths.", ["noisy extraction", "extraction error*", "extraction noise", "noise and incomplete*", "noise within", "error propagation", "accumulat*"]),
    _t("limitation_latency", "Limitation", "Latency and inference cost", "Multi-stage retrieval adds runtime cost.", ["latency", "inference cost", "computational cost", "runtime overhead", "resource-intensive", "slow"]),
    _t("limitation_scalability", "Limitation", "Scalability", "Large corpora or graphs make indexing or traversal hard.", ["scalability", "large-scale graph*", "large graph*", "does not scale", "scale poorly"]),
    _t("limitation_dynamic_updates", "Limitation", "Hard to update", "Adding documents needs costly re-indexing.", ["update latency", "update cost", "static knowledge", "static knowledge base*", "dynamic dataset*", "dynamic knowledge", "frequently updated", "incremental update*"]),
    _t("limitation_domain_specificity", "Limitation", "Limited generalisation", "Results may not transfer beyond the evaluated domain.", ["limited generaliz*", "limited generalis*", "generalizability", "tailored to specific"]),
    _t("limitation_missing_provenance", "Limitation", "Opaque reasoning / weak attribution", "Answers cannot be verified against sources.", ["opaque reasoning", "black-box nature", "lack of explainability", "source attribution", "citation accuracy", "hardly scrutable"]),
    _t("limitation_context_loss", "Limitation", "Context loss", "Chunking or compression loses context.", ["context loss", "semantic fragmentation", "information loss", "semantic gaps", "disrupts the intrinsic", "flat, unstructured", "flat-text", "flattening", "isolated chunks"]),
    _t("limitation_binary_relations", "Limitation", "Binary-relation limit", "Ordinary graphs cannot express n-ary facts.", ["binary relations", "constrained by binary"]),
    _t("limitation_flat_retrieval", "Limitation", "Flat retrieval misses structure", "Chunk retrieval ignores relations between pieces of knowledge.", ["fail to capture", "struggle to capture", "conventional rags", "traditional rag", "traditional retrieval", "flat retrieval", "reliance on flat"]),
    _t("limitation_reasoning_depth", "Limitation", "Shallow multi-step reasoning", "Single-pass retrieval cannot support multi-step inference.", ["multi-step inference", "multi-hop reasoning", "limited reasoning", "intricacies of multi-step", "over-planning"]),
    _t("limitation_token_cost", "Limitation", "Token and compute overhead", "Graph methods consume many tokens.", ["token cost", "token overhead", "overwhelming token", "context overload"]),
    # --------------------------------------------------------------- directions
    # Directions a paper *pursues* (or names as future work), see PURSUES_DIRECTION.
    _t("direction_efficient_graph_retrieval", "ResearchDirection", "Efficient graph retrieval", "Cut retrieval cost while keeping useful paths.", ["efficient graph retrieval", "adaptive pruning", "graph pruning", "context pruning", "efficient retriev*", "lightweight", "token efficien*", "reduc* token"]),
    _t("direction_hybrid_retrieval", "ResearchDirection", "Hybrid graph-vector retrieval", "Combine graph structure with semantic retrieval.", ["hybrid retriev*", "hybrid graph", "graph and vector", "hybrid search", "hybrid integration", "dual retrieval", "blended"]),
    _t("direction_dynamic_knowledge", "ResearchDirection", "Updateable / dynamic knowledge", "Keep structured knowledge current as sources change.", ["incremental*", "continual update", "dynamic knowledge", "dynamically construct*", "dynamic local", "real-time", "update mechanism*", "knowledge update*", "online"]),
    _t("direction_citation_aware", "ResearchDirection", "Citation-aware research synthesis", "Use citation structure to synthesise research.", ["citation-aware", "citation graph", "research synthesis", "citation network"]),
    _t("direction_robust_evaluation", "ResearchDirection", "Rigorous GraphRAG evaluation", "Benchmarks and protocols for graph-RAG quality.", ["graphrag-bench", "systematic evaluation", "comprehensive evaluation", "benchmark study", "evaluation protocol", "unified evaluation", "empirical stud*", "benchmarking"]),
    _t("direction_multihop_reasoning", "ResearchDirection", "Better multi-hop reasoning", "Improve decomposition, traversal and synthesis over hops.", ["reasoning chain*", "iterative reasoning", "multi-hop reasoning", "reasoning structure*", "multi-step"]),
    _t("direction_agentic", "ResearchDirection", "Agentic and self-correcting RAG", "Agents critique and repair retrieval.", ["agentic", "self-correct*", "self-cognitive", "metacognitive", "reflective", "reinforcement learning", "process-constrained"]),
    _t("direction_adaptive_retrieval", "ResearchDirection", "Adaptive retrieval", "Decide when and what to retrieve per query.", ["adaptive retriev*", "adaptive reasoning", "adaptive multi*", "query-driven", "adaptive"]),
    _t("direction_multimodal_structured", "ResearchDirection", "Multimodal / heterogeneous knowledge", "Retrieve over tables, images and text together.", ["multimodal", "heterogeneous", "tabular"]),
    # ------------------------------------------------------- application domains
    _t("domain_medical", "ApplicationDomain", "Medicine and healthcare", "Clinical and biomedical question answering.", ["medical", "clinical", "healthcare", "patient*", "hepatology", "kidney", "diagnos*", "traditional chinese medicine", "dietary supplement*", "biomedical"]),
    _t("domain_legal", "ApplicationDomain", "Law", "Legal research and statutes.", ["legal", "statute*", "law"]),
    _t("domain_finance", "ApplicationDomain", "Finance and economics", "Financial and economic data.", ["finance", "financial", "economic"]),
    _t("domain_security", "ApplicationDomain", "Cybersecurity", "Threat intelligence and network security.", ["cybersecurity", "cyber threat", "network security", "threat intelligence"]),
    _t("domain_education", "ApplicationDomain", "Education", "Tutoring and learning-path support.", ["education*", "learner*", "tutor*", "learning path*", "course"]),
    _t("domain_telecom", "ApplicationDomain", "Telecommunications", "Wireless and radio networks.", ["wireless", "oran", "telecom*", "radio access"]),
    _t("domain_agriculture", "ApplicationDomain", "Agriculture", "Crop and pest knowledge.", ["crop*", "agricultur*", "pest*"]),
    _t("domain_engineering", "ApplicationDomain", "Engineering and infrastructure", "Construction, power systems, geospatial and systems engineering.", ["construction", "power system*", "geospatial", "system of systems", "engineering"]),
    _t("domain_public_policy", "ApplicationDomain", "Government, policy and logistics", "Public-sector assistants and policy generation.", ["e-government", "policy", "government", "logistics", "customer service"]),
    _t("domain_culture_tourism", "ApplicationDomain", "Culture and tourism", "Art, tourism and historical text.", ["tourism", "art understanding", "visual art", "historical text*", "cultural"]),
    _t("domain_scholarly", "ApplicationDomain", "Scientific literature", "Research papers and scholarly knowledge.", ["research paper*", "scientific paper*", "academic literature", "scientific knowledge", "scholarly"]),
    # ----------------------------------------------------------------- concepts
    _t("concept_entity_relation_graph", "Concept", "Entity-relation graph", "Nodes are entities, edges are typed relations.", ["entity-relation", "entities and relations", "entity relation", "knowledge graph*", "knowledge-graph", "triple*", "triplet*"]),
    _t("concept_evidence_path", "Concept", "Evidence path", "A connected sequence of relations that supports a claim.", ["evidence path*", "source path*", "reasoning path*", "knowledge path*", "path retrieval"]),
    _t("concept_citation_network", "Concept", "Citation network", "Papers linked by citations.", ["citation network*", "citation graph*", "mutual referencing", "citation link*"]),
    _t("concept_graph_indexing", "Concept", "Graph indexing", "Turn documents into a graph for retrieval.", ["graph indexing", "graph-indexed", "graph index"]),
    _t("concept_query_decomposition", "Concept", "Query decomposition", "Split a question into sub-questions.", ["query decomposition", "question decomposition", "decompos*"]),
    _t("concept_provenance", "Concept", "Provenance", "Source-location metadata attached to a claim.", ["provenance", "source attribution", "evidence source*"]),
    _t("concept_multi_hop", "Concept", "Multi-hop", "Reasoning over several linked facts.", ["multi-hop"]),
    _t("concept_graph_community", "Concept", "Graph communities", "Clusters used to summarise or navigate a graph.", ["community detection", "community summar*", "graph communit*"]),
)


# Bare long-form phrases that every RAG paper contains; never evidence of a comparison target.
GENERIC_PARADIGM_ALIASES = frozenset({"retrieval-augmented generation", "retrieval augmented generation",
                                      "graph retrieval-augmented generation", "graph retrieval augmented generation",
                                      "graph-based retrieval-augmented generation"})


def iter_terms(entity_type: str | None = None) -> Iterable[TermSpec]:
    for term in TERMS:
        if entity_type is None or term.entity_type == entity_type:
            yield term


# --------------------------------------------------------------------- cue lists
# Regular-expression fragments (case-insensitive, word-boundary applied by the caller).
CUES = {
    "intro": r"propos\w*|introduc\w*|present\w*|develop\w*|devis\w*|design\w*|novel",
    "improve": r"outperform\w*|surpass\w*|exceed\w*|improv\w* (?:over|upon|on)|better than|superior|beat\w*",
    "negated_limitation": r"negligible|minimal|without|no additional|lower|eliminat\w*|avoid\w*",
    "extend": r"extend\w*|build\w* (?:on|upon)|based on|inspired by|follow\w* (?:up )?(?:on|the)|enhanc\w*|upon",
    "compare": r"compar\w*|baseline\w*|versus|vs\.?|against",
    "limitation": r"however|suffer\w*|struggl\w*|fail\w*|limited|limitation\w*|lack\w*|costly|expensive|hinder\w*|bottleneck\w*|overhead|constrain\w*|difficult\w*|prone|degrad\w*|unreliab\w*|overwhelming|ineffective|opaque|falter\w*|fall\w* short|resource-intensive|rely heavily|rely on",
    "problem": r"challeng\w*|problem\w*|issue\w*|address\w*|tackl\w*|struggl\w*|difficult\w*|limit\w*|task\w*|need\w*|overcome|mitigat\w*|alleviat\w*|reduc\w*",
    "evaluation": r"experiment\w*|evaluat\w*|benchmark\w*|dataset\w*|results?|demonstrat\w*|achiev\w*|metric\w*|score\w*|compar\w*|test\w*|validat\w*",
    "future": r"future (?:work|research|direction\w*)|promising direction\w*|remains? (?:open|an open)|open (?:challenge\w*|problem\w*)|we plan|research direction\w*",
}

# Paper-type rules applied to title (T) and abstract (A); first match wins.
PAPER_KIND_RULES = (
    ("survey", r"\b(survey|review|overview|tutorial|systematic literature|this book)\b", r"\b(this (survey|review|tutorial|book)|we (survey|review)|comprehensive (survey|review)|systematic literature review)\b"),
    ("benchmark", r"\b(bench|benchmark\w*|dataset)\b|bench\b", r"\bwe (introduce|present|propose|construct|build|release) (the |a |an |two )?(novel )?(new )?[^.]{0,40}(benchmark|dataset)s?\b"),
    ("empirical_study", r"\b(evaluation|comparative|comparison|vs\.?|study|investigat\w*|validation|case study|under fire|key insights)\b", r"\b(comprehensive benchmark study|systematic(ally)? (evaluat|compar)\w*|we (conduct|perform) a (comparative|systematic))\b"),
)

# ------------------------------------------------------------------- synonyms
# Hand-written phrase -> entity ids. Used only as a *fallback* match
# (confidence 0.7) when a user writes an idea in different words.
SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"scholarly (?:link|graph|network)s?|bibliograph\w+|who cites whom|reference list", ("concept_citation_network",)),
    (r"cites?|citing|cited", ("concept_citation_network",)),
    (r"hops?|several steps|chain of (?:facts|evidence)|connect(?:ing)? (?:facts|documents)|across (?:several|multiple) documents", ("problem_multihop_reasoning", "concept_multi_hop")),
    (r"make up facts|made-up|unsupported claims|wrong answers?|fabricat\w+", ("problem_hallucination",)),
    (r"show (?:its|the) sources?|cite sources?|audit(?:able)?|why (?:did|does) it answer|justif\w+", ("problem_provenance",)),
    (r"slow|too expensive|cost of (?:building|indexing)|cheaper|speed up|faster", ("problem_efficiency_cost", "direction_efficient_graph_retrieval")),
    (r"new documents arrive|keep(?:ing)? (?:it )?up to date|streaming|continuously updated|refresh", ("direction_dynamic_knowledge", "limitation_dynamic_updates")),
    (r"llm agents?|self-?check|critic|reflection", ("arch_agentic", "direction_agentic")),
    (r"images?|pictures?|tables?|charts?|multi-?modal", ("direction_multimodal_structured",)),
    (r"clinic\w+|doctors?|hospital\w*|drug\w*|disease\w*", ("domain_medical",)),
    (r"contracts?|court\w*|lawyer\w*|regulation\w*", ("domain_legal",)),
    (r"hyper-?edges?|n-ary", ("arch_hypergraph",)),
    (r"reinforce\w*|reward\w*|policy gradient|rl-trained", ("arch_reinforcement_learning",)),
    (r"poison\w*|jailbreak\w*|malicious|prompt injection", ("problem_security",)),
)

# ------------------------------------------------- curated limitation -> direction
# Hand-written priors. They are marked ``curated_ontology_prior`` in the graph and
# carry a one-line rationale so a reviewer can disagree with any of them.
CURATED_LIMITATION_DIRECTIONS = (
    ("limitation_graph_construction_cost", "direction_efficient_graph_retrieval", "Cheaper graph building and retrieval directly targets construction cost."),
    ("limitation_latency", "direction_efficient_graph_retrieval", "Latency is reduced by pruning and efficient retrieval."),
    ("limitation_scalability", "direction_efficient_graph_retrieval", "Scaling to large graphs needs efficient traversal."),
    ("limitation_token_cost", "direction_efficient_graph_retrieval", "Token overhead is attacked by pruning retrieved context."),
    ("limitation_dynamic_updates", "direction_dynamic_knowledge", "Incremental maintenance is the stated remedy for hard-to-update graphs."),
    ("limitation_context_loss", "direction_hybrid_retrieval", "Combining chunks with graph structure restores context that either loses alone."),
    ("limitation_flat_retrieval", "direction_hybrid_retrieval", "Adding structure to flat retrieval is the hybrid argument."),
    ("limitation_reasoning_depth", "direction_multihop_reasoning", "Shallow reasoning is addressed by better multi-hop reasoning."),
    ("limitation_reasoning_depth", "direction_agentic", "Iterative agents are a proposed fix for single-pass retrieval."),
    ("limitation_domain_specificity", "direction_robust_evaluation", "Generalisation claims need benchmarks beyond one domain."),
    ("limitation_missing_provenance", "direction_citation_aware", "Citation structure is one way to expose sources."),
)

# ------------------------------------------------------------------- schema
ENTITY_TYPES = {
    "Paper": "A paper in the indexed corpus (provenance anchor).",
    "Author": "Author from OpenAlex metadata.",
    "Method": "A RAG method. kind=paradigm (a family) or kind=named (one published method).",
    "ResearchProblem": "A problem a paper says it addresses.",
    "Dataset": "A benchmark or corpus used or introduced.",
    "Metric": "An evaluation criterion.",
    "Architecture": "A design pattern (hybrid, hypergraph, agentic ...).",
    "Limitation": "A weakness of prior approaches as stated in an abstract.",
    "ResearchDirection": "A direction a paper pursues or names as future work.",
    "ApplicationDomain": "A field the work is applied to (medicine, law ...).",
    "Concept": "A cross-paper structural idea.",
}

RELATION_TYPES = {
    # paper -> paper
    "CITES": ("Paper", "Paper", "OpenAlex lists the target among the source's references (both in corpus)."),
    "BUILDS_ON": ("Paper|Method", "Paper|Method", "Source's text names the target's method with an extension cue (extend, based on, inspired by)."),
    "COMPARES_AGAINST": ("Paper|Method", "Paper|Method", "Source's text names the target's method as a baseline or comparison."),
    "IMPROVES_ON": ("Method", "Method", "Source method is reported to outperform / improve the target in the same sentence."),
    "SHARES_REFERENCES": ("Paper", "Paper", "Two papers cite at least N of the same works (bibliographic coupling)."),
    # paper -> entity
    "AUTHORED_BY": ("Paper", "Author", "OpenAlex authorship."),
    "INTRODUCES_METHOD": ("Paper", "Method", "Title names the method and the paper presents it (title prefix or intro cue)."),
    "USES_METHOD": ("Paper", "Method", "Paper uses or builds on a method / paradigm."),
    "EVALUATES_METHOD": ("Paper", "Method", "Benchmark, survey or empirical-study paper that evaluates the method."),
    "MENTIONS_METHOD": ("Paper", "Method", "Abstract names a named method (often as a baseline)."),
    "INTRODUCES_DATASET": ("Paper", "Dataset", "Benchmark paper names the dataset in its title."),
    "ADDRESSES_PROBLEM": ("Paper", "ResearchProblem", "Problem alias present; high confidence when in title or a problem-cue sentence."),
    "EVALUATES_ON": ("Paper", "Dataset", "Dataset alias in an evaluation sentence."),
    "USES_METRIC": ("Paper", "Metric", "Metric alias present."),
    "IMPLEMENTS_ARCHITECTURE": ("Paper", "Architecture", "Architecture alias present."),
    "MOTIVATED_BY_LIMITATION": ("Paper", "Limitation", "Alias occurs in a sentence that also carries a limitation cue (however, struggle, costly ...)."),
    "PURSUES_DIRECTION": ("Paper", "ResearchDirection", "Alias in title, or in a sentence with an intro/future cue."),
    "APPLIED_IN_DOMAIN": ("Paper", "ApplicationDomain", "Domain alias present."),
    "RELATES_TO_CONCEPT": ("Paper", "Concept", "Concept alias present."),
    # entity -> entity (derived)
    "INSTANCE_OF": ("Method", "Method", "A named method's introducing paper uses a paradigm."),
    "TARGETS_PROBLEM": ("Method", "ResearchProblem", "Aggregated from the introducing paper."),
    "MOTIVATED_BY": ("Method", "Limitation", "Aggregated from the introducing paper."),
    "USES_ARCHITECTURE": ("Method", "Architecture", "Aggregated from the introducing paper."),
    "EVALUATED_ON": ("Method", "Dataset", "Aggregated from the introducing paper."),
    "MOTIVATES": ("Limitation", "ResearchDirection", "Co-occurs in one paper (corpus-derived) or a curated prior."),
}

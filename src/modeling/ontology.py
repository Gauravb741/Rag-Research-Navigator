"""Human-auditable ontology and controlled vocabulary for the project.

This is intentionally a small, hand-authored mapping layer rather than an
automatic knowledge-graph-construction package. The rules are inspectable and
relationship evidence is retained in the serialized knowledge state.
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


TERMS: tuple[TermSpec, ...] = (
    TermSpec("method_standard_rag", "Method", "Standard RAG", "Retrieve passages and condition generation on them.", ("retrieval-augmented generation", "retrieval augmented generation", "standard rag")),
    TermSpec("method_dpr", "Method", "DPR", "Dense Passage Retrieval for open-domain question answering.", ("dense passage retrieval", "dpr")),
    TermSpec("method_graphrag", "Method", "GraphRAG", "Graph-based indexing and retrieval for retrieval-augmented generation.", ("graphrag", "graph rag", "graph retrieval-augmented generation", "graph retrieval augmented generation")),
    TermSpec("method_g_retriever", "Method", "G-Retriever", "Retrieval-augmented generation over textual graphs using a retrieved subgraph.", ("g-retriever", "g retriever")),
    TermSpec("method_grag", "Method", "GRAG", "Graph Retrieval-Augmented Generation for networked textual documents.", ("grag", "graph retrieval-augmented generation")),
    TermSpec("method_kg2rag", "Method", "KG²RAG", "Knowledge-graph-guided chunk expansion and organization for RAG.", ("kg²rag", "kg2rag", "knowledge graph-guided retrieval augmented generation")),
    TermSpec("method_hipporag", "Method", "HippoRAG", "Graph-inspired long-term memory and retrieval for language models.", ("hipporag", "hippo rag")),
    TermSpec("method_raptor", "Method", "RAPTOR", "Recursive abstractive tree-organized retrieval.", ("raptor", "recursive abstractive processing")),
    TermSpec("method_gnn_rag", "Method", "GNN-RAG", "Graph neural retrieval for language-model reasoning over knowledge graphs.", ("gnn-rag", "gnn rag", "graph neural retrieval")),
    TermSpec("method_sg_rag", "Method", "SG-RAG", "Subgraph retrieval for multi-hop question answering.", ("sg-rag", "subgraph retrieval augmented generation")),
    TermSpec("method_multihop_rag", "Method", "MultiHop-RAG", "Benchmark and methods for multi-hop retrieval-augmented generation.", ("multihop-rag", "multi-hop-rag", "multi hop rag")),
    TermSpec("method_structugraphrag", "Method", "StructuGraphRAG", "Document-structure-informed graph RAG.", ("structugraphrag", "structured document-informed knowledge graphs")),
    TermSpec("method_document_graphrag", "Method", "Document GraphRAG", "Document-oriented knowledge-graph-enhanced RAG.", ("document graphrag", "document graph rag")),
    TermSpec("method_hypergraphrag", "Method", "HyperGraphRAG", "Hypergraph-structured knowledge representation for RAG.", ("hypergraphrag", "hypergraph rag")),
    TermSpec("method_active_rag", "Method", "Active RAG", "Retrieval triggered during generation based on predicted need or uncertainty.", ("active retrieval augmented generation", "active rag")),
    TermSpec("method_self_rag", "Method", "Self-RAG", "Adaptive retrieval, generation, and self-critique.", ("self-rag", "self rag")),
    TermSpec("method_graph_prompting", "Method", "Knowledge Graph Prompting", "Uses graph structure to prompt multi-document question answering.", ("knowledge graph prompting", "graph prompting")),
    TermSpec("method_cg_rag", "Method", "CG-RAG", "Citation-graph retrieval-augmented language-model research question answering.", ("cg-rag", "citation graph retrieval-augmented")),
    TermSpec("method_global_discovery", "Method", "Global Discovery", "Global graph-RAG for query-focused synthesis across PDF papers.", ("global discovery", "global graph-rag")),
    TermSpec("problem_multihop_reasoning", "ResearchProblem", "Multi-hop reasoning", "Answering requires chaining evidence across multiple relations or documents.", ("multi-hop reasoning", "multi hop reasoning", "multi-hop question answering", "multi hop question answering")),
    TermSpec("problem_evidence_fragmentation", "ResearchProblem", "Evidence fragmentation", "Relevant evidence is distributed across chunks, documents, or graph neighborhoods.", ("evidence fragmentation", "fragmented evidence", "dispersed evidence", "distributed evidence", "evidence distributed", "evidence across", "information silo")),
    TermSpec("problem_hallucination", "ResearchProblem", "Hallucination and factuality", "Generated claims may be unsupported or factually incorrect.", ("hallucination", "factuality", "fact consistency", "faithfulness")),
    TermSpec("problem_long_context", "ResearchProblem", "Long-context document understanding", "Reasoning over long or document-intensive contexts without losing relevant structure.", ("long-context", "long context", "long document", "long papers", "long documents", "full papers", "document-intensive")),
    TermSpec("problem_graph_qa", "ResearchProblem", "Knowledge-graph question answering", "Answering questions by querying or reasoning over entity-relation graphs.", ("knowledge graph question answering", "graph question answering", "graphqa")),
    TermSpec("problem_citation_research_qa", "ResearchProblem", "Citation-aware research question answering", "Answering research questions using citation and scholarly-relationship structure.", ("research question answering", "citation graph", "citation-aware", "academic literature")),
    TermSpec("problem_noisy_retrieval", "ResearchProblem", "Noisy or irrelevant retrieval", "Retrieved context can be irrelevant, redundant, or misleading.", ("noisy retrieval", "irrelevant retrieval", "retrieval noise", "inter-context conflict", "inter-context conflicts", "conflicting evidence", "conflicting documents")),
    TermSpec("problem_provenance", "ResearchProblem", "Evidence provenance and explainability", "Users need to inspect why a claim was produced and which source supports it.", ("provenance", "explainability", "transparent", "traceability")),
    TermSpec("dataset_hotpotqa", "Dataset", "HotpotQA", "Multi-hop question answering benchmark over linked passages.", ("hotpotqa",)),
    TermSpec("dataset_2wikimultihopqa", "Dataset", "2WikiMultiHopQA", "Multi-hop question answering benchmark based on Wikipedia.", ("2wikimultihopqa", "2wiki multi-hop")),
    TermSpec("dataset_musique", "Dataset", "MuSiQue", "Multi-hop compositional question answering benchmark.", ("musique",)),
    TermSpec("dataset_webqsp", "Dataset", "WebQSP", "Question answering over Freebase knowledge graphs.", ("webqsp",)),
    TermSpec("dataset_metaqa", "Dataset", "MetaQA", "Multi-hop question answering over a movie knowledge graph.", ("metaqa",)),
    TermSpec("dataset_graphqa", "Dataset", "GraphQA", "Question answering over textual or structured graphs.", ("graphqa",)),
    TermSpec("dataset_multihop_rag", "Dataset", "MultiHop-RAG benchmark", "Benchmark for retrieval-augmented generation on multi-hop queries.", ("multihop-rag", "multi-hop rag benchmark")),
    TermSpec("dataset_graphrag_bench", "Dataset", "GraphRAG-Bench", "Benchmark for domain-specific graph-RAG reasoning.", ("graphrag-bench",)),
    TermSpec("dataset_magic", "Dataset", "MAGIC", "Multi-hop graph-based benchmark for inter-context conflicts in RAG.", ("magic", "inter-context conflicts")),
    TermSpec("dataset_qasper", "Dataset", "QASPER", "Question answering over scientific papers.", ("qasper",)),
    TermSpec("dataset_nq", "Dataset", "Natural Questions", "Open-domain question answering benchmark.", ("natural questions", "nq dataset")),
    TermSpec("dataset_triviaqa", "Dataset", "TriviaQA", "Open-domain question answering benchmark.", ("triviaqa",)),
    TermSpec("dataset_pubmedqa", "Dataset", "PubMedQA", "Biomedical question answering benchmark.", ("pubmedqa",)),
    TermSpec("metric_exact_match", "Metric", "Exact Match", "Exact string match of an answer.", ("exact match", "em score")),
    TermSpec("metric_f1", "Metric", "F1", "Token-level or span-level F1 score.", ("f1", "f1-score", "f1 score")),
    TermSpec("metric_accuracy", "Metric", "Accuracy", "Fraction of correct predictions.", ("accuracy", "acc")),
    TermSpec("metric_recall_at_k", "Metric", "Recall@k", "Recall of relevant evidence in the top-k retrieval results.", ("recall@k", "recall at k", "answer recall")),
    TermSpec("metric_hit_at_1", "Metric", "Hit@1", "Whether the top-ranked result contains the answer or relevant evidence.", ("hit@1", "hit at 1")),
    TermSpec("metric_rouge", "Metric", "ROUGE", "N-gram overlap metric for generated summaries or answers.", ("rouge",)),
    TermSpec("metric_bertscore", "Metric", "BERTScore", "Embedding-based semantic similarity metric.", ("bertscore", "bert score")),
    TermSpec("metric_faithfulness", "Metric", "Faithfulness", "Consistency of a generated response with its evidence.", ("faithfulness", "groundedness", "answer faithfulness")),
    TermSpec("metric_context_relevance", "Metric", "Context relevance", "Relevance of retrieved context to the input.", ("context relevance", "context relevancy")),
    TermSpec("arch_vector_rag", "Architecture", "Vector RAG", "Flat text chunk retrieval followed by context-conditioned generation.", ("vector retrieval", "vector database", "dense retrieval")),
    TermSpec("arch_graph_rag", "Architecture", "Graph RAG pipeline", "Graph indexing, graph-guided retrieval, and graph-enhanced generation.", ("graph-based retrieval", "graph-guided retrieval", "graph-enhanced generation")),
    TermSpec("arch_subgraph_retrieval", "Architecture", "Subgraph retrieval", "Retrieve a connected graph neighborhood or path as evidence.", ("subgraph retrieval", "retrieved subgraph", "subgraph construction")),
    TermSpec("arch_hierarchical_retrieval", "Architecture", "Hierarchical retrieval", "Retrieve information at multiple levels of abstraction.", ("hierarchical retrieval", "hierarchical graph", "tree-organized retrieval", "community summary")),
    TermSpec("arch_hybrid_graph_vector", "Architecture", "Hybrid graph and vector retrieval", "Combine semantic chunk retrieval with graph structure or paths.", ("hybrid graph", "graph and vector", "dual pathway", "dual knowledge")),
    TermSpec("limitation_graph_construction_cost", "Limitation", "Graph construction cost", "Extracting and maintaining entities, relations, and graph structures is expensive.", ("construction cost", "graph construction", "extraction cost", "knowledge graph construction")),
    TermSpec("limitation_noisy_extraction", "Limitation", "Noisy entity or relation extraction", "Incorrect or incomplete extraction can create misleading graph paths.", ("noisy extraction", "entity extraction errors", "relation extraction", "extraction noise")),
    TermSpec("limitation_latency", "Limitation", "Retrieval and inference latency", "Graph expansion or multi-stage reasoning may add runtime cost.", ("latency", "inference cost", "computational cost", "runtime overhead")),
    TermSpec("limitation_scalability", "Limitation", "Graph scalability", "Large graphs and large corpora make indexing or traversal difficult.", ("scalability", "scale", "large-scale graph", "large graph")),
    TermSpec("limitation_dynamic_updates", "Limitation", "Dynamic graph updates", "Adding or removing papers may require expensive graph or summary updates.", ("dynamic dataset", "dynamic datasets", "new documents", "new data", "updates", "updating", "update cost", "incremental")),
    TermSpec("limitation_domain_specificity", "Limitation", "Domain or benchmark specificity", "Results may not generalize beyond the evaluated domain or benchmark.", ("domain-specific", "domain specific", "limited generalizability", "generalizability")),
    TermSpec("limitation_missing_provenance", "Limitation", "Missing evidence provenance", "A response or path may not expose enough source evidence for verification.", ("source attribution", "citation accuracy", "provenance", "traceability")),
    TermSpec("limitation_context_loss", "Limitation", "Context loss from chunking or compression", "Chunking, summarization, or graph textualization can lose important context.", ("context loss", "semantic fragmentation", "lost in the middle", "information loss")),
    TermSpec("direction_efficient_graph_retrieval", "ResearchDirection", "Efficient graph retrieval", "Reduce graph retrieval cost while retaining useful paths.", ("efficient graph retrieval", "adaptive pruning", "efficient retrieval", "pruning")),
    TermSpec("direction_hybrid_retrieval", "ResearchDirection", "Hybrid graph-vector retrieval", "Combine graph structure with semantic or lexical retrieval.", ("hybrid retrieval", "hybrid graph", "graph and vector")),
    TermSpec("direction_dynamic_knowledge", "ResearchDirection", "Dynamic and updateable knowledge", "Maintain structured knowledge as papers or evidence change.", ("dynamic knowledge", "incremental update", "continual update", "knowledge update", "updateable knowledge", "new documents")),
    TermSpec("direction_citation_aware_research", "ResearchDirection", "Citation-aware research synthesis", "Use scholarly citations and research networks to guide evidence synthesis.", ("citation-aware", "citation graph", "research synthesis", "research question")),
    TermSpec("direction_robust_graph_evaluation", "ResearchDirection", "Robust GraphRAG evaluation", "Evaluate path quality, evidence grounding, robustness, and generalization.", ("graphrag-bench", "benchmark", "robustness", "evaluation")),
    TermSpec("direction_multihop_reasoning", "ResearchDirection", "Improved multi-hop reasoning", "Improve decomposition, traversal, and synthesis over multiple evidence hops.", ("multi-hop reasoning", "reasoning chain", "iterative reasoning", "multi-hop")),
    TermSpec("concept_entity_relation_graph", "Concept", "Entity-relation graph", "Nodes represent entities and edges represent typed relations.", ("entity-relation graph", "entity-relation", "entities and relationships", "entity relation", "knowledge graph")),
    TermSpec("concept_evidence_path", "Concept", "Evidence path", "A connected sequence of graph relations supporting a claim.", ("evidence path", "evidence paths", "source path", "source paths", "reasoning path", "knowledge path", "path retrieval")),
    TermSpec("concept_citation_network", "Concept", "Citation network", "A network formed by papers citing other papers.", ("citation network", "citation graph", "mutual referencing")),
    TermSpec("concept_graph_indexing", "Concept", "Graph indexing", "Converting source documents into a graph representation for retrieval.", ("graph indexing", "graph-indexed", "graph indexed", "indexing", "knowledge graph construction")),
    TermSpec("concept_query_decomposition", "Concept", "Query decomposition", "Break a complex question into subquestions or retrieval steps.", ("query decomposition", "question decomposition", "decompose")),
    TermSpec("concept_provenance", "Concept", "Provenance", "Source-location metadata attached to a claim or relation.", ("provenance", "source attribution", "evidence source")),
    TermSpec("concept_multi_hop", "Concept", "Multi-hop reasoning", "Reasoning over multiple linked facts or documents.", ("multi-hop", "multi hop")),
    TermSpec("concept_graph_community", "Concept", "Graph communities", "Clusters or communities used to summarize or navigate a graph.", ("community detection", "community summary", "graph community")),
)


def iter_terms(entity_type: str | None = None) -> Iterable[TermSpec]:
    for term in TERMS:
        if entity_type is None or term.entity_type == entity_type:
            yield term

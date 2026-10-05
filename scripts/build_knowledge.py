#!/usr/bin/env python3
"""Build the inspectable knowledge state from the selected paper corpus.

Every edge is produced by a named, hand-written rule (see RULES) and carries a
confidence level, a quote and its location. No extraction library is used; the
vocabulary and cue lists live in src/modeling/ontology.py.

Passes:  A discover method/dataset names -> B paper-level edges -> C textual
dependencies between methods -> D aggregate to method level -> E limitation ->
direction -> F citation structure -> G scores.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.matching import alias_regex, cue_regex, find_alias, sentences  # noqa: E402
from src.modeling.ontology import (  # noqa: E402
    CUES, CURATED_LIMITATION_DIRECTIONS, ENTITY_TYPES, GENERIC_PARADIGM_ALIASES, PAPER_KIND_RULES, RELATION_TYPES, TERMS, TermSpec,
)

PAPERS_PATH = ROOT / "data" / "processed" / "papers.json"
CANDIDATES_PATH = ROOT / "data" / "raw" / "all_candidates.json"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"

SHARED_REFERENCE_THRESHOLD = 2      # papers sharing >= N references get a SHARES_REFERENCES edge
EXTERNAL_FOUNDATION_MIN_CITERS = 3  # outside-corpus work cited by >= N corpus papers
PRIOR_MARKER = r"existing|current|prior|conventional|traditional|previous|recent|most|often|standard|state-of-the-art|many"

RULES = {
    "title_prefix_name": "Title starts with 'Name:' (<=2 tokens, internal capital/digit) -> paper introduces that method or dataset.",
    "title_starts_with_alias": "Title begins with a named-method alias (e.g. 'Dense Passage Retrieval for ...') -> INTRODUCES_METHOD.",
    "title_alias_with_intro_cue": "Named-method alias in title and an intro cue (propose/introduce/present...) in the abstract -> INTRODUCES_METHOD.",
    "named_method_in_title": "Named method in title of a non-introducing paper -> USES_METHOD (EVALUATES_METHOD for survey/benchmark/empirical papers).",
    "named_method_in_abstract": "Named method only in the abstract -> MENTIONS_METHOD (medium if the sentence compares, else low).",
    "paradigm_mention": "Paradigm alias (GraphRAG, standard RAG) -> USES_METHOD.",
    "problem_alias": "Problem alias: high in title, medium in a sentence with a problem cue, else low.",
    "dataset_eval_sentence": "Dataset alias: high in an evaluation sentence, else medium.",
    "metric_alias": "Metric alias: high in an evaluation sentence, else medium.",
    "architecture_alias": "Architecture alias: high in title, medium in an intro-cue sentence, else low.",
    "limitation_cue_sentence": "Limitation alias in a sentence that also has a limitation cue; high if the sentence refers to prior/existing approaches.",
    "direction_alias": "Direction alias: high in title or a future-work sentence, medium in an intro-cue sentence; otherwise no edge.",
    "domain_alias": "Application-domain alias: high in title, medium if >=2 distinct aliases in abstract, else low.",
    "concept_alias": "Concept alias: medium in title, else low.",
    "textual_method_dependency": "Another method named in a sentence with improve / extend / compare cue.",
    "aggregated_from_introducing_paper": "Method-level edge copied from the paper that introduces the method.",
    "limitation_direction_cooccurrence": "A paper has both the limitation and the direction edge.",
    "curated_ontology_prior": "Hand-written limitation -> direction prior with rationale.",
    "metadata_citation": "OpenAlex referenced_works contains a corpus paper.",
    "bibliographic_coupling": "Two corpus papers share >= threshold referenced works.",
    "openalex_authorship": "OpenAlex authorship metadata.",
}
CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:80]


# ----------------------------------------------------------------- paper helpers
def classify_kind(title: str, abstract: str) -> str:
    for kind, title_rx, abstract_rx in PAPER_KIND_RULES:
        if re.search(title_rx, title, re.IGNORECASE) or re.search(abstract_rx, abstract or "", re.IGNORECASE):
            return kind
    return "method"


def title_prefix_name(title: str) -> str | None:
    """'HyperGraphRAG: ...' -> 'HyperGraphRAG'. At most two tokens, one with an internal capital or digit."""
    m = re.match(r"^\s*([^:]{2,32}?)\s*:\s+\S", title)
    if not m:
        return None
    prefix = m.group(1).strip()
    tokens = prefix.split()
    if not 1 <= len(tokens) <= 2:
        return None
    if not any(re.search(r"[A-Z0-9²].*|.*[A-Z]", t[1:]) for t in tokens):
        return None
    if re.search(r"[A-Z][a-z]+[A-Z]|[A-Z]{2,}|\d|[a-z][A-Z]", prefix) is None:
        return None
    return prefix


def first_hit(paper_title: str, sents: list[str], alias: str) -> tuple[str, str, str] | None:
    if find_alias(paper_title, alias):
        return paper_title, "title", "explicit_title_match"
    for s in sents:
        if find_alias(s, alias):
            return s, "abstract", "explicit_abstract_match"
    return None


def main() -> int:
    corpus = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    papers = corpus["papers"]
    retrieved_on = corpus.get("metadata", {}).get("retrieved_on", "unknown")
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    candidate_titles: dict[str, dict[str, Any]] = {}
    if CANDIDATES_PATH.exists():
        for item in json.loads(CANDIDATES_PATH.read_text(encoding="utf-8")).get("papers", []):
            candidate_titles[item.get("openalex_id")] = item

    paper_by_oa = {p["openalex_id"]: p for p in papers}
    entities: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    rel_index: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_rel(source, relation, target, evidence, location, evidence_type, rule, confidence,
                paper_id=None, **extra) -> dict[str, Any]:
        key = (source, relation, target)
        if key in rel_index:
            existing = rel_index[key]
            sp = existing.get("supporting_papers")
            if sp is not None and paper_id and paper_id not in sp:
                sp.append(paper_id)
            if CONF_ORDER[confidence] > CONF_ORDER[existing["confidence"]]:
                existing.update({"confidence": confidence, "evidence": evidence, "source_location": location,
                                 "evidence_type": evidence_type, "rule": rule})
            return existing
        rel = {"id": f"rel_{len(relationships) + 1:05d}", "source": source, "relation": relation, "target": target,
               "evidence": evidence, "source_location": location, "evidence_type": evidence_type,
               "rule": rule, "confidence": confidence, "source_paper": paper_id}
        rel.update(extra)
        relationships.append(rel)
        rel_index[key] = rel
        return rel

    # ------------------------------------------------------ vocabulary entities
    spec_by_id: dict[str, TermSpec] = {}
    for spec in TERMS:
        spec_by_id[spec.entity_id] = spec
        entities[spec.entity_id] = {
            "id": spec.entity_id, "type": spec.entity_type, "name": spec.name, "description": spec.description,
            "aliases": list(spec.aliases), "kind": spec.kind or None, "controlled_vocabulary": True,
            "discovered": False, "mapping_rule": "hand-authored alias match in src/modeling/ontology.py",
        }

    # ------------------------------------------------------------------ pass A
    info: dict[str, dict[str, Any]] = {}
    for paper in papers:
        pid = paper["paper_id"]
        title = paper["title"]
        abstract = paper.get("abstract") or ""
        usable = len(abstract) >= 200
        sents = sentences(abstract) if usable else []
        kind = classify_kind(title, abstract if usable else "")
        info[pid] = {"title": title, "sents": sents, "kind": kind, "usable_abstract": usable,
                     "introduces": [], "introduced_datasets": [], "intro_rule": {}}

    for paper in papers:
        pid = paper["paper_id"]
        title, sents, kind = paper["title"], info[pid]["sents"], info[pid]["kind"]
        prefix = title_prefix_name(title)
        intro_cue = any(cue_regex(CUES["intro"]).search(s) for s in sents)
        if prefix:
            named_hit = next((s for s in TERMS if s.entity_type == "Method" and s.kind == "named"
                              and any(find_alias(prefix, a) for a in s.aliases)), None)
            para_hit = next((s for s in TERMS if s.entity_type == "Method" and s.kind == "paradigm"
                             and any(find_alias(prefix, a) and len(prefix) <= len(a) + 2 for a in s.aliases)), None)
            data_hit = next((s for s in TERMS if s.entity_type == "Dataset"
                             and any(find_alias(prefix, a) for a in s.aliases)), None)
            if kind == "benchmark" or (data_hit and not named_hit):
                if data_hit:
                    ds_id = data_hit.entity_id
                else:
                    ds_id = f"dataset_{slug(prefix)}"
                    entities.setdefault(ds_id, {
                        "id": ds_id, "type": "Dataset", "name": prefix, "description": title.split(":", 1)[1].strip(),
                        "aliases": [prefix], "kind": None, "controlled_vocabulary": False, "discovered": True,
                        "mapping_rule": "title_prefix_name"})
                info[pid]["introduced_datasets"].append(ds_id)
            elif named_hit:
                info[pid]["introduces"].append(named_hit.entity_id)
                info[pid]["intro_rule"][named_hit.entity_id] = "title_prefix_name"
            elif para_hit:
                pass  # paper is *about* a paradigm; it does not introduce a named method
            elif kind not in ("survey", "empirical_study"):
                m_id = f"method_{slug(prefix)}"
                entities.setdefault(m_id, {
                    "id": m_id, "type": "Method", "name": prefix, "description": title.split(":", 1)[1].strip(),
                    "aliases": [prefix], "kind": "named", "controlled_vocabulary": False, "discovered": True,
                    "mapping_rule": "title_prefix_name"})
                info[pid]["introduces"].append(m_id)
                info[pid]["intro_rule"][m_id] = "title_prefix_name"
        elif kind == "method":
            for spec in TERMS:
                if spec.entity_type != "Method" or spec.kind != "named":
                    continue
                starts = any(alias_regex(a).match(title) for a in spec.aliases)
                if starts or (intro_cue and any(find_alias(title, a) for a in spec.aliases)):
                    info[pid]["introduces"].append(spec.entity_id)
                    info[pid]["intro_rule"][spec.entity_id] = "title_starts_with_alias" if starts else "title_alias_with_intro_cue"
                    break

    introducers: dict[str, list[str]] = defaultdict(list)
    for pid, data in info.items():
        for m in data["introduces"]:
            introducers[m].append(pid)

    # ------------------------------------------------------------------ pass B
    domain_hits: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    paper_hits: dict[str, dict[str, Any]] = {}
    for paper in papers:
        pid = paper["paper_id"]
        d = info[pid]
        title, sents, kind = d["title"], d["sents"], d["kind"]
        entities[pid] = {
            "id": pid, "type": "Paper", "name": title, "title": title, "year": paper.get("year"),
            "abstract": paper.get("abstract", ""), "doi": paper.get("doi"), "arxiv_id": paper.get("arxiv_id"),
            "openalex_id": paper.get("openalex_id"), "source_url": paper.get("source_url"),
            "citation_count": paper.get("citation_count", 0), "corpus_index": paper.get("corpus_index"),
            "corpus_role": paper.get("corpus_role"), "paper_kind": kind,
            "usable_abstract": d["usable_abstract"], "reference_count_in_openalex": len(paper.get("referenced_work_ids", [])),
        }
        for author in paper.get("authors", []):
            if not author.get("name"):
                continue
            aid = "author_" + slug(author["name"])
            entities.setdefault(aid, {"id": aid, "type": "Author", "name": author["name"], "orcid": author.get("orcid"),
                                      "mapping_rule": "OpenAlex authorship metadata"})
            add_rel(pid, "AUTHORED_BY", aid, f"OpenAlex authorship metadata lists {author['name']}.",
                    "metadata.authorships", "direct_metadata", "openalex_authorship", "high", pid)

        for ds_id in d["introduced_datasets"]:
            add_rel(pid, "INTRODUCES_DATASET", ds_id, title, "title", "explicit_title_match", "title_prefix_name", "high", pid,
                    matched_alias=entities[ds_id]["name"])
        for m_id in d["introduces"]:
            add_rel(pid, "INTRODUCES_METHOD", m_id, title, "title", "explicit_title_match", d["intro_rule"][m_id], "high", pid,
                    matched_alias=entities[m_id]["name"])

        for spec in list(entities.values()):
            if spec["type"] in ("Paper", "Author") or "aliases" not in spec:
                continue
            etype, eid = spec["type"], spec["id"]
            if eid in d["introduces"] or eid in d["introduced_datasets"]:
                continue
            hits = []  # (text, location, evidence_type, alias)
            for alias in spec["aliases"]:
                h = first_hit(title, sents, alias)
                if h:
                    hits.append((*h, alias))
            sent_hits = []
            for s in sents:
                for alias in spec["aliases"]:
                    if find_alias(s, alias):
                        sent_hits.append((s, alias))
                        break
            title_alias = next((h[3] for h in hits if h[1] == "title"), None)
            if not hits:
                continue
            best = next((h for h in hits if h[1] == "title"), hits[0])

            def emit(rel, conf, rule, hit=best, **extra):
                add_rel(pid, rel, eid, hit[0], hit[1], hit[2], rule, conf, pid, matched_alias=hit[3], **extra)

            if etype == "Method":
                if spec["kind"] == "paradigm":
                    emit("USES_METHOD", "high" if title_alias else "medium", "paradigm_mention")
                elif title_alias:
                    if kind in ("survey", "benchmark", "empirical_study"):
                        emit("EVALUATES_METHOD", "medium", "named_method_in_title")
                    else:
                        emit("USES_METHOD", "medium", "named_method_in_title")
                else:
                    comp = next((sh for sh in sent_hits if cue_regex(CUES["compare"]).search(sh[0]) or cue_regex(CUES["improve"]).search(sh[0])), None)
                    hit = (comp[0], "abstract", "explicit_abstract_match", comp[1]) if comp else best
                    emit("MENTIONS_METHOD", "medium" if comp else "low", "named_method_in_abstract", hit)
            elif etype == "ResearchProblem":
                cue = next((sh for sh in sent_hits if cue_regex(CUES["problem"]).search(sh[0])), None)
                if title_alias:
                    emit("ADDRESSES_PROBLEM", "high", "problem_alias")
                elif cue:
                    emit("ADDRESSES_PROBLEM", "medium", "problem_alias", (cue[0], "abstract", "explicit_abstract_match", cue[1]))
                else:
                    emit("ADDRESSES_PROBLEM", "low", "problem_alias")
            elif etype == "Dataset":
                cue = next((sh for sh in sent_hits if cue_regex(CUES["evaluation"]).search(sh[0])), None)
                emit("EVALUATES_ON", "high" if cue else "medium", "dataset_eval_sentence",
                     (cue[0], "abstract", "explicit_abstract_match", cue[1]) if cue else best)
            elif etype == "Metric":
                cue = next((sh for sh in sent_hits if cue_regex(CUES["evaluation"]).search(sh[0])), None)
                emit("USES_METRIC", "high" if cue else "medium", "metric_alias",
                     (cue[0], "abstract", "explicit_abstract_match", cue[1]) if cue else best)
            elif etype == "Architecture":
                cue = next((sh for sh in sent_hits if cue_regex(CUES["intro"]).search(sh[0])), None)
                if title_alias:
                    emit("IMPLEMENTS_ARCHITECTURE", "high", "architecture_alias")
                elif cue:
                    emit("IMPLEMENTS_ARCHITECTURE", "medium", "architecture_alias", (cue[0], "abstract", "explicit_abstract_match", cue[1]))
                else:
                    emit("IMPLEMENTS_ARCHITECTURE", "low", "architecture_alias")
            elif etype == "Limitation":
                cue = next((sh for sh in sent_hits if cue_regex(CUES["limitation"]).search(sh[0]) and not cue_regex(CUES["improve"]).search(sh[0]) and not cue_regex(CUES["negated_limitation"]).search(sh[0])), None)
                if cue:
                    prior = cue_regex(PRIOR_MARKER).search(cue[0])
                    emit("MOTIVATED_BY_LIMITATION", "high" if prior else "medium", "limitation_cue_sentence",
                         (cue[0], "abstract", "explicit_abstract_match", cue[1]))
            elif etype == "ResearchDirection":
                fut = next((sh for sh in sent_hits if cue_regex(CUES["future"]).search(sh[0])), None)
                intro = next((sh for sh in sent_hits if cue_regex(CUES["intro"]).search(sh[0])), None)
                if fut:
                    emit("PURSUES_DIRECTION", "high", "direction_alias", (fut[0], "abstract", "explicit_abstract_match", fut[1]), qualifier="future_work")
                elif title_alias:
                    emit("PURSUES_DIRECTION", "high", "direction_alias", qualifier="pursued")
                elif intro:
                    emit("PURSUES_DIRECTION", "medium", "direction_alias", (intro[0], "abstract", "explicit_abstract_match", intro[1]), qualifier="pursued")
            elif etype == "ApplicationDomain":
                distinct = {sh[1] for sh in sent_hits}
                domain_hits[pid][eid] = distinct
                emit("APPLIED_IN_DOMAIN", "high" if title_alias else ("medium" if len(distinct) >= 2 else "low"), "domain_alias")
            elif etype == "Concept":
                emit("RELATES_TO_CONCEPT", "medium" if title_alias else "low", "concept_alias")

    # ------------------------------------------------------------------ pass C
    method_ids = [e["id"] for e in entities.values() if e["type"] == "Method"]
    paper_text_mentions: dict[str, list[tuple[str, str, str]]] = defaultdict(list)  # pid -> (method, relation, sentence)
    for paper in papers:
        pid = paper["paper_id"]
        d = info[pid]
        own = set(d["introduces"])
        own_names = [a for o in own for a in entities[o]["aliases"]]
        own_paren = re.compile(r"\([^)]*\)")
        for s_raw in d["sents"]:
            s = own_paren.sub(" ", s_raw)
            for a in sorted(own_names, key=len, reverse=True):
                s = alias_regex(a).sub(" ", s)
            # the paper's own long-form name (title tail) must not count as a mention either
            for mid in method_ids:
                if mid in own:
                    continue
                m = entities[mid]
                paradigm = m.get("kind") == "paradigm"
                usable = [a for a in m["aliases"] if not (paradigm and a in GENERIC_PARADIGM_ALIASES and mid == "method_standard_rag")]
                if not any(find_alias(s, a) for a in usable):
                    continue
                if cue_regex(CUES["improve"]).search(s):
                    rel = "IMPROVES_ON"
                elif not paradigm and cue_regex(CUES["extend"]).search(s):
                    rel = "BUILDS_ON"
                elif cue_regex(CUES["compare"]).search(s):
                    rel = "COMPARES_AGAINST"
                else:
                    continue
                paper_text_mentions[pid].append((mid, rel, s_raw))

    cites_pairs = set()
    for paper in papers:
        for ref in paper.get("referenced_work_ids", []):
            tgt = paper_by_oa.get(ref)
            if tgt and tgt["paper_id"] != paper["paper_id"]:
                cites_pairs.add((paper["paper_id"], tgt["paper_id"]))

    for pid, mentions in paper_text_mentions.items():
        for mid, rel, sentence in mentions:
            for q in introducers.get(mid, []):
                if q == pid:
                    continue
                p_rel = "BUILDS_ON" if rel == "BUILDS_ON" else "COMPARES_AGAINST"
                conf = "high" if (pid, q) in cites_pairs else "medium"
                add_rel(pid, p_rel, q, sentence, "abstract", "rule_derived_text", "textual_method_dependency", conf, pid,
                        outperforms=(rel == "IMPROVES_ON"), via_method=mid)
            for own_m in info[pid]["introduces"]:
                add_rel(own_m, rel, mid, sentence, "abstract", "rule_derived_text", "textual_method_dependency", "medium", pid,
                        supporting_papers=[pid])

    # ------------------------------------------------------------------ pass D
    paper_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in relationships:
        if rel["source"] in info:
            paper_edges[rel["source"]].append(rel)
    agg_map = {"ADDRESSES_PROBLEM": "TARGETS_PROBLEM", "MOTIVATED_BY_LIMITATION": "MOTIVATED_BY",
               "IMPLEMENTS_ARCHITECTURE": "USES_ARCHITECTURE", "EVALUATES_ON": "EVALUATED_ON"}
    for pid, d in info.items():
        for mid in d["introduces"]:
            for edge in paper_edges[pid]:
                tgt = edge["target"]
                if edge["relation"] in agg_map and edge["confidence"] != "low":
                    add_rel(mid, agg_map[edge["relation"]], tgt, edge["evidence"], edge["source_location"],
                            "aggregated_from_paper", "aggregated_from_introducing_paper", edge["confidence"], pid,
                            supporting_papers=[pid], derived_from=[edge["id"]])
                elif (edge["relation"] == "USES_METHOD" and entities[tgt].get("kind") == "paradigm"
                      and not (tgt == "method_standard_rag" and str(edge.get("matched_alias", "")).lower() in GENERIC_PARADIGM_ALIASES)):
                    add_rel(mid, "INSTANCE_OF", tgt, edge["evidence"], edge["source_location"], "aggregated_from_paper",
                            "aggregated_from_introducing_paper", edge["confidence"], pid, supporting_papers=[pid],
                            derived_from=[edge["id"]])

    # ------------------------------------------------------------------ pass E
    for pid, edges in paper_edges.items():
        lims = [e for e in edges if e["relation"] == "MOTIVATED_BY_LIMITATION"]
        dirs = [e for e in edges if e["relation"] == "PURSUES_DIRECTION" and e["confidence"] != "low"]
        for l in lims:
            for dr in dirs:
                add_rel(l["target"], "MOTIVATES", dr["target"],
                        f"{info[pid]['title']} — limitation: {l['evidence']} || direction: {dr['evidence']}",
                        "abstract", "corpus_cooccurrence", "limitation_direction_cooccurrence", "medium", pid,
                        supporting_papers=[pid], derived_from=[l["id"], dr["id"]])
    for lim, dire, why in CURATED_LIMITATION_DIRECTIONS:
        add_rel(lim, "MOTIVATES", dire, why, "ontology", "curated_ontology_prior", "curated_ontology_prior", "medium",
                None, supporting_papers=[])

    # ------------------------------------------------------------------ pass F
    for src, tgt in sorted(cites_pairs):
        add_rel(src, "CITES", tgt, f"OpenAlex lists '{info[tgt]['title']}' among the references of '{info[src]['title']}'.",
                "metadata.referenced_works", "citation_metadata", "metadata_citation", "high", src)
    refs = {p["paper_id"]: set(p.get("referenced_work_ids", [])) for p in papers}
    ids = [p["paper_id"] for p in papers]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            shared = refs[a] & refs[b]
            if len(shared) >= SHARED_REFERENCE_THRESHOLD:
                add_rel(a, "SHARES_REFERENCES", b,
                        f"Both papers cite {len(shared)} of the same works (e.g. {', '.join(sorted(shared)[:3])}).",
                        "metadata.referenced_works", "bibliographic_coupling", "bibliographic_coupling", "high", a,
                        shared_count=len(shared))

    cited_by: Counter = Counter()
    for p in papers:
        for r in refs[p["paper_id"]]:
            if r not in paper_by_oa:
                cited_by[r] += 1
    external = []
    for oa_id, n in sorted(cited_by.items(), key=lambda kv: (-kv[1], kv[0])):
        if n < EXTERNAL_FOUNDATION_MIN_CITERS:
            break
        cand = candidate_titles.get(oa_id) or {}
        external.append({"openalex_id": oa_id, "title": cand.get("title"), "year": cand.get("year"),
                         "cited_by_corpus_papers": n, "source_url": oa_id,
                         "citing_papers": sorted(p["paper_id"] for p in papers if oa_id in refs[p["paper_id"]]),
                         "title_resolved": bool(cand.get("title"))})

    # ------------------------------------------------------------------ pass G
    inbound = defaultdict(Counter)
    for rel in relationships:
        if rel["relation"] in ("CITES", "BUILDS_ON", "COMPARES_AGAINST") and rel["target"] in info and rel["source"] in info:
            inbound[rel["target"]][rel["relation"]] += 1
    for pid in info:
        ib = inbound[pid]
        entities[pid]["inbound_links"] = dict(ib)
        entities[pid]["foundation_score"] = round(ib["CITES"] + ib["BUILDS_ON"] + 0.5 * ib["COMPARES_AGAINST"], 2)
    df = Counter()
    for rel in relationships:
        if rel["source"] in info and rel["target"] in entities and entities[rel["target"]].get("aliases") is not None:
            df[rel["target"]] += 1
    for eid, e in entities.items():
        if e.get("aliases") is not None:
            e["paper_count"] = df[eid]
            e["in_corpus"] = df[eid] > 0

    entity_type_counts = Counter(e["type"] for e in entities.values())
    relation_counts = Counter(r["relation"] for r in relationships)
    conf_counts = Counter(r["confidence"] for r in relationships)
    rule_counts = Counter(r["rule"] for r in relationships)
    ee = sum(1 for r in relationships if r["source"] not in info and entities[r["source"]]["type"] != "Author"
             and entities[r["target"]]["type"] not in ("Author", "Paper"))
    state = {
        "metadata": {
            "schema_version": "2.0", "retrieved_on": retrieved_on, "source_corpus": "data/processed/papers.json",
            "ontology_source": "src/modeling/ontology.py", "builder": "scripts/build_knowledge.py",
            "construction_method": "hand-written alias/cue rules; every edge carries rule + confidence + quote",
            "human_auditable": True, "automatic_kg_library_used": False, "corpus_size": len(papers),
            "entity_type_counts": dict(entity_type_counts), "relationship_type_counts": dict(relation_counts),
            "confidence_counts": dict(conf_counts), "entity_to_entity_edge_count": ee,
            "papers_without_openalex_references": sum(1 for p in papers if not p.get("referenced_work_ids")),
            "papers_without_usable_abstract": sum(1 for d in info.values() if not d["usable_abstract"]),
            "shared_reference_threshold": SHARED_REFERENCE_THRESHOLD,
        },
        "schema": {"entity_types": ENTITY_TYPES,
                   "relation_types": {k: {"source": v[0], "target": v[1], "meaning": v[2]} for k, v in RELATION_TYPES.items()}},
        "rules": RULES,
        "entities": list(entities.values()),
        "relationships": relationships,
        "external_foundations": external,
        "papers": [paper_by_oa[p["openalex_id"]] for p in papers],
    }
    (KNOWLEDGE_DIR / "knowledge_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    (KNOWLEDGE_DIR / "entities.json").write_text(json.dumps(state["entities"], indent=2, ensure_ascii=False), encoding="utf-8")
    (KNOWLEDGE_DIR / "relationships.json").write_text(json.dumps(relationships, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "corpus_size": len(papers), "entity_count": len(entities), "relationship_count": len(relationships),
        "entity_type_counts": dict(entity_type_counts), "relationship_type_counts": dict(relation_counts),
        "confidence_counts": dict(conf_counts), "rule_counts": dict(rule_counts), "entity_to_entity_edges": ee,
        "paper_kind_counts": dict(Counter(d["kind"] for d in info.values())),
        "discovered_methods": sorted(e["name"] for e in entities.values() if e["type"] == "Method" and e.get("discovered")),
    }
    (KNOWLEDGE_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Graph reasoner: positions a new research idea inside the knowledge state.

Pipeline for a new input
  1. match_terms      idea text -> ontology entities (exact alias, stem, synonym, description overlap)
  2. score            weighted facet overlap + entity-to-entity propagation + citation expansion
  3. paths            typed multi-hop chains over entity-to-entity edges (method lineage, limitation -> direction)
  4. overlap_and_gap  which facet combinations the corpus covers, and which nobody has combined
  5. reading order    topological over citation/build-on edges, surveys and foundations first

Everything is deterministic and uses only the knowledge state; no network or model call.
All tunable weights are the constants directly below.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any

from src.modeling.matching import GENERIC, STOPWORDS, alias_regex, find_alias, stem, stems
from src.modeling.ontology import SYNONYMS
from src.retrieval.lexical import flat_retrieve

# ----------------------------------------------------------------- tunable weights
CONF_W = {"high": 1.0, "medium": 0.7, "low": 0.4}
TYPE_W = {"Method": 1.5, "ResearchProblem": 1.3, "Architecture": 1.0, "Dataset": 0.8, "Metric": 0.3,
          "Limitation": 0.7, "ResearchDirection": 0.7, "ApplicationDomain": 1.0, "Concept": 0.4}
PARADIGM_FACTOR = 0.4          # paradigm methods (GraphRAG, standard RAG) are near-ubiquitous
REL_W = {"INTRODUCES_METHOD": 1.0, "INTRODUCES_DATASET": 1.0, "USES_METHOD": 0.8, "EVALUATES_METHOD": 0.8, "MENTIONS_METHOD": 0.4}
MATCH_CONF = {"exact": 1.0, "stem": 0.85, "synonym": 0.7, "description": 0.4}
PROPAGATION_FACTOR = 0.35      # score passed through an entity-to-entity edge
CITATION_FACTOR = 0.25         # score passed to a citation / build-on neighbour
LEXICAL_WEIGHT = 2.0
FOUNDATION_TIEBREAK = 0.02
NOVELTY_CLOSE, NOVELTY_PARTIAL = 0.75, 0.40
MIN_ONTOLOGY_COVERAGE = 0.25

FACET_RELS = {"INTRODUCES_METHOD", "USES_METHOD", "EVALUATES_METHOD", "MENTIONS_METHOD", "INTRODUCES_DATASET",
              "ADDRESSES_PROBLEM", "EVALUATES_ON", "USES_METRIC", "IMPLEMENTS_ARCHITECTURE",
              "MOTIVATED_BY_LIMITATION", "PURSUES_DIRECTION", "APPLIED_IN_DOMAIN", "RELATES_TO_CONCEPT"}
PAPER_LINK_RELS = {"CITES", "BUILDS_ON", "COMPARES_AGAINST", "SHARES_REFERENCES"}
NON_VOCAB = {"Paper", "Author"}
HUB_TYPES = {"ResearchDirection", "ResearchProblem", "Architecture", "Dataset"}  # shared by many methods: never hop *through* them
DESC_TYPES = {"ResearchProblem", "Limitation", "ResearchDirection", "Architecture", "Concept", "ApplicationDomain"}
KIND_RANK = {"survey": 0, "benchmark": 3, "empirical_study": 3, "method": 2, "other": 4}


def contains_alias(text: str, alias: str) -> bool:  # kept for backward compatibility
    return find_alias(text, alias)


class GraphReasoner:
    def __init__(self, state: dict[str, Any]):
        self.state = state
        self.entities = {e["id"]: e for e in state["entities"]}
        self.papers = {i: e for i, e in self.entities.items() if e["type"] == "Paper"}
        self.relationships = state["relationships"]
        self.relationship_by_id = {r["id"]: r for r in self.relationships}
        self.outgoing: dict[str, list] = defaultdict(list)
        self.incoming: dict[str, list] = defaultdict(list)
        for r in self.relationships:
            self.outgoing[r["source"]].append(r)
            self.incoming[r["target"]].append(r)
        self.vocab = {i: e for i, e in self.entities.items() if e["type"] not in NON_VOCAB and e.get("aliases")}
        self.ee_adj: dict[str, list] = defaultdict(list)
        for r in self.relationships:
            if self._is_ee(r):
                self.ee_adj[r["source"]].append((r["target"], r, False))
                self.ee_adj[r["target"]].append((r["source"], r, True))
        df: Counter = Counter()
        for pid in self.papers:
            seen = {r["target"] for r in self.outgoing[pid] if r["relation"] in FACET_RELS}
            df.update(seen)
        self.df = df
        n = max(len(self.papers), 1)
        self.n_papers = n
        self.idf = {eid: math.log((n + 1) / (df[eid] + 1)) + 0.5 for eid in self.vocab}
        self.max_idf = math.log(n + 1) + 0.5
        self._alias_stems = {eid: [tuple(stems(a)) for a in e["aliases"] if stems(a)] for eid, e in self.vocab.items()}
        self._desc_stems = {eid: {s for s in stems(f"{e['name']} {e.get('description', '')}") if s not in GENERIC}
                            for eid, e in self.vocab.items() if e["type"] in DESC_TYPES}
        self._syn = [(re.compile(r"(?<![A-Za-z0-9])(?:" + pat + r")(?![A-Za-z0-9])", re.I), ids) for pat, ids in SYNONYMS]
        self.corpus_vocab = Counter()
        for p in self.papers.values():
            self.corpus_vocab.update(set(stems(f"{p['title']} {p.get('abstract', '')}")))

    # ------------------------------------------------------------------ helpers
    def _is_ee(self, r: dict) -> bool:
        s, t = self.entities.get(r["source"]), self.entities.get(r["target"])
        return bool(s and t and s["type"] not in NON_VOCAB and t["type"] not in NON_VOCAB)

    def _skip_edge(self, r: dict, exclude: set[str]) -> bool:
        if not exclude:
            return False
        if r["source"] in exclude or r["target"] in exclude:
            return True
        sp = r.get("supporting_papers")
        return bool(sp) and all(p in exclude for p in sp)

    def _step(self, rel: dict, reverse: bool = False) -> dict[str, Any]:
        a, b = (rel["target"], rel["source"]) if reverse else (rel["source"], rel["target"])
        return {"from": self.entities[a]["name"], "relation": rel["relation"], "to": self.entities[b]["name"],
                "evidence_id": rel["id"], "confidence": rel.get("confidence"), "reverse": reverse}

    # ------------------------------------------------------------- 1. matching
    def match_terms(self, idea: str) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}

        def record(eid: str, mtype: str, evidence: str) -> None:
            conf = MATCH_CONF[mtype]
            if eid in found:
                if conf > found[eid]["match_confidence"]:
                    found[eid].update({"match_type": mtype, "match_confidence": conf})
                if evidence not in found[eid]["matched_aliases"]:
                    found[eid]["matched_aliases"].append(evidence)
                return
            e = self.vocab[eid]
            found[eid] = {"entity_id": eid, "name": e["name"], "type": e["type"], "matched_aliases": [evidence],
                          "match_type": mtype, "match_confidence": conf, "kind": e.get("kind")}

        for eid, e in self.vocab.items():
            for alias in e["aliases"]:
                if find_alias(idea, alias):
                    record(eid, "exact", alias)
        idea_stems = stems(idea)
        joined = " " + " ".join(idea_stems) + " "
        for eid in self.vocab:
            if eid in found:
                continue
            for tup in self._alias_stems[eid]:
                if " " + " ".join(tup) + " " in joined:
                    record(eid, "stem", " ".join(tup))
                    break
        for rx, ids in self._syn:
            m = rx.search(idea)
            if m:
                for eid in ids:
                    if eid in self.vocab and eid not in found:
                        record(eid, "synonym", m.group(0))
        idea_set = set(idea_stems) - GENERIC
        for eid, dstems in self._desc_stems.items():
            if eid in found or not dstems:
                continue
            shared = idea_set & dstems
            if len(shared) >= 3 and len(shared) / len(dstems) >= 0.3:
                record(eid, "description", "+".join(sorted(shared)[:4]))
        out = []
        for m in found.values():
            e = self.vocab[m["entity_id"]]
            df = self.df[m["entity_id"]]
            idf = self.idf[m["entity_id"]] if df else self.max_idf
            w = idf * TYPE_W.get(m["type"], 0.5) * m["match_confidence"]
            if e.get("kind") == "paradigm":
                w *= PARADIGM_FACTOR
            m.update({"paper_count": df, "supported": df > 0, "weight": round(w, 4)})
            out.append(m)
        out.sort(key=lambda m: (-m["weight"], m["name"]))
        return out

    # --------------------------------------------------------------- 2. scoring
    def _score(self, matched, idea, exclude, citation_expansion, propagation):
        graph: dict[str, float] = defaultdict(float)
        contrib: dict[str, list] = defaultdict(list)
        for m in matched:
            eid = m["entity_id"]
            for r in self.incoming[eid]:
                pid = r["source"]
                if pid not in self.papers or pid in exclude or r["relation"] not in FACET_RELS:
                    continue
                val = m["weight"] * REL_W.get(r["relation"], 1.0) * CONF_W[r.get("confidence", "medium")]
                graph[pid] += val
                contrib[pid].append({"entity_id": eid, "value": val, "rel": r})
        if propagation:
            for m in matched:
                for nbr, r, rev in self.ee_adj.get(m["entity_id"], []):
                    if self._skip_edge(r, exclude) or self.entities[nbr]["type"] != "Method":
                        continue
                    for r2 in self.incoming[nbr]:
                        pid = r2["source"]
                        if r2["relation"] != "INTRODUCES_METHOD" or pid in exclude or pid not in self.papers:
                            continue
                        val = PROPAGATION_FACTOR * m["weight"] * CONF_W[r.get("confidence", "medium")]
                        graph[pid] += val
                        contrib[pid].append({"entity_id": m["entity_id"], "value": val, "rel": r2, "via": r, "via_reverse": rev})
        base = dict(graph)
        if citation_expansion and base:
            top = sorted(base.items(), key=lambda kv: -kv[1])[:8]
            for pid, score in top:
                links = [(r, r["target"]) for r in self.outgoing[pid] if r["relation"] in PAPER_LINK_RELS]
                links += [(r, r["source"]) for r in self.incoming[pid] if r["relation"] in PAPER_LINK_RELS]
                for r, other in links:
                    if other in exclude or other not in self.papers or self._skip_edge(r, exclude):
                        continue
                    factor = 0.5 if r["relation"] == "SHARES_REFERENCES" else 1.0
                    add = CITATION_FACTOR * score * factor
                    graph[other] += add
                    contrib[other].append({"entity_id": None, "value": add, "rel": r, "via_paper": pid})
        return graph, contrib

    def _lexical(self, idea: str, exclude: set[str]) -> dict[str, float]:
        pool = [p for pid, p in self.papers.items() if pid not in exclude]
        hits = flat_retrieve(idea, [{"paper_id": p["id"], "title": p["title"], "abstract": p.get("abstract", ""),
                                     "source_url": p.get("source_url")} for p in pool], limit=len(pool))
        top = max((h["score"] for h in hits), default=0) or 1.0
        return {h["paper_id"]: h["score"] / top for h in hits}

    def _paper_facets(self, pid: str, exclude_low: bool = True) -> set[str]:
        return {r["target"] for r in self.outgoing[pid]
                if r["relation"] in FACET_RELS and not (exclude_low and r.get("confidence") == "low")}

    # ------------------------------------------------------------------ 3. paths
    def _entity_chains(self, matched, exclude, limit=6):
        results = []
        for m in matched:
            start = m["entity_id"]
            queue = deque([(start, [], {start})])
            while queue:
                node, steps, seen = queue.popleft()
                if len(steps) >= 3 or (steps and (self.entities[node]["type"] in HUB_TYPES or self.entities[node].get("kind") == "paradigm")):
                    continue  # directions and paradigm hubs are terminals: hopping through them links unrelated siblings
                for nbr, r, rev in self.ee_adj.get(node, []):
                    if nbr in seen or self._skip_edge(r, exclude):
                        continue
                    new = steps + [(r, rev)]
                    if self.entities[nbr]["type"] in ("Limitation", "ResearchDirection", "Method"):
                        support = sum(math.log(1 + len(x.get("supporting_papers") or [1])) + 0.3 * CONF_W[x["confidence"]] for x, _ in new)
                        bonus = (1.0 if self.entities[nbr]["type"] in ("Limitation", "ResearchDirection") else 0) + 0.5 * len(new)
                        results.append((m["weight"] * 0.1 + support + bonus, start, nbr, new))
                    queue.append((nbr, new, seen | {nbr}))
        results.sort(key=lambda x: (-x[0], x[1], x[2]))
        paths, used = [], set()
        for score, start, end, steps in results:
            if (start, end) in used:
                continue
            used.add((start, end))
            nodes = [self.entities[start]["name"]] + [s["to"] for s in (self._step(r, rev) for r, rev in steps)]
            paths.append({"path_type": "entity_chain", "nodes": nodes, "steps": [self._step(r, rev) for r, rev in steps],
                          "evidence_ids": [r["id"] for r, _ in steps], "score": round(score, 3),
                          "supporting_papers": sorted({p for r, _ in steps for p in (r.get("supporting_papers") or [])})})
            if len(paths) >= limit:
                break
        return paths

    def _paper_paths(self, pid, exclude, limit=3):
        out = []
        for r in self.outgoing[pid]:
            if r["relation"] in ("MOTIVATED_BY_LIMITATION", "PURSUES_DIRECTION", "ADDRESSES_PROBLEM") and r.get("confidence") != "low":
                out.append({"path_type": "paper_to_entity", "nodes": [self.papers[pid]["title"], self.entities[r["target"]]["name"]],
                            "steps": [self._step(r)], "evidence_ids": [r["id"]]})
        for r in self.outgoing[pid]:
            if r["relation"] in ("BUILDS_ON", "COMPARES_AGAINST", "CITES") and r["target"] in self.papers and r["target"] not in exclude:
                for r2 in self.outgoing[r["target"]]:
                    if r2["relation"] in ("ADDRESSES_PROBLEM", "INTRODUCES_METHOD") and r2.get("confidence") != "low":
                        out.append({"path_type": "paper_to_paper_to_entity",
                                    "nodes": [self.papers[pid]["title"], self.papers[r["target"]]["title"], self.entities[r2["target"]]["name"]],
                                    "steps": [self._step(r), self._step(r2)], "evidence_ids": [r["id"], r2["id"]]})
                        break
        return out[:limit]

    # ------------------------------------------------------- 4. overlap and gap
    def _overlap_and_gap(self, matched, ranked, exclude, ontology_coverage):
        supported = [m for m in matched if m["supported"] and m["type"] not in ("Metric", "Concept")]
        facets = supported[:8]
        idea_facets: dict[str, list] = defaultdict(list)
        for m in matched:
            idea_facets[m["type"]].append({"entity_id": m["entity_id"], "name": m["name"], "weight": m["weight"],
                                           "supported_by_corpus": m["supported"], "match_type": m["match_type"]})
        unsupported = [m["name"] for m in matched if not m["supported"]]
        total_w = sum(m["weight"] for m in facets) or 1.0
        paper_sets = {m["entity_id"]: {r["source"] for r in self.incoming[m["entity_id"]]
                                       if r["source"] in self.papers and r["source"] not in exclude and r["relation"] in FACET_RELS
                                       and r.get("confidence") != "low"} for m in facets}
        closest = []
        for item in ranked[:10]:
            have = [m for m in facets if item["paper_id"] in paper_sets[m["entity_id"]]]
            miss = [m for m in facets if item["paper_id"] not in paper_sets[m["entity_id"]]]
            closest.append({"paper_id": item["paper_id"], "title": item["title"], "year": item.get("year"),
                            "weighted_overlap": round(sum(m["weight"] for m in have) / total_w, 3),
                            "overlapping_facets": [{"name": m["name"], "type": m["type"]} for m in have],
                            "missing_facets": [{"name": m["name"], "type": m["type"]} for m in miss]})
        closest.sort(key=lambda c: (-c["weighted_overlap"], c["title"]))
        covered, uncovered = [], []
        by_id = {m["entity_id"]: m for m in facets}
        for k in (2, 3, 4):
            for combo in combinations(facets, k):
                if len({m["type"] for m in combo}) < 2:
                    continue
                papers = set.intersection(*(paper_sets[m["entity_id"]] for m in combo))
                entry = {"facets": [m["name"] for m in combo], "types": sorted({m["type"] for m in combo}),
                         "weight": round(sum(m["weight"] for m in combo), 3), "_ids": frozenset(m["entity_id"] for m in combo)}
                if papers:
                    entry["papers"] = sorted(self.papers[p]["title"] for p in papers)[:5]
                    entry["paper_count"] = len(papers)
                    covered.append(entry)
                else:
                    uncovered.append(entry)
        cov_ids = [c["_ids"] for c in covered]
        maximal = [c for c in covered if not any(c["_ids"] < o for o in cov_ids)]
        unc_ids = {c["_ids"] for c in uncovered}
        covered_sets = set(cov_ids)
        minimal = []
        for c in uncovered:
            subs = [frozenset(s) for s in combinations(c["_ids"], len(c["_ids"]) - 1) if len(s) >= 2]
            typed = [s for s in subs if len({by_id[i]["type"] for i in s}) >= 2]
            if all(s in covered_sets for s in typed):
                minimal.append(c)
        for c in maximal + minimal:
            c.pop("_ids", None)
        for c in covered + uncovered:
            c.pop("_ids", None)
        maximal.sort(key=lambda c: (-c["weight"], c["facets"]))
        minimal.sort(key=lambda c: (-c["weight"], c["facets"]))
        best = closest[0]["weighted_overlap"] if closest else 0.0
        if not facets:
            signal, note = "insufficient_ontology_coverage", "No corpus-supported facet of the idea could be matched, so overlap cannot be judged."
        elif ontology_coverage < MIN_ONTOLOGY_COVERAGE and best < NOVELTY_CLOSE:
            signal, note = "insufficient_ontology_coverage", "Most of the idea's wording lies outside the ontology; the overlap estimate below covers only the matched part."
        elif best >= NOVELTY_CLOSE:
            signal, note = "close_match_found", "At least one indexed paper covers most of the idea's weighted facets."
        elif best >= NOVELTY_PARTIAL:
            signal, note = "partial_overlap", "Indexed papers cover parts of the idea; the uncovered combinations are the candidate gaps."
        else:
            signal, note = "no_close_match_in_indexed_corpus", "No indexed paper covers a large share of the idea's facets. This is not proof of novelty."
        return {"idea_facets": dict(idea_facets), "closest_prior_work": closest[:6],
                "covered_combinations": maximal[:5], "uncovered_combinations": minimal[:5],
                "facets_without_corpus_support": unsupported,
                "novelty_signal": signal, "novelty_note": note, "best_weighted_overlap": best,
                "thresholds": {"close": NOVELTY_CLOSE, "partial": NOVELTY_PARTIAL}}

    # -------------------------------------------------------------- unmatched
    def _unmatched(self, idea: str, matched) -> tuple[list[dict], float]:
        covered = set()
        for m in matched:
            for a in m["matched_aliases"]:
                covered.update(stems(a))
            covered.update(self._desc_stems.get(m["entity_id"], set()) if m["match_type"] == "description" else set())
        toks = [t for t in stems(idea) if len(t) >= 4 and t not in GENERIC]
        uniq = list(dict.fromkeys(toks))
        if not uniq:
            return [], 0.0
        missing = [t for t in uniq if t not in covered]
        coverage = 1 - len(missing) / len(uniq)
        listing = [{"term": t, "in_corpus_vocabulary": self.corpus_vocab[t] > 0, "corpus_paper_count": self.corpus_vocab[t]}
                   for t in missing]
        listing.sort(key=lambda x: (-x["corpus_paper_count"], x["term"]))
        return listing[:15], round(coverage, 3)

    # ------------------------------------------------------------ reading order
    def _reading_order(self, ranked, exclude, facet_ids, limit=7):
        pool: dict[str, str] = {}
        for item in ranked[:5]:
            pool[item["paper_id"]] = "directly matches the idea"
        for item in ranked[:5]:
            for r in self.outgoing[item["paper_id"]]:
                if r["relation"] in ("CITES", "BUILDS_ON") and r["target"] in self.papers and r["target"] not in exclude:
                    pool.setdefault(r["target"], f"prerequisite: cited or built on by {self.papers[item['paper_id']]['title'][:50]}")
        surveys = [p for p in self.papers.values() if p.get("paper_kind") == "survey" and p["id"] not in exclude
                   and facet_ids & self._paper_facets(p["id"], False)]
        surveys.sort(key=lambda p: (-len(facet_ids & self._paper_facets(p["id"], False)), p["title"]))
        for p in surveys[:2]:
            pool.setdefault(p["id"], "survey overlapping the idea's facets")
        deps = defaultdict(set)
        for pid in pool:
            for r in self.outgoing[pid]:
                if r["relation"] in ("CITES", "BUILDS_ON") and r["target"] in pool:
                    deps[pid].add(r["target"])

        def key(pid):
            p = self.papers[pid]
            return (KIND_RANK.get(p.get("paper_kind", "other"), 4) - (1 if p.get("foundation_score", 0) >= 2 else 0),
                    p.get("year") or 9999, -p.get("foundation_score", 0), p["title"])

        order, done = [], set()
        remaining = set(pool)
        while remaining:
            ready = [p for p in remaining if deps[p] <= done] or list(remaining)
            nxt = min(ready, key=key)
            order.append(nxt); done.add(nxt); remaining.discard(nxt)
        out = []
        for pid in order[:limit]:
            p = self.papers[pid]
            parts = [pool[pid]]
            if p.get("paper_kind") == "survey":
                parts.append("a survey: start here for orientation")
            if p.get("foundation_score", 0) >= 1:
                ib = p.get("inbound_links", {})
                parts.append(f"foundation score {p['foundation_score']} ({ib.get('CITES', 0)} corpus citations, {ib.get('BUILDS_ON', 0)} build-on, {ib.get('COMPARES_AGAINST', 0)} comparisons)")
            have = sorted(self.entities[i]["name"] for i in facet_ids & self._paper_facets(pid))[:3]
            if have:
                parts.append("shares: " + ", ".join(have))
            ev = [r["id"] for r in self.outgoing[pid] if r["relation"] in ("INTRODUCES_METHOD", "ADDRESSES_PROBLEM") and r["target"] in facet_ids][:2]
            out.append({"order": len(out) + 1, "paper_id": pid, "title": p["title"], "year": p.get("year"),
                        "paper_kind": p.get("paper_kind"), "reason": "; ".join(parts), "evidence_ids": ev})
        return out

    # ------------------------------------------------------------------- report
    def _unique_relation_targets(self, paper_ids, types, relation_types):
        grouped: dict[str, dict] = {}
        for pid in paper_ids:
            for r in self.outgoing[pid]:
                if r["relation"] not in relation_types:
                    continue
                t = self.entities.get(r["target"])
                if not t or t["type"] not in types or r.get("confidence") == "low":
                    continue
                g = grouped.setdefault(t["id"], {"id": t["id"], "entity_id": t["id"], "name": t["name"], "type": t["type"],
                                                 "supporting_papers": [], "evidence_ids": []})
                if pid not in g["supporting_papers"]:
                    g["supporting_papers"].append(pid)
                g["evidence_ids"].append(r["id"])
        return sorted(grouped.values(), key=lambda g: (-len(g["supporting_papers"]), g["name"]))

    def reason(self, idea: str, limit: int = 8, exclude_paper_ids: set[str] | None = None,
               use_citation_expansion: bool = True, use_propagation: bool = True) -> dict[str, Any]:
        exclude = set(exclude_paper_ids or ())
        matched = self.match_terms(idea)
        if not any(m["match_type"] in ("exact", "stem") for m in matched) and len(matched) < 2:
            matched = []  # a lone synonym/description match is too weak to switch the graph on
        unmatched, coverage = self._unmatched(idea, matched)
        ranked: list[dict[str, Any]] = []
        if matched:
            graph, contrib = self._score(matched, idea, exclude, use_citation_expansion, use_propagation)
            lex = self._lexical(idea, exclude)
            for pid, g in graph.items():
                p = self.papers[pid]
                final = g + LEXICAL_WEIGHT * lex.get(pid, 0.0) + FOUNDATION_TIEBREAK * p.get("foundation_score", 0)
                best = sorted(contrib[pid], key=lambda c: -c["value"])[:3]
                seed = []
                for c in best:
                    if "via_paper" in c:
                        seed.append(self._step(c["rel"], reverse=(c["rel"]["source"] != pid)))
                    elif c.get("via"):
                        seed.append(self._step(c["via"], c["via_reverse"])); seed.append(self._step(c["rel"], True))
                    else:
                        seed.append(self._step(c["rel"], True))
                ranked.append({"paper_id": pid, "title": p["title"], "year": p.get("year"), "paper_kind": p.get("paper_kind"),
                               "source_url": p.get("source_url"), "graph_score": round(g, 4),
                               "lexical_tiebreak": round(lex.get(pid, 0.0), 4), "score": round(final, 4),
                               "matched_entities": sorted({self.entities[c["entity_id"]]["name"] for c in contrib[pid] if c["entity_id"]}),
                               "seed_path": seed})
            ranked.sort(key=lambda r: (-r["score"], r["title"]))
        top = ranked[:limit]
        facet_ids = {m["entity_id"] for m in matched if m["supported"]}
        total_w = sum(m["weight"] for m in matched if m["supported"]) or 1.0
        for item in top:
            direct = self._paper_facets(item["paper_id"])
            have = [m for m in matched if m["supported"] and m["entity_id"] in direct]
            item["facet_overlap"] = round(sum(m["weight"] for m in have) / total_w, 3)
            item["overlapping_facets"] = [m["name"] for m in have]
            item["missing_facets"] = [m["name"] for m in matched if m["supported"] and m["entity_id"] not in direct]
        top_ids = [i["paper_id"] for i in top]
        chains = self._entity_chains(matched, exclude) if matched else []
        paths = list(chains)
        for item in top[:5]:
            for pth in self._paper_paths(item["paper_id"], exclude):
                pth["source_paper_id"] = item["paper_id"]
                paths.append(pth)
        paths = paths[:14]
        limitations = self._unique_relation_targets(top_ids, {"Limitation"}, {"MOTIVATED_BY_LIMITATION"})
        directions = self._unique_relation_targets(top_ids, {"ResearchDirection"}, {"PURSUES_DIRECTION"})
        problems = self._unique_relation_targets(top_ids, {"ResearchProblem"}, {"ADDRESSES_PROBLEM"})
        methods = self._unique_relation_targets(top_ids, {"Method"}, {"USES_METHOD", "INTRODUCES_METHOD", "EVALUATES_METHOD"})
        datasets = self._unique_relation_targets(top_ids, {"Dataset"}, {"EVALUATES_ON", "INTRODUCES_DATASET"})
        architectures = self._unique_relation_targets(top_ids, {"Architecture"}, {"IMPLEMENTS_ARCHITECTURE"})
        domains = self._unique_relation_targets(top_ids, {"ApplicationDomain"}, {"APPLIED_IN_DOMAIN"})
        foundational, seen = [], set()
        for pid in top_ids:
            for r in self.outgoing[pid]:
                if r["relation"] in ("CITES", "BUILDS_ON") and r["target"] in self.papers and r["target"] not in seen and r["target"] not in exclude:
                    seen.add(r["target"]); c = self.papers[r["target"]]
                    foundational.append({"paper_id": r["target"], "title": c["title"], "relationship": r["relation"],
                                         "evidence_ids": [r["id"]], "source_url": c.get("source_url")})
        method_ids = {m["entity_id"] for m in matched if m["type"] == "Method"} | {m["entity_id"] for m in methods}
        lineage = []
        for r in self.relationships:
            if r["relation"] in ("IMPROVES_ON", "BUILDS_ON", "COMPARES_AGAINST") and r["source"] in method_ids and r["target"] in self.entities \
                    and self.entities[r["source"]]["type"] == "Method" and not self._skip_edge(r, exclude):
                lineage.append({"from": self.entities[r["source"]]["name"], "relation": r["relation"], "to": self.entities[r["target"]]["name"],
                                "evidence_id": r["id"], "supporting_papers": r.get("supporting_papers", [])})
        reading = self._reading_order(top, exclude, facet_ids) if top else []
        og = self._overlap_and_gap(matched, top, exclude, coverage)
        uncertainty = [
            "The system reports overlap within the indexed corpus; it does not prove or disprove absolute novelty.",
            "Edges come from hand-written alias and cue rules over titles and abstracts; inspect the evidence IDs and confidence before relying on a claim.",
            f"{self.state['metadata'].get('papers_without_openalex_references', 0)} corpus papers have no OpenAlex reference list, so citation evidence under-represents recent work.",
        ]
        if not matched:
            uncertainty.insert(0, "No controlled-vocabulary entity matched the input; the input appears to be outside the indexed topic or uses unfamiliar wording. See unmatched_terms.")
        elif coverage < MIN_ONTOLOGY_COVERAGE:
            uncertainty.insert(0, "Less than a quarter of the idea's key terms are covered by the ontology; treat the positioning as partial.")
        if og["facets_without_corpus_support"]:
            uncertainty.insert(0, "Known concepts with no supporting paper in the corpus: " + ", ".join(og["facets_without_corpus_support"]) + ".")
        evidence_ids: list[str] = []
        for item in top:
            evidence_ids.extend(s["evidence_id"] for s in item.get("seed_path", []))
        for pth in paths:
            evidence_ids.extend(pth["evidence_ids"])
        for item in limitations + directions + problems + methods + datasets + architectures + domains:
            evidence_ids.extend(item["evidence_ids"][:3])
        evidence_ids.extend(l["evidence_id"] for l in lineage)
        for item in reading:
            evidence_ids.extend(item["evidence_ids"])
        evidence = []
        for eid in dict.fromkeys(evidence_ids):
            r = self.relationship_by_id.get(eid)
            if r:
                evidence.append({"evidence_id": eid, "source_paper_id": r.get("source_paper"), "relation": r["relation"],
                                 "source": r["source"], "target": r["target"], "quote": r.get("evidence"),
                                 "source_location": r.get("source_location"), "evidence_type": r.get("evidence_type"),
                                 "confidence": r.get("confidence"), "rule": r.get("rule")})
        return {
            "input_summary": idea.strip(),
            "matched_concepts": matched,
            "unmatched_terms": unmatched,
            "related_methods": methods,
            "related_architectures": architectures,
            "related_problems": problems,
            "related_datasets": datasets,
            "related_domains": domains,
            "related_papers": top,
            "foundational_connections": foundational,
            "method_lineage": lineage,
            "multi_hop_evidence_paths": paths,
            "known_limitations": limitations,
            "research_directions": directions,
            "overlap_and_gap": og,
            "recommended_reading_order": reading,
            "external_foundations": self.state.get("external_foundations", [])[:5],
            "evidence": evidence,
            "flat_retrieval_baseline": flat_retrieve(idea, list(self.papers.values()), limit=5, exclude_ids=exclude),
            "uncertainty_notes": uncertainty,
            "reasoning_summary": {
                "graph_used": bool(ranked), "candidate_count": len(ranked), "matched_concept_count": len(matched),
                "ontology_coverage": coverage, "novelty_signal": og["novelty_signal"],
                "entity_chain_count": len(chains),
                "match_types": dict(Counter(m["match_type"] for m in matched)),
            },
        }

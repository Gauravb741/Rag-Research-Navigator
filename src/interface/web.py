"""Secure same-origin web interface for the research-onboarding reasoner."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.reasoning.engine import GraphReasoner

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data" / "knowledge" / "knowledge_state.json"
MAX_REQUEST_BYTES = 12_000
MAX_IDEA_CHARS = 5_000
MIN_IDEA_CHARS = 12


def load_reasoner() -> GraphReasoner:
    if not STATE_PATH.is_file():
        raise RuntimeError("Knowledge state is unavailable. Build it before starting the web interface.")
    return GraphReasoner(json.loads(STATE_PATH.read_text(encoding="utf-8")))


def validate_payload(raw: bytes) -> str:
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Request body is too large.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Request must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"idea"}:
        raise ValueError("Request must contain only an idea field.")
    idea = payload.get("idea")
    if not isinstance(idea, str):
        raise ValueError("The idea must be text.")
    idea = idea.strip()
    if len(idea) < MIN_IDEA_CHARS:
        raise ValueError(f"The idea must contain at least {MIN_IDEA_CHARS} characters.")
    if len(idea) > MAX_IDEA_CHARS:
        raise ValueError(f"The idea must be at most {MAX_IDEA_CHARS} characters.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in idea):
        raise ValueError("The idea contains unsupported control characters.")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in idea):
        raise ValueError("The idea contains invalid Unicode characters.")
    return idea


def public_graph(state: dict[str, Any]) -> dict[str, Any]:
    entity_fields = {
        "id", "type", "name", "title", "description", "abstract", "year",
        "doi", "arxiv_id", "source_url", "citation_count", "corpus_index",
        "corpus_role", "orcid", "paper_kind", "kind",
    }
    entities = [{key: value for key, value in entity.items() if key in entity_fields} for entity in state.get("entities", [])]
    relationships = []
    for relationship in state.get("relationships", []):
        relationships.append({
            "id": relationship.get("id"),
            "source": relationship.get("source"),
            "relation": relationship.get("relation"),
            "target": relationship.get("target"),
            "evidence": relationship.get("evidence", ""),
            "source_location": relationship.get("source_location", ""),
            "evidence_type": relationship.get("evidence_type", ""),
            "source_paper": relationship.get("source_paper"),
        })
    metadata = state.get("metadata", {})
    return {
        "metadata": {
            "corpus_size": metadata.get("corpus_size", 0),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "entity_type_counts": metadata.get("entity_type_counts", {}),
            "relationship_type_counts": metadata.get("relationship_type_counts", {}),
        },
        "entities": entities,
        "relationships": relationships,
    }


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Interactive evidence-backed exploration of the RAG research landscape.">
<title>RAG Research Navigator</title>
<style>
:root {
  --bg: #080d12;
  --bg-2: #0e151c;
  --panel: rgba(17, 26, 34, .88);
  --panel-2: #111b24;
  --line: rgba(159, 188, 201, .16);
  --line-strong: rgba(159, 188, 201, .3);
  --text: #eef5f5;
  --muted: #91a5ad;
  --dim: #5f727b;
  --mint: #62e1c8;
  --mint-soft: rgba(98, 225, 200, .13);
  --sky: #72b8f6;
  --sky-soft: rgba(114, 184, 246, .13);
  --gold: #e7ba72;
  --gold-soft: rgba(231, 186, 114, .14);
  --coral: #f28f83;
  --violet: #c49cff;
  --shadow: 0 24px 90px rgba(0, 0, 0, .28);
  --radius: 18px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; min-width: 320px; background: radial-gradient(circle at 12% 0%, rgba(36, 103, 104, .22), transparent 28rem), radial-gradient(circle at 90% 12%, rgba(58, 79, 133, .18), transparent 30rem), var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; }
button, textarea, input, select { font: inherit; }
button { cursor: pointer; }
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible, [tabindex]:focus-visible { outline: 2px solid var(--mint); outline-offset: 3px; }
.app-shell { width: min(1480px, calc(100% - 42px)); margin: 0 auto; }
.topbar { height: 76px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--line); position: relative; z-index: 5; }
.brand { display: flex; align-items: center; gap: 12px; font-weight: 820; letter-spacing: -.03em; }
.brand-mark { width: 31px; height: 31px; border: 1px solid rgba(98, 225, 200, .5); border-radius: 10px; display: grid; place-items: center; color: var(--mint); background: var(--mint-soft); box-shadow: 0 0 28px rgba(98, 225, 200, .14); }
.brand-sub { color: var(--dim); font-size: 11px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; margin-left: 6px; }
.top-actions { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; }
.server-status { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; }
.server-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 12px var(--mint); }
.eyebrow { color: var(--mint); font-size: 10px; font-weight: 850; letter-spacing: .18em; text-transform: uppercase; }
.hero { min-height: 610px; display: grid; grid-template-columns: .9fr 1.1fr; gap: 30px; align-items: center; padding: 56px 0 54px; position: relative; }
.hero-copy { position: relative; z-index: 2; max-width: 620px; }
.hero h1 { font-size: clamp(44px, 6.8vw, 94px); line-height: .91; letter-spacing: -.075em; margin: 15px 0 25px; max-width: 760px; }
.hero h1 .accent { color: var(--mint); }
.hero-description { color: var(--muted); font-size: 17px; max-width: 535px; margin: 0 0 25px; }
.hero-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.tag { border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; color: var(--muted); font-size: 11px; background: rgba(255, 255, 255, .025); }
.hero-map { min-height: 500px; position: relative; border: 1px solid var(--line); border-radius: 26px; overflow: hidden; background: linear-gradient(145deg, rgba(16, 29, 37, .8), rgba(7, 14, 20, .74)); box-shadow: var(--shadow); }
.hero-map:before { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(150, 190, 200, .04) 1px, transparent 1px), linear-gradient(90deg, rgba(150, 190, 200, .04) 1px, transparent 1px); background-size: 34px 34px; mask-image: linear-gradient(to bottom right, black, transparent 85%); pointer-events: none; }
.hero-map-caption { position: absolute; left: 20px; top: 18px; z-index: 2; color: var(--muted); font-size: 11px; letter-spacing: .1em; text-transform: uppercase; }
#mini-graph { width: 100%; height: 100%; min-height: 500px; opacity: .8; }
.idea-card { position: absolute; left: 22px; right: 22px; bottom: 22px; z-index: 3; border: 1px solid rgba(98, 225, 200, .3); border-radius: 16px; padding: 17px; background: rgba(9, 17, 23, .86); backdrop-filter: blur(18px); box-shadow: 0 15px 48px rgba(0, 0, 0, .28); }
.idea-card label { display: block; color: var(--muted); font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
textarea { width: 100%; min-height: 88px; resize: vertical; border: 1px solid var(--line-strong); border-radius: 12px; padding: 12px; color: var(--text); background: rgba(255, 255, 255, .04); outline: none; }
textarea:focus { border-color: var(--mint); box-shadow: 0 0 0 4px rgba(98, 225, 200, .08); }
.idea-actions { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 11px; }
.input-help { color: var(--dim); font-size: 11px; }
.primary { border: 1px solid rgba(98, 225, 200, .45); border-radius: 10px; padding: 10px 14px; background: var(--mint); color: #09201d; font-weight: 850; box-shadow: 0 0 25px rgba(98, 225, 200, .15); transition: transform .18s ease, box-shadow .18s ease; }
.primary:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(98, 225, 200, .22); }
.primary:disabled { cursor: wait; opacity: .65; transform: none; }
.stats-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: -5px 0 32px; position: relative; z-index: 2; }
.stat { border: 1px solid var(--line); border-radius: 14px; padding: 15px 18px; background: rgba(16, 25, 33, .72); }
.stat-value { color: var(--text); font-size: 28px; font-weight: 820; letter-spacing: -.05em; }
.stat-label { color: var(--muted); font-size: 11px; margin-top: 2px; }
.analysis { display: none; padding-bottom: 78px; }
.analysis.visible { display: block; }
.analysis-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin: 24px 0 17px; }
.analysis-title { margin: 5px 0 0; font-size: clamp(28px, 4vw, 48px); line-height: .98; letter-spacing: -.06em; }
.analysis-summary { color: var(--muted); font-size: 12px; margin: 9px 0 0; }
.analysis-actions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.ghost { border: 1px solid var(--line); border-radius: 9px; padding: 8px 10px; background: rgba(255, 255, 255, .035); color: var(--muted); font-size: 11px; font-weight: 760; }
.ghost:hover, .ghost.active { border-color: rgba(98, 225, 200, .5); color: var(--mint); background: var(--mint-soft); }
.ghost:disabled { opacity: .48; cursor: default; }
.workspace { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 14px; align-items: start; }
.panel { border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); box-shadow: var(--shadow); }
.graph-panel { min-width: 0; overflow: hidden; }
.graph-toolbar { min-height: 60px; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 11px 13px; border-bottom: 1px solid var(--line); background: rgba(255, 255, 255, .015); }
.mode-tabs { display: flex; flex-wrap: wrap; gap: 5px; }
.mode-tab { border: 0; border-radius: 7px; padding: 7px 9px; background: transparent; color: var(--dim); font-size: 11px; font-weight: 790; }
.mode-tab:hover, .mode-tab.active { background: var(--mint-soft); color: var(--mint); }
.toolbar-tools { display: flex; gap: 5px; }
.icon-button { width: 31px; height: 31px; display: grid; place-items: center; border: 1px solid var(--line); border-radius: 8px; background: rgba(255, 255, 255, .035); color: var(--muted); font-weight: 820; }
.icon-button:hover { color: var(--text); border-color: var(--line-strong); }
.graph-search { width: 160px; border: 1px solid var(--line); border-radius: 8px; padding: 8px 9px; color: var(--text); background: rgba(0, 0, 0, .18); font-size: 11px; }
.graph-stage { position: relative; height: 660px; background: radial-gradient(circle at 50% 47%, rgba(98, 225, 200, .05), transparent 38%), #0a1118; overflow: hidden; cursor: grab; }
.graph-stage:active { cursor: grabbing; }
.graph-stage:before { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(159, 188, 201, .045) 1px, transparent 1px), linear-gradient(90deg, rgba(159, 188, 201, .045) 1px, transparent 1px); background-size: 42px 42px; mask-image: radial-gradient(circle at center, black, transparent 82%); pointer-events: none; }
#graph-svg { width: 100%; height: 100%; display: block; touch-action: none; position: relative; z-index: 1; }
.cluster-label { fill: rgba(145, 165, 173, .5); font-size: 9px; font-weight: 850; letter-spacing: .18em; text-transform: uppercase; pointer-events: none; }
.edge { fill: none; stroke: rgba(128, 163, 176, .17); stroke-width: 1; transition: stroke .25s ease, stroke-width .25s ease, opacity .25s ease; }
.edge.highlight { stroke: var(--mint); stroke-width: 2.7; opacity: 1; filter: drop-shadow(0 0 6px rgba(98, 225, 200, .52)); }
.edge.hover-highlight { stroke: rgba(231, 186, 114, .72); stroke-width: 1.8; opacity: .95; }
.edge.dimmed { opacity: .035; }
.edge-hit { fill: none; stroke: transparent; stroke-width: 15; cursor: pointer; }
.edge-label { fill: var(--gold); font-size: 9px; font-weight: 800; letter-spacing: .04em; pointer-events: none; paint-order: stroke; stroke: #0a1118; stroke-width: 4px; stroke-linejoin: round; transition: opacity .18s ease; }
.edge-label-hidden { opacity: 0; }
.node-group { cursor: pointer; transition: opacity .25s ease; }
.node-shape { stroke-width: 1.3; transition: transform .18s ease, opacity .25s ease, stroke .25s ease; }
.node-group:hover .node-shape, .node-group.selected .node-shape { stroke: var(--text); stroke-width: 2.4; filter: drop-shadow(0 0 10px currentColor); }
.node-group.hovered .node-shape { stroke: var(--gold); stroke-width: 2.2; filter: drop-shadow(0 0 9px rgba(231, 186, 114, .6)); }
.node-group.highlight .node-shape { stroke: var(--mint); stroke-width: 2.2; filter: drop-shadow(0 0 9px rgba(98, 225, 200, .65)); }
.node-group.relation-focus .node-shape { stroke: var(--gold); stroke-width: 2.3; filter: drop-shadow(0 0 8px rgba(231, 186, 114, .55)); }
.node-group.dimmed { opacity: .12; }
.node-label, .node-hover-label { fill: var(--muted); font-size: 10px; pointer-events: none; paint-order: stroke; stroke: #0a1118; stroke-width: 3px; stroke-linejoin: round; }
.node-hover-label { opacity: 0; transition: opacity .14s ease; }
.node-group:hover .node-hover-label { opacity: 1; fill: var(--text); font-weight: 800; }
.node-group.selected .node-label, .node-group.highlight .node-label { fill: var(--text); font-weight: 800; }
.graph-tooltip { position: absolute; z-index: 4; max-width: 300px; pointer-events: none; padding: 10px 12px; border: 1px solid var(--line-strong); border-radius: 10px; background: rgba(5, 10, 14, .96); color: var(--text); font-size: 11px; line-height: 1.4; box-shadow: var(--shadow); opacity: 0; transform: translateY(4px); transition: opacity .14s ease, transform .14s ease; }
.graph-tooltip.visible { opacity: 1; transform: translateY(0); }
.tooltip-type { color: var(--mint); font-size: 9px; font-weight: 850; letter-spacing: .12em; text-transform: uppercase; }
.tooltip-title { margin-top: 3px; color: var(--text); font-weight: 800; }
.tooltip-meta { margin-top: 3px; color: var(--muted); font-size: 10px; }
.graph-hint { position: absolute; left: 15px; bottom: 13px; z-index: 2; color: var(--dim); font-size: 9px; letter-spacing: .08em; text-transform: uppercase; pointer-events: none; }
.graph-legend { display: flex; flex-wrap: wrap; gap: 8px 13px; padding: 11px 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 10px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-dot { width: 9px; height: 9px; border-radius: 3px; background: var(--muted); }
.legend-dot.paper { border-radius: 50%; background: var(--sky); }
.legend-dot.method { transform: rotate(45deg); background: var(--mint); }
.legend-dot.problem { background: var(--coral); clip-path: polygon(50% 0, 100% 100%, 0 100%); }
.legend-dot.direction { background: var(--violet); border-radius: 50%; }
.filters { display: none; padding: 13px 14px; border-top: 1px solid var(--line); background: rgba(255, 255, 255, .02); }
.filters.visible { display: block; }
.filter-head { display: flex; align-items: center; justify-content: space-between; color: var(--muted); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
.filter-grid { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }
.filter-label { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 11px; }
.filter-label input { accent-color: var(--mint); }
.filter-select { border: 1px solid var(--line); border-radius: 7px; color: var(--muted); background: var(--bg-2); padding: 6px 8px; font-size: 11px; }
.inspector { min-height: 660px; padding: 20px; position: sticky; top: 14px; }
.inspector-empty { min-height: 600px; display: grid; place-items: center; text-align: center; color: var(--muted); }
.inspector-empty-inner { max-width: 220px; }
.inspector-orbit { width: 90px; height: 90px; margin: 0 auto 19px; border: 1px dashed rgba(98, 225, 200, .5); border-radius: 50%; display: grid; place-items: center; color: var(--mint); font-size: 26px; box-shadow: 0 0 45px rgba(98, 225, 200, .08); }
.inspector h3 { margin: 6px 0 8px; font-size: 20px; line-height: 1.08; letter-spacing: -.04em; }
.inspector-type { color: var(--mint); font-size: 10px; font-weight: 850; text-transform: uppercase; letter-spacing: .13em; }
.inspector-expand { margin: 2px 0 14px; }
.inspector-copy { color: var(--muted); font-size: 12px; margin: 0 0 14px; }
.inspector-section { border-top: 1px solid var(--line); padding: 14px 0; }
.inspector-section h4 { margin: 0 0 8px; color: var(--dim); font-size: 10px; letter-spacing: .12em; text-transform: uppercase; }
.meta-row { display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; color: var(--muted); font-size: 11px; }
.meta-row strong { color: var(--text); text-align: right; }
.evidence-card { border: 1px solid rgba(231, 186, 114, .28); border-radius: 11px; padding: 12px; background: var(--gold-soft); }
.evidence-card-title { color: var(--gold); font-size: 10px; font-weight: 850; letter-spacing: .1em; text-transform: uppercase; }
.evidence-card-quote { color: #d3c2a5; font-size: 12px; margin-top: 6px; }
.evidence-card-source { color: var(--muted); font-size: 10px; margin-top: 7px; }
.small-link { color: var(--mint); font-size: 11px; font-weight: 760; text-decoration: none; }
.small-link:hover { text-decoration: underline; }
.pill-list { display: flex; flex-wrap: wrap; gap: 5px; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 5px 7px; color: var(--muted); background: rgba(255, 255, 255, .03); font-size: 10px; }
.report { margin-top: 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.report .panel { padding: 18px; }
.report details { border-top: 1px solid var(--line); padding: 12px 0; }
.report details:first-of-type { border-top: 0; padding-top: 0; }
.report summary { cursor: pointer; color: var(--text); font-size: 12px; font-weight: 800; }
.report-content { color: var(--muted); font-size: 12px; padding-top: 10px; }
.report-path { padding: 9px 0; border-bottom: 1px solid rgba(159, 188, 201, .1); }
button.report-path { width: 100%; color: var(--muted); background: transparent; border: 0; text-align: left; cursor: pointer; }
.report-path:last-child { border-bottom: 0; }
.timeline-panel { margin-top: 14px; padding: 18px; overflow: hidden; }
.timeline-head { display: flex; align-items: end; justify-content: space-between; gap: 10px; margin-bottom: 15px; }
.timeline-head h3 { margin: 0; font-size: 16px; letter-spacing: -.03em; }
.timeline-head span { color: var(--dim); font-size: 10px; }
.timeline { display: flex; gap: 0; overflow-x: auto; padding: 12px 2px 8px; }
.timeline-year { min-width: 150px; position: relative; padding-right: 14px; }
.timeline-year:before { content: ""; position: absolute; left: 0; right: 0; top: 15px; height: 1px; background: var(--line-strong); }
.timeline-year-label { position: relative; display: inline-block; padding: 0 7px 0 0; color: var(--mint); background: var(--panel); font-size: 12px; font-weight: 850; }
.timeline-papers { display: flex; gap: 5px; padding-top: 15px; min-height: 52px; flex-wrap: wrap; align-content: flex-start; }
.timeline-paper { width: 12px; height: 12px; border: 2px solid var(--sky); border-radius: 50%; padding: 0; background: rgba(114, 184, 246, .22); transition: transform .15s ease, background .15s ease; }
.timeline-paper:hover, .timeline-paper.selected { background: var(--mint); border-color: var(--mint); transform: scale(1.35); }
.timeline-paper[aria-label] { position: relative; }
.loading { position: fixed; inset: 0; z-index: 20; display: none; place-items: center; background: rgba(5, 10, 14, .83); backdrop-filter: blur(12px); }
.loading.visible { display: grid; }
.loading-card { width: min(420px, calc(100% - 36px)); border: 1px solid var(--line-strong); border-radius: 18px; padding: 22px; background: var(--panel-2); box-shadow: var(--shadow); }
.loading-kicker { color: var(--mint); font-size: 10px; font-weight: 850; letter-spacing: .15em; text-transform: uppercase; }
.loading h2 { margin: 7px 0 17px; font-size: 23px; letter-spacing: -.04em; }
.loading-steps { display: grid; gap: 9px; }
.loading-step { display: flex; align-items: center; gap: 9px; color: var(--dim); font-size: 12px; transition: color .2s ease; }
.loading-step.active { color: var(--text); }
.loading-step.done { color: var(--mint); }
.loading-step i { width: 8px; height: 8px; border: 1px solid currentColor; border-radius: 50%; display: inline-block; }
.loading-step.active i { background: var(--mint); box-shadow: 0 0 12px var(--mint); }
.status { min-height: 18px; color: var(--muted); font-size: 11px; margin-top: 8px; }
.status.error { color: var(--coral); font-weight: 760; }
.footer { display: flex; justify-content: space-between; gap: 20px; border-top: 1px solid var(--line); padding: 22px 0 34px; color: var(--dim); font-size: 11px; }
.hidden { display: none !important; }
@media (max-width: 1080px) { .hero { grid-template-columns: 1fr; } .hero-copy { max-width: 780px; } .hero-map { min-height: 470px; } .workspace { grid-template-columns: minmax(0, 1fr) 310px; } .graph-search { width: 120px; } }
@media (max-width: 820px) { .app-shell { width: min(100% - 24px, 1480px); } .topbar { height: 64px; } .brand-sub, .topnote { display: none; } .workspace { grid-template-columns: 1fr; } .inspector { position: static; min-height: 0; } .inspector-empty { min-height: 230px; } .analysis-head { align-items: flex-start; flex-direction: column; } .analysis-actions { justify-content: flex-start; } .report { grid-template-columns: 1fr; } }
@media (max-width: 540px) { .hero { padding: 35px 0 35px; } .hero h1 { font-size: 50px; } .hero-description { font-size: 15px; } .stats-strip { gap: 6px; } .stat { padding: 12px 10px; } .stat-value { font-size: 22px; } .stat-label { font-size: 9px; } .hero-map { min-height: 570px; } .idea-card { left: 12px; right: 12px; bottom: 12px; } .idea-actions { align-items: flex-start; flex-direction: column; } .primary { width: 100%; } .graph-toolbar { align-items: flex-start; flex-direction: column; } .toolbar-tools { width: 100%; } .graph-search { flex: 1; width: auto; } .graph-stage { height: 500px; } .footer { flex-direction: column; } }
@media (prefers-reduced-motion: reduce) { *, *:before, *:after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; } }
</style>
</head>
<body>
<div class="app-shell">
  <header class="topbar">
    <div class="brand"><div class="brand-mark" aria-hidden="true">↗</div><span>RAG Research Navigator</span><span class="brand-sub">Evidence atlas</span></div>
    <div class="top-actions"><span class="server-status"><i class="server-dot" aria-hidden="true"></i> knowledge state online</span></div>
  </header>
  <main>
    <section class="hero" id="landing">
      <div class="hero-copy">
        <div class="eyebrow">Research paper onboarding · graph-aware</div>
        <h1>Find your idea<br>in the <span class="accent">network.</span></h1>
        <p class="hero-description">A visual research workspace for seeing how a new RAG idea connects to methods, problems, papers, limitations, and the next direction worth exploring.</p>
        <div class="hero-tags"><span class="tag">multi-hop reasoning</span><span class="tag">evidence paths</span><span class="tag">72-paper corpus</span></div>
      </div>
      <div class="hero-map" aria-label="Live preview of the research graph">
        <div class="hero-map-caption">live knowledge landscape</div>
        <svg id="mini-graph" viewBox="0 0 700 500" role="img" aria-label="A preview of the RAG research knowledge graph"></svg>
        <div class="idea-card">
          <label for="idea">Describe a new RAG research idea</label>
          <textarea id="idea" maxlength="5000" aria-describedby="idea-help">I want to build an explainable RAG system for research question answering that follows citation networks and entity-relation paths across papers instead of retrieving isolated chunks.</textarea>
          <div class="idea-actions"><span class="input-help" id="idea-help"><span id="char-count">0</span> / 5,000 · your idea stays in this request</span><button class="primary" id="analyze" type="button">Analyze research idea <span aria-hidden="true">→</span></button></div>
          <div class="status" id="status" aria-live="polite"></div>
        </div>
      </div>
    </section>
    <section class="stats-strip" aria-label="Knowledge state counts"><div class="stat"><div class="stat-value" id="stat-papers">—</div><div class="stat-label">research papers</div></div><div class="stat"><div class="stat-value" id="stat-entities">—</div><div class="stat-label">typed entities</div></div><div class="stat"><div class="stat-value" id="stat-relationships">—</div><div class="stat-label">evidence-bearing relationships</div></div></section>
    <section class="analysis" id="analysis" aria-live="polite">
      <div class="analysis-head"><div><div class="eyebrow">Research position</div><h2 class="analysis-title" id="analysis-title">Your idea, in context.</h2><p class="analysis-summary" id="analysis-summary"></p></div><div class="analysis-actions"><button class="ghost active" data-mode="landscape" type="button">Overview</button><button class="ghost" data-mode="position" type="button">Research position</button><button class="ghost" data-mode="focus" type="button">Focus</button><button class="ghost" data-mode="paths" type="button">Reasoning paths</button><button class="ghost" data-mode="paper" type="button">Paper network</button><button class="ghost" data-mode="method" type="button">Method network</button></div></div>
      <div class="workspace">
        <section class="panel graph-panel" aria-label="Interactive research knowledge graph">
          <div class="graph-toolbar"><div class="mode-tabs"><button class="icon-button" id="zoom-in" type="button" aria-label="Zoom in" title="Zoom in">+</button><button class="icon-button" id="zoom-out" type="button" aria-label="Zoom out" title="Zoom out">−</button><button class="ghost" id="fit" type="button" title="Fit graph">Fit</button><button class="ghost" id="reset" type="button" title="Reset graph">Reset</button><button class="ghost" id="focus" type="button" title="Focus selected node">Focus</button><button class="ghost" id="filters-toggle" type="button" title="Show filters">Filters</button></div><div class="toolbar-tools"><input class="graph-search" id="node-search" type="search" placeholder="Search graph…" aria-label="Search graph nodes"><button class="icon-button" id="timeline-toggle" type="button" aria-label="Toggle timeline" title="Toggle timeline">⌁</button></div></div>
          <div class="filters" id="filters"><div class="filter-head"><span>show entity types</span><button class="ghost" id="reset-filters" type="button">Reset filters</button></div><div class="filter-grid" id="type-filters"></div><div style="margin-top:10px"><select class="filter-select" id="relation-filter" aria-label="Filter relationship type"><option value="all">All relationship types</option></select></div></div>
          <div class="graph-stage"><svg id="graph-svg" viewBox="0 0 1000 700" role="application" aria-label="Interactive RAG research graph"></svg><div class="graph-tooltip" id="tooltip" role="status"></div><div class="graph-hint" aria-hidden="true">hover for context · click to focus · drag to explore</div></div>
          <div class="graph-legend" id="legend"></div>
        </section>
        <aside class="panel inspector" id="inspector" aria-label="Selected entity and evidence inspector"><div class="inspector-empty"><div class="inspector-empty-inner"><div class="inspector-orbit" aria-hidden="true">◎</div><div class="eyebrow">Explore the atlas</div><p>Click a paper, method, problem, or relationship to inspect its evidence and connected research.</p></div></div></aside>
      </div>
      <section class="timeline-panel panel" id="timeline-panel"><div class="timeline-head"><h3>Research evolution</h3><span>publication years from the indexed corpus · click a paper to focus it</span></div><div class="timeline" id="timeline"></div></section>
      <section class="report" id="report"></section>
    </section>
  </main>
  <footer class="footer"><span>RAG Research Navigator · structured evidence, not novelty proof</span><span>counts: data/knowledge/build_summary.json</span></footer>
</div>
<div class="loading" id="loading" aria-hidden="true"><div class="loading-card"><div class="loading-kicker">building research position</div><h2>Following the graph…</h2><div class="loading-steps"><div class="loading-step" data-step="0"><i></i>Understanding idea</div><div class="loading-step" data-step="1"><i></i>Mapping concepts</div><div class="loading-step" data-step="2"><i></i>Exploring research</div><div class="loading-step" data-step="3"><i></i>Traversing relationships</div><div class="loading-step" data-step="4"><i></i>Collecting evidence</div></div></div></div>
<script>
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const TYPE_COLORS = { Paper: "#72b8f6", Method: "#62e1c8", ResearchProblem: "#f28f83", Dataset: "#e7ba72", Metric: "#b8d9a4", Architecture: "#c49cff", Limitation: "#ef9b6c", ResearchDirection: "#c49cff", Concept: "#89c8ce", ApplicationDomain: "#d9a7e8", Author: "#5f727b" };
  const TYPE_LABELS = { Paper: "Paper", Method: "Method", ResearchProblem: "Problem", Dataset: "Dataset", Metric: "Metric", Architecture: "Architecture", Limitation: "Limitation", ResearchDirection: "Direction", Concept: "Concept", ApplicationDomain: "Domain", Author: "Author" };
  const DEFAULT_TYPES = new Set(["Paper", "Method", "ResearchProblem", "Dataset", "Metric", "Architecture", "Limitation", "ResearchDirection", "Concept", "ApplicationDomain"]);
  const state = { graph: null, report: null, mode: "landscape", selectedId: null, selectedRelationId: null, visibleTypes: new Set(DEFAULT_TYPES), relationType: "all", query: "", zoom: 1, panX: 0, panY: 0, positions: new Map(), drag: null, pan: null, hoverId: null, pathRelationIds: new Set(), pathNodeIds: new Set(), animationRun: 0, filtersOpen: false, expandLevel: 1, layoutKey: "", layoutLabels: new Set(), visibleIds: new Set(), searchMatchIds: new Set(), hoverRelationId: null, moved: false, requestId: 0 };

  function el(tag, className, text) { const value = document.createElement(tag); if (className) value.className = className; if (text !== undefined) value.textContent = String(text); return value; }
  function svgEl(tag, attrs) { const value = document.createElementNS(SVG_NS, tag); Object.entries(attrs || {}).forEach(([key, item]) => value.setAttribute(key, String(item))); return value; }
  function clear(value) { while (value.firstChild) value.removeChild(value.firstChild); }
  function escUrl(raw) { if (typeof raw !== "string" || raw.length > 2000) return null; try { const url = new URL(raw, window.location.origin); return ["http:", "https:"].includes(url.protocol) ? url.href : null; } catch (_) { return null; } }
  function link(raw, label) { const url = escUrl(raw); if (!url) return el("span", "item-meta", label || "Source unavailable"); const value = el("a", "small-link", label || url); value.href = url; value.target = "_blank"; value.rel = "noopener noreferrer"; return value; }
  function entityKey(id) { return id === undefined || id === null || String(id) === "" ? null : String(id); }
  function entityById(id) { const key = entityKey(id); return key ? state.graph?.entityMap.get(key) || null : null; }
  function entityName(id) { return entityById(id)?.name || entityKey(id) || "Unknown entity"; }
  function relationName(value) { return String(value || "").replaceAll("_", " "); }
  function normalizeGraphPayload(payload) { if (!payload || typeof payload !== "object" || !Array.isArray(payload.entities) || !Array.isArray(payload.relationships)) throw new Error("The graph response has an invalid shape."); const seenEntities = new Set(); const entities = payload.entities.filter((item) => item && typeof item === "object" && entityKey(item.id) && item.type && !seenEntities.has(entityKey(item.id)) && seenEntities.add(entityKey(item.id))).map((item) => ({ ...item, id: entityKey(item.id), type: String(item.type), name: String(item.name || item.title || item.id) })); const entityIds = new Set(entities.map((item) => item.id)); const seenRelationships = new Set(); const relationships = payload.relationships.filter((item) => item && typeof item === "object" && entityKey(item.id) && entityKey(item.source) && entityKey(item.target) && entityIds.has(entityKey(item.source)) && entityIds.has(entityKey(item.target)) && !seenRelationships.has(entityKey(item.id)) && seenRelationships.add(entityKey(item.id))).map((item) => ({ ...item, id: entityKey(item.id), source: entityKey(item.source), target: entityKey(item.target), relation: String(item.relation || "RELATED_TO"), source_paper: entityKey(item.source_paper) })); return { ...payload, metadata: payload.metadata && typeof payload.metadata === "object" ? payload.metadata : {}, entities, relationships }; }
  const REPORT_ARRAY_FIELDS = ["matched_concepts", "related_methods", "related_architectures", "related_problems", "related_datasets", "known_limitations", "research_directions", "related_papers", "multi_hop_evidence_paths", "evidence", "recommended_reading_order", "uncertainty_notes"];
  function normalizeReportPayload(payload) { if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("The research response has an invalid shape."); const report = { ...payload }; REPORT_ARRAY_FIELDS.forEach((key) => { report[key] = Array.isArray(report[key]) ? report[key] : []; }); report.matched_concepts = report.matched_concepts.filter((item) => item && typeof item === "object").map((item) => ({ ...item, entity_id: entityKey(item.entity_id) })).filter((item) => item.entity_id); ["related_methods", "related_architectures", "related_problems", "related_datasets", "known_limitations", "research_directions"].forEach((key) => { report[key] = report[key].filter((item) => item && typeof item === "object").map((item) => ({ ...item, id: entityKey(item.id) })).filter((item) => item.id); }); report.related_papers = report.related_papers.filter((item) => item && typeof item === "object").map((item) => ({ ...item, paper_id: entityKey(item.paper_id), title: String(item.title || item.name || "Untitled paper") })).filter((item) => item.paper_id); report.multi_hop_evidence_paths = report.multi_hop_evidence_paths.filter((item) => item && typeof item === "object").map((item) => ({ ...item, nodes: Array.isArray(item.nodes) ? item.nodes.map((node) => String(node)) : [], evidence_ids: Array.isArray(item.evidence_ids) ? item.evidence_ids.map((id) => String(id)) : [], steps: Array.isArray(item.steps) ? item.steps : [] })); report.evidence = report.evidence.filter((item) => item && typeof item === "object").map((item) => ({ ...item, evidence_id: entityKey(item.evidence_id), source: entityKey(item.source), target: entityKey(item.target), relation: String(item.relation || "") })); return report; }
  function setStatus(message, error) { const status = $("status"); status.textContent = message || ""; status.className = error ? "status error" : "status"; }
  function refreshCounter() { $("char-count").textContent = String($("idea").value.length); }
  function neighbors(id) { const result = new Set(); state.graph.relationships.forEach((rel) => { if (rel.source === id) result.add(rel.target); if (rel.target === id) result.add(rel.source); }); return result; }
  function neighborhood(ids, depth = 1, limit = 90) { const result = new Set(ids); let frontier = new Set(ids); for (let level = 0; level < depth && result.size < limit; level += 1) { const next = new Set(); frontier.forEach((id) => neighbors(id).forEach((neighbor) => { if (!result.has(neighbor)) next.add(neighbor); })); [...next].sort((a, b) => (state.graph.degrees.get(b) || 0) - (state.graph.degrees.get(a) || 0)).slice(0, Math.max(0, limit - result.size)).forEach((id) => result.add(id)); frontier = next; } return result; }
  function reportEntityIds() { const result = new Set(); if (!state.report) return result; (state.report.matched_concepts || []).forEach((item) => result.add(item.entity_id)); ["related_methods", "related_architectures", "related_problems", "related_datasets", "known_limitations", "research_directions"].forEach((key) => (state.report[key] || []).forEach((item) => result.add(item.id))); (state.report.related_papers || []).forEach((item) => result.add(item.paper_id)); return result; }
  function pathRelations() { const result = new Set(); (state.report?.multi_hop_evidence_paths || []).forEach((path) => (path.evidence_ids || []).forEach((id) => result.add(id))); return result; }
  function isSelectedRelationEndpoint(id) { const rel = state.selectedRelationId ? state.graph.relationMap.get(state.selectedRelationId) : null; return Boolean(rel && (rel.source === id || rel.target === id)); }
  function queryMatches() { const search = state.query.trim().toLowerCase(); if (!search || !state.graph) return new Set(); return new Set(state.graph.entities.filter((item) => `${item.name || ""} ${item.title || ""} ${item.description || ""} ${item.abstract || ""}`.toLowerCase().includes(search)).map((item) => item.id)); }
  function nodePriority(item) { const degree = state.graph.degrees.get(item.id) || 0; const report = reportEntityIds().has(item.id) ? 90 : 0; const path = state.pathNodeIds.has(item.id) ? 110 : 0; const selected = state.selectedId === item.id ? 180 : 0; const search = state.searchMatchIds.has(item.id) ? 140 : 0; const typeBoost = item.type === "Paper" ? 18 : item.type === "Method" || item.type === "ResearchProblem" ? 24 : item.type === "Concept" ? 16 : 0; return report + path + selected + search + degree * 2 + typeBoost; }
  function displayLabel(item) { if (item.type !== "Paper") { const value = item.name || item.title || item.id; return value.length > 28 ? `${value.slice(0, 27)}…` : value; } const title = item.title || item.name || "Paper"; const acronym = title.match(/\b(?:GraphRAG|KG-RAG|G-Retriever|GRAG|CG-RAG|S-Path-RAG|GNN-RAG|MAGIC|RAG)\b/i); if (acronym && acronym[0].toUpperCase() !== "RAG") return acronym[0]; const words = title.replace(/[,:;!?()[\]{}]/g, " ").split(/\s+/).filter(Boolean).slice(0, 3).join(" "); return words.length > 25 ? `${words.slice(0, 24)}…` : words; }
  function shortMetadata(item) { if (item.type === "Paper") return `${item.year || "Year unavailable"} · ${displayLabel(item)}`; if (item.description) return item.description.length > 100 ? `${item.description.slice(0, 97)}…` : item.description; return TYPE_LABELS[item.type] || item.type; }
  function typeShape(group, type, x, y, radius, color) { let shape; if (type === "Paper") shape = svgEl("circle", { cx: x, cy: y, r: radius }); else if (type === "Method") shape = svgEl("polygon", { points: `${x},${y - radius} ${x + radius},${y} ${x},${y + radius} ${x - radius},${y}` }); else if (type === "ResearchProblem") shape = svgEl("polygon", { points: `${x},${y - radius} ${x + radius},${y + radius * .72} ${x - radius},${y + radius * .72}` }); else if (type === "Dataset") shape = svgEl("rect", { x: x - radius, y: y - radius, width: radius * 2, height: radius * 2, rx: 3 }); else if (type === "Limitation") shape = svgEl("polygon", { points: `${x},${y - radius} ${x + radius},${y + radius} ${x - radius},${y + radius}` }); else shape = svgEl("circle", { cx: x, cy: y, r: radius * .84 }); shape.setAttribute("fill", color); shape.setAttribute("fill-opacity", type === "Paper" ? ".82" : ".8"); shape.setAttribute("stroke", color); shape.classList.add("node-shape"); group.appendChild(shape); }
  function visibleEntities() {
    const all = state.graph.entities.filter((item) => state.visibleTypes.has(item.type));
    state.searchMatchIds = queryMatches();
    let ids;
    const focusIds = reportEntityIds();
    if ((state.mode === "focus" || state.mode === "paper" || state.mode === "method") && (state.selectedId || state.selectedRelationId)) {
      const seeds = state.selectedId ? new Set([state.selectedId]) : new Set();
      if (state.selectedRelationId) { const rel = state.graph.relationMap.get(state.selectedRelationId); if (rel) { seeds.add(rel.source); seeds.add(rel.target); } }
      ids = neighborhood(seeds, state.expandLevel, state.expandLevel > 1 ? 88 : 48);
    } else if (state.mode === "paths") {
      const pathSeeds = new Set(state.pathNodeIds); const chosen = (state.report?.multi_hop_evidence_paths || []).find((item) => item.steps && item.steps.length > 1) || state.report?.multi_hop_evidence_paths?.[0]; if (!pathSeeds.size && chosen) (chosen.evidence_ids || []).forEach((relId) => { const rel = state.graph.relationMap.get(relId); if (rel) { pathSeeds.add(rel.source); pathSeeds.add(rel.target); } }); ids = neighborhood(pathSeeds, 1, 44); pathSeeds.forEach((id) => ids.add(id));
    } else if (state.mode === "position") {
      ids = neighborhood(focusIds, 1, 64);
      focusIds.forEach((id) => ids.add(id));
    } else if (state.searchMatchIds.size) {
      ids = neighborhood(state.searchMatchIds, 1, 68);
      state.searchMatchIds.forEach((id) => ids.add(id));
    } else {
      const candidates = all.filter((item) => item.type !== "Author").sort((a, b) => nodePriority(b) - nodePriority(a) || a.name.localeCompare(b.name)); const overviewLimit = state.zoom < .68 ? 45 : state.zoom > 1.35 ? 124 : 92; const quotas = state.zoom < .68 ? { Paper: 12, Method: 8, ResearchProblem: 6, Dataset: 5, Metric: 3, Architecture: 3, Limitation: 3, ResearchDirection: 3, Concept: 2, ApplicationDomain: 2 } : { Paper: state.zoom > 1.35 ? 42 : 28, Method: 16, ResearchProblem: 8, Dataset: 9, Metric: 6, Architecture: 5, Limitation: 6, ResearchDirection: 6, Concept: 8, ApplicationDomain: 6 }; const overview = []; Object.entries(quotas).forEach(([type, quota]) => { overview.push(...candidates.filter((item) => item.type === type).slice(0, quota)); }); candidates.filter((item) => !overview.includes(item)).slice(0, Math.max(0, overviewLimit - overview.length)).forEach((item) => overview.push(item)); ids = new Set(overview.slice(0, overviewLimit).map((item) => item.id));
    }
    let allowed = all.filter((item) => ids.has(item.id));
    const baseCap = state.mode === "landscape" ? 92 : state.mode === "paths" ? 54 : state.mode === "focus" ? 56 : 68; const detailCap = state.zoom > 1.35 ? baseCap + 28 : baseCap; const zoomCap = state.zoom < .68 ? 45 : detailCap; const cap = Math.min(detailCap, zoomCap);
    if (allowed.length > cap) allowed = allowed.sort((a, b) => nodePriority(b) - nodePriority(a)).slice(0, cap);
    state.visibleIds = new Set(allowed.map((item) => item.id));
    return allowed;
  }
  function activePathSequence() { const sequence = []; const chosen = (state.report?.multi_hop_evidence_paths || []).find((item) => item.steps && item.steps.length > 1) || state.report?.multi_hop_evidence_paths?.[0]; const relationIds = state.pathRelationIds.size ? [...state.pathRelationIds] : chosen?.evidence_ids || []; if (!relationIds.length) return sequence; relationIds.forEach((relId) => { const rel = state.graph.relationMap.get(relId); if (!rel) return; if (!sequence.length) sequence.push(rel.source, rel.target); else if (sequence[sequence.length - 1] === rel.source) sequence.push(rel.target); else if (sequence[sequence.length - 1] === rel.target) sequence.push(rel.source); else sequence.push(rel.source, rel.target); }); return [...new Set(sequence)]; }
  function clusterCenters() { return { Concept: { x: 210, y: 170 }, ResearchProblem: { x: 210, y: 390 }, Method: { x: 500, y: 155 }, Architecture: { x: 790, y: 170 }, Paper: { x: 500, y: 370 }, Dataset: { x: 790, y: 380 }, Metric: { x: 790, y: 575 }, Limitation: { x: 215, y: 610 }, ResearchDirection: { x: 500, y: 585 }, Author: { x: 790, y: 650 }, ApplicationDomain: { x: 650, y: 650 } }; }
  function packCluster(items, center, result) { items.forEach((item, index) => { const ring = Math.floor(index / 9); const count = Math.min(9, items.length - ring * 9); const angle = (index % 9) / Math.max(count, 1) * Math.PI * 2 - Math.PI / 2; const radius = 25 + ring * 33; result.set(item.id, { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius }); }); }
  function layout(nodes) {
    const key = `${state.mode}|${state.selectedId || ""}|${state.selectedRelationId || ""}|${state.query}|${state.expandLevel}|${[...nodes].map((item) => item.id).sort().join(",")}`;
    if (key === state.layoutKey && state.positions.size) { const labelCandidates = nodes.slice().sort((a, b) => nodePriority(b) - nodePriority(a)); const labelLimit = state.zoom > 1.35 ? 28 : state.mode === "landscape" ? 12 : 18; state.layoutLabels = new Set(labelCandidates.slice(0, labelLimit).map((item) => item.id)); return new Map(nodes.map((item) => [item.id, state.positions.get(item.id)])); }
    state.layoutKey = key;
    const result = new Map(); const degree = state.graph.degrees; const center = { x: 500, y: 350 };
    const sequence = state.mode === "paths" ? activePathSequence().filter((id) => nodes.some((item) => item.id === id)) : [];
    if (sequence.length > 1) {
      const span = Math.min(780, Math.max(430, (sequence.length - 1) * 135)); const start = center.x - span / 2;
      sequence.forEach((id, index) => result.set(id, { x: start + index * (span / (sequence.length - 1)), y: center.y }));
      nodes.filter((item) => !result.has(item.id)).sort((a, b) => nodePriority(b) - nodePriority(a)).forEach((item, index) => { const anchor = result.get([...state.pathNodeIds].find((id) => result.has(id))) || center; const angle = (index / Math.max(nodes.length, 1)) * Math.PI * 2; result.set(item.id, { x: anchor.x + Math.cos(angle) * (88 + Math.floor(index / 10) * 30), y: anchor.y + Math.sin(angle) * (88 + Math.floor(index / 10) * 30) }); });
    } else if (state.mode === "focus" || state.mode === "paper" || state.mode === "method" || state.mode === "position" || state.searchMatchIds.size) {
      const selectedRel = state.selectedRelationId ? state.graph.relationMap.get(state.selectedRelationId) : null; const focus = state.selectedId || selectedRel?.source || [...state.searchMatchIds][0] || [...reportEntityIds()][0] || [...state.pathNodeIds][0];
      if (focus && nodes.some((item) => item.id === focus)) {
        result.set(focus, center); const ring = nodes.filter((item) => item.id !== focus).sort((a, b) => nodePriority(b) - nodePriority(a)); ring.forEach((item, index) => { const angle = (index / Math.max(ring.length, 1)) * Math.PI * 2 - Math.PI / 2; const radius = 112 + Math.floor(index / 12) * 78; result.set(item.id, { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius }); });
      }
    }
    if (!result.size) { const centers = clusterCenters(); const types = Object.keys(centers); types.forEach((type) => { const items = nodes.filter((item) => item.type === type).sort((a, b) => nodePriority(b) - nodePriority(a)); if (items.length) packCluster(items, centers[type], result); }); }
    nodes.forEach((item, index) => { if (!result.has(item.id)) result.set(item.id, { x: 500 + Math.cos(index) * 70, y: 350 + Math.sin(index) * 70 }); });
    const labelCandidates = nodes.slice().sort((a, b) => nodePriority(b) - nodePriority(a)); const labelLimit = state.zoom > 1.35 ? 28 : state.mode === "landscape" ? 12 : 18; state.layoutLabels = new Set(labelCandidates.slice(0, labelLimit).map((item) => item.id));
    return result;
  }
  function shouldShowLabel(item) { if (state.selectedId === item.id || isSelectedRelationEndpoint(item.id) || state.hoverId === item.id || state.pathNodeIds.has(item.id) || state.searchMatchIds.has(item.id)) return true; if (state.selectedId && neighbors(state.selectedId).has(item.id)) return state.zoom >= 1.16 || nodePriority(item) > 65; if (state.zoom >= 1.45) return state.layoutLabels.has(item.id); return state.layoutLabels.has(item.id) && state.zoom >= .78; }
  function nodeRadius(item) { if (state.selectedId === item.id || isSelectedRelationEndpoint(item.id)) return 12; if (state.pathNodeIds.has(item.id) || state.searchMatchIds.has(item.id)) return 9.5; if (state.hoverId === item.id) return 10.5; const priority = nodePriority(item); return Math.max(4.5, Math.min(8.2, 4.5 + priority / 45)); }
  function applyTransform(group) { group.setAttribute("transform", `translate(${state.panX} ${state.panY}) scale(${state.zoom})`); }
  function renderGraphMessage(title, detail) { const svg = $("graph-svg"); clear(svg); const group = svgEl("g", {}); const heading = svgEl("text", { x: 500, y: 325, "text-anchor": "middle" }); heading.setAttribute("fill", "#eef5f5"); heading.setAttribute("font-size", "18"); heading.setAttribute("font-weight", "800"); heading.textContent = title; group.appendChild(heading); const copy = svgEl("text", { x: 500, y: 355, "text-anchor": "middle" }); copy.setAttribute("fill", "#91a5ad"); copy.setAttribute("font-size", "12"); copy.textContent = detail || "Try another research idea or change the current graph mode."; group.appendChild(copy); svg.appendChild(group); state.positions = new Map(); state.visibleIds = new Set(); }
  function graphEmptyState() { if (!state.report) return ["No graph data is available.", "The knowledge state could not be projected into the visualization."]; if (state.mode === "paths" && !(state.report.multi_hop_evidence_paths || []).length) return ["No multi-hop paths were identified.", "Relevant research may still be available in the report below."]; if (!(reportEntityIds().size)) return ["No strong graph connections were found.", "Try adding a method, problem, dataset, or research direction."]; if (!(state.report.multi_hop_evidence_paths || []).length) return ["Relevant research was found, but no multi-hop path was identified.", "Inspect the related papers and evidence below."]; if (!(state.report.evidence || []).length) return ["Related research found, but supporting evidence is unavailable.", "Inspect the related papers and limitations below."]; return ["The graph has no visible nodes in this view.", "Use Overview, reset the filters, or expand the current focus."]; }
  function edgePath(source, target, index) { const dx = target.x - source.x; const dy = target.y - source.y; const distance = Math.max(Math.hypot(dx, dy), 1); const bend = Math.min(22, distance * .12) * (index % 2 ? 1 : -1); const cx = (source.x + target.x) / 2 - (dy / distance) * bend; const cy = (source.y + target.y) / 2 + (dx / distance) * bend; return `M ${source.x} ${source.y} Q ${cx} ${cy} ${target.x} ${target.y}`; }
  function renderGraphUnsafe() {
    if (!state.graph) return;
    const svg = $("graph-svg"); clear(svg); const defs = svgEl("defs", {}); const marker = svgEl("marker", { id: "edge-arrow", markerWidth: 7, markerHeight: 7, refX: 6, refY: 3.5, orient: "auto", markerUnits: "strokeWidth" }); marker.appendChild(svgEl("path", { d: "M0,0 L7,3.5 L0,7 z", fill: "#62e1c8" })); defs.appendChild(marker); svg.appendChild(defs);
    const root = svgEl("g", {}); svg.appendChild(root); const nodes = visibleEntities(); if (!nodes.length) { const [title, detail] = graphEmptyState(); renderGraphMessage(title, detail); return; } const nodeMap = new Map(nodes.map((item) => [item.id, item])); const positions = layout(nodes); const relationFilter = state.relationType; const selectedId = state.selectedId; const relationFocusId = selectedId || state.hoverId; const hasFocus = Boolean(selectedId || state.selectedRelationId || state.hoverId || state.mode === "position" || state.mode === "paths" || state.query);
    const clusterLayer = svgEl("g", {}); const edgeLayer = svgEl("g", {}); const nodeLayer = svgEl("g", {}); const labelLayer = svgEl("g", {}); root.appendChild(clusterLayer); root.appendChild(edgeLayer); root.appendChild(nodeLayer); root.appendChild(labelLayer); if (state.mode === "landscape" && !state.query) { const centers = clusterCenters(); Object.entries(centers).forEach(([type, center]) => { if (!nodes.some((item) => item.type === type)) return; const label = svgEl("text", { x: center.x, y: center.y - 62, "text-anchor": "middle" }); label.classList.add("cluster-label"); label.textContent = TYPE_LABELS[type] || type; clusterLayer.appendChild(label); }); }
    const candidates = state.graph.relationships.filter((rel) => nodeMap.has(rel.source) && nodeMap.has(rel.target) && (relationFilter === "all" || rel.relation === relationFilter));
    const edgeScore = (rel) => (state.pathRelationIds.has(rel.id) ? 10000 : 0) + (state.selectedRelationId === rel.id ? 9000 : 0) + ((selectedId && (rel.source === selectedId || rel.target === selectedId)) ? 4000 : 0) + ((state.searchMatchIds.has(rel.source) || state.searchMatchIds.has(rel.target)) ? 1800 : 0) + (state.graph.degrees.get(rel.source) || 0) + (state.graph.degrees.get(rel.target) || 0);
    const edgeLimit = state.mode === "landscape" ? 180 : state.mode === "paths" ? 150 : 220; const relations = candidates.length > edgeLimit ? candidates.slice().sort((a, b) => edgeScore(b) - edgeScore(a)).slice(0, edgeLimit) : candidates;
    relations.forEach((rel, index) => { if (state.mode === "paths" && !state.pathRelationIds.has(rel.id)) return; const source = positions.get(rel.source); const target = positions.get(rel.target); const d = edgePath(source, target, index); const highlighted = state.pathRelationIds.has(rel.id) || state.selectedRelationId === rel.id; const connected = relationFocusId && (rel.source === relationFocusId || rel.target === relationFocusId); const line = svgEl("path", { d }); line.classList.add("edge"); if (highlighted) line.classList.add("highlight"); else if (connected) line.classList.add("hover-highlight"); else if (hasFocus && !connected && !state.pathNodeIds.has(rel.source) && !state.pathNodeIds.has(rel.target) && !state.searchMatchIds.has(rel.source) && !state.searchMatchIds.has(rel.target)) line.classList.add("dimmed"); if (highlighted) line.setAttribute("marker-end", "url(#edge-arrow)"); line.addEventListener("click", (event) => { event.stopPropagation(); selectRelationship(rel.id); }); edgeLayer.appendChild(line); const hit = svgEl("path", { d }); hit.classList.add("edge-hit"); hit.addEventListener("click", (event) => { event.stopPropagation(); selectRelationship(rel.id); }); const mid = { x: (source.x + target.x) / 2, y: (source.y + target.y) / 2 }; const relationLabel = svgEl("text", { x: mid.x, y: mid.y - 7, "text-anchor": "middle" }); relationLabel.classList.add("edge-label"); const showRelationLabel = state.selectedRelationId === rel.id || (highlighted && state.pathRelationIds.size <= 10); if (!showRelationLabel) relationLabel.classList.add("edge-label-hidden"); relationLabel.textContent = relationName(rel.relation); hit.addEventListener("mouseenter", (event) => { state.hoverRelationId = rel.id; relationLabel.classList.remove("edge-label-hidden"); showTooltip(event, `${relationName(rel.relation)} · ${displayLabel(state.graph.entityMap.get(rel.source))} → ${displayLabel(state.graph.entityMap.get(rel.target))}`); }); hit.addEventListener("mouseleave", () => { state.hoverRelationId = null; if (!showRelationLabel) relationLabel.classList.add("edge-label-hidden"); hideTooltip(); }); edgeLayer.appendChild(hit); labelLayer.appendChild(relationLabel); });
    nodes.forEach((item) => { const pos = positions.get(item.id); const group = svgEl("g", { tabindex: 0, role: "button", "aria-label": `${TYPE_LABELS[item.type] || item.type}: ${item.name}` }); group.classList.add("node-group"); const active = selectedId === item.id || state.hoverId === item.id || isSelectedRelationEndpoint(item.id) || state.pathNodeIds.has(item.id) || state.searchMatchIds.has(item.id) || (state.mode === "position" && reportEntityIds().has(item.id)); const direct = relationFocusId && neighbors(relationFocusId).has(item.id); const dim = hasFocus && !active && !direct && !state.searchMatchIds.has(item.id); if (selectedId === item.id) group.classList.add("selected"); else if (state.hoverId === item.id) group.classList.add("hovered"); else if (isSelectedRelationEndpoint(item.id)) group.classList.add("relation-focus"); else if (state.pathNodeIds.has(item.id) || state.searchMatchIds.has(item.id)) group.classList.add("highlight"); if (dim) group.classList.add("dimmed"); typeShape(group, item.type, pos.x, pos.y, nodeRadius(item), TYPE_COLORS[item.type] || "#91a5ad"); if (shouldShowLabel(item)) { const text = svgEl("text", { x: pos.x + (pos.x > 820 ? -12 : 12), y: pos.y + 3, "text-anchor": pos.x > 820 ? "end" : "start" }); text.classList.add("node-label"); text.style.pointerEvents = "auto"; text.textContent = displayLabel(item); text.addEventListener("click", (event) => { event.stopPropagation(); selectNode(item.id); }); labelLayer.appendChild(text); } const hoverText = svgEl("text", { x: pos.x + (pos.x > 820 ? -12 : 12), y: pos.y + 3, "text-anchor": pos.x > 820 ? "end" : "start" }); hoverText.classList.add("node-hover-label"); hoverText.textContent = displayLabel(item); group.appendChild(hoverText); const title = svgEl("title", {}); title.textContent = `${TYPE_LABELS[item.type] || item.type}: ${item.name}`; group.appendChild(title); group.addEventListener("click", (event) => { event.stopPropagation(); if (state.moved) { state.moved = false; return; } selectNode(item.id); }); group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(item.id); } }); group.addEventListener("pointerdown", (event) => startDrag(event, item.id)); group.addEventListener("mouseenter", (event) => hoverNode(event, item)); group.addEventListener("mousemove", (event) => positionTooltip(event)); group.addEventListener("mouseleave", () => leaveNode(item)); nodeLayer.appendChild(group); });
    applyTransform(root); state.positions = positions;
  }
  function renderGraph() { try { renderGraphUnsafe(); } catch (error) { console.error("Graph rendering failed", error); const [title, detail] = ["The graph could not be rendered.", "The research result is still available below. Try Overview or Reset."]; renderGraphMessage(title, detail); setStatus("The result loaded, but the graph view encountered a rendering error.", true); } }
  function positionTooltip(event) { const tip = $("tooltip"); const stage = $("graph-svg").parentElement.getBoundingClientRect(); tip.style.left = `${Math.max(10, Math.min(event.clientX - stage.left + 14, stage.width - 300))}px`; tip.style.top = `${Math.max(event.clientY - stage.top - 12, 12)}px`; }
  function showTooltip(event, value) { const tip = $("tooltip"); clear(tip); tip.appendChild(el("div", "tooltip-copy", value)); positionTooltip(event); tip.classList.add("visible"); }
  function showEntityTooltip(event, item) { const tip = $("tooltip"); clear(tip); tip.appendChild(el("div", "tooltip-type", TYPE_LABELS[item.type] || item.type)); tip.appendChild(el("div", "tooltip-title", item.type === "Paper" ? (item.title || item.name) : item.name)); tip.appendChild(el("div", "tooltip-meta", shortMetadata(item))); if (item.type === "Paper") { const authors = graphNeighbors(item.id).filter((candidate) => candidate.type === "Author").slice(0, 3).map((candidate) => candidate.name); if (authors.length) tip.appendChild(el("div", "tooltip-meta", authors.join(", "))); } positionTooltip(event); tip.classList.add("visible"); }
  function hoverNode(event, item) { state.hoverId = item.id; renderGraph(); showEntityTooltip(event, item); }
  function leaveNode(item) { if (state.hoverId === item.id) { state.hoverId = null; renderGraph(); } hideTooltip(); }
  function hideTooltip() { $("tooltip").classList.remove("visible"); }
  function startDrag(event, id) { state.moved = false; state.drag = { id, x: event.clientX, y: event.clientY, original: state.positions.get(id) }; event.currentTarget.setPointerCapture?.(event.pointerId); }
  function startPan(event) { if (event.target.closest?.(".node-group, .edge, .edge-hit")) return; state.moved = false; state.pan = { x: event.clientX, y: event.clientY, originalX: state.panX, originalY: state.panY }; $("graph-svg").setPointerCapture?.(event.pointerId); }
  function moveDrag(event) { if (state.drag) { const dx = (event.clientX - state.drag.x) / state.zoom; const dy = (event.clientY - state.drag.y) / state.zoom; if (Math.abs(dx) > 2 || Math.abs(dy) > 2) state.moved = true; const original = state.drag.original || { x: 500, y: 350 }; state.positions.set(state.drag.id, { x: original.x + dx, y: original.y + dy }); renderGraph(); } else if (state.pan) { if (Math.abs(event.clientX - state.pan.x) > 2 || Math.abs(event.clientY - state.pan.y) > 2) state.moved = true; state.panX = state.pan.originalX + (event.clientX - state.pan.x); state.panY = state.pan.originalY + (event.clientY - state.pan.y); renderGraph(); } }
  function endDrag() { state.drag = null; state.pan = null; }
  function selectNode(id) { state.animationRun += 1; state.selectedId = id; state.selectedRelationId = null; state.pathRelationIds = new Set(); state.pathNodeIds = new Set(); state.mode = "focus"; state.expandLevel = 1; state.layoutKey = ""; updateModes(); renderGraph(); renderInspector(); const item = state.graph.entityMap.get(id); if (item?.type === "Paper") focusTimeline(id); }
  function selectRelationship(id) { state.animationRun += 1; state.selectedRelationId = id; state.selectedId = null; state.pathRelationIds = new Set(); state.pathNodeIds = new Set(); state.mode = "focus"; state.expandLevel = 1; state.layoutKey = ""; renderGraph(); renderRelationshipInspector(id); }
  function expandSelected() { if (!state.selectedId && !state.selectedRelationId) return; state.expandLevel = Math.min(3, state.expandLevel + 1); state.layoutKey = ""; renderGraph(); renderInspector(); }
  function focusTimeline(id) { document.querySelectorAll(".timeline-paper").forEach((button) => button.classList.toggle("selected", button.dataset.id === id)); }
  function updateModes() { document.querySelectorAll("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === state.mode)); }
  function graphNeighbors(id) { return [...neighbors(id)].map((neighbor) => state.graph.entityMap.get(neighbor)).filter(Boolean); }
  function createPills(items, fallback) { const wrap = el("div", "pill-list"); if (!items.length) wrap.appendChild(el("span", "item-meta", fallback || "None recorded")); items.forEach((item) => wrap.appendChild(el("span", "pill", item.name || item.title || item))); return wrap; }
  function renderInspector() { const panel = $("inspector"); clear(panel); if (!state.selectedId) { panel.appendChild(el("div", "inspector-empty", "Select a node to inspect its evidence and connected research.")); return; } const item = entityById(state.selectedId); if (!item) { renderGraphMessage("This node is no longer available.", "The graph selection did not match a knowledge-state entity."); return; } const type = el("div", "inspector-type", TYPE_LABELS[item.type] || item.type); panel.appendChild(type); panel.appendChild(el("h3", "", item.name)); const expand = el("button", "ghost inspector-expand", state.expandLevel >= 3 ? "Connections fully expanded" : `Expand connections · level ${state.expandLevel}`); expand.type = "button"; expand.disabled = state.expandLevel >= 3; expand.addEventListener("click", expandSelected); panel.appendChild(expand); if (item.description) panel.appendChild(el("p", "inspector-copy", item.description)); if (item.abstract) { const section = el("div", "inspector-section"); section.appendChild(el("h4", "", "abstract")); section.appendChild(el("p", "inspector-copy", item.abstract)); panel.appendChild(section); } const meta = el("div", "inspector-section"); meta.appendChild(el("h4", "", "at a glance")); [["year", item.year], ["citation count", item.citation_count], ["corpus index", item.corpus_index]].forEach(([label, value]) => { if (value !== undefined && value !== null) { const row = el("div", "meta-row"); row.appendChild(el("span", "", label)); row.appendChild(el("strong", "", value)); meta.appendChild(row); } }); if (item.source_url) { meta.appendChild(link(item.source_url, "Open source paper ↗")); } panel.appendChild(meta); const connected = graphNeighbors(item.id); const related = el("div", "inspector-section"); related.appendChild(el("h4", "", `connected research · ${connected.length}`)); related.appendChild(createPills(connected.slice(0, 18), "No connected entities")); panel.appendChild(related); const rels = state.graph.relationships.filter((rel) => rel.source === item.id || rel.target === item.id).slice(0, 18); const evidenceSection = el("div", "inspector-section"); evidenceSection.appendChild(el("h4", "", "relationships")); const list = el("div", "list"); rels.forEach((rel) => { const row = el("button", "ghost", `${relationName(rel.relation)} · ${rel.source === item.id ? entityName(rel.target) : entityName(rel.source)}`); row.type = "button"; row.style.textAlign = "left"; row.style.width = "100%"; row.addEventListener("click", () => selectRelationship(rel.id)); list.appendChild(row); }); evidenceSection.appendChild(list); panel.appendChild(evidenceSection); }
  function renderRelationshipInspector(id) { const panel = $("inspector"); clear(panel); const rel = state.graph.relationMap.get(entityKey(id)); if (!rel) { panel.appendChild(el("div", "inspector-empty", "This relationship is no longer available.")); return; } panel.appendChild(el("div", "inspector-type", "relationship evidence")); panel.appendChild(el("h3", "", relationName(rel.relation))); const route = el("div", "inspector-section"); route.appendChild(el("h4", "", "connection")); route.appendChild(el("div", "meta-row", `${entityName(rel.source)} → ${entityName(rel.target)}`)); panel.appendChild(route); const evidence = el("div", "inspector-section"); const card = el("div", "evidence-card"); card.appendChild(el("div", "evidence-card-title", "why this connection exists")); card.appendChild(el("div", "evidence-card-quote", rel.evidence || "The relationship is recorded in the knowledge state without a quote.")); card.appendChild(el("div", "evidence-card-source", `${rel.evidence_type || "knowledge-state evidence"} · ${rel.source_location || "source metadata"}`)); evidence.appendChild(card); panel.appendChild(evidence); if (rel.source_paper) { const source = state.graph.entityMap.get(rel.source_paper); if (source) { const section = el("div", "inspector-section"); section.appendChild(el("h4", "", "source paper")); section.appendChild(link(source.source_url, source.name)); panel.appendChild(section); } } const back = el("button", "ghost", "← inspect source node"); back.type = "button"; back.addEventListener("click", () => selectNode(rel.source)); panel.appendChild(back); }
  function renderLegend() { const legend = $("legend"); clear(legend); ["Paper", "Method", "ResearchProblem", "Dataset", "Limitation", "ResearchDirection", "Concept"].forEach((type) => { const item = el("span", "legend-item"); const dot = el("i", `legend-dot ${type === "Paper" ? "paper" : type === "Method" ? "method" : type === "ResearchProblem" ? "problem" : type === "ResearchDirection" ? "direction" : ""}`); dot.style.background = TYPE_COLORS[type]; item.appendChild(dot); item.appendChild(el("span", "", TYPE_LABELS[type])); legend.appendChild(item); }); }
  function renderFilters() { const types = $("type-filters"); clear(types); const typeCounts = state.graph.metadata.entity_type_counts || {}; Object.keys(TYPE_LABELS).forEach((type) => { const label = el("label", "filter-label"); const input = document.createElement("input"); input.type = "checkbox"; input.checked = state.visibleTypes.has(type); input.dataset.type = type; input.addEventListener("change", () => { input.checked ? state.visibleTypes.add(type) : state.visibleTypes.delete(type); renderGraph(); }); label.appendChild(input); label.appendChild(el("span", "", `${TYPE_LABELS[type]} (${typeCounts[type] || 0})`)); types.appendChild(label); }); const select = $("relation-filter"); clear(select); const all = document.createElement("option"); all.value = "all"; all.textContent = "All relationship types"; select.appendChild(all); Object.keys(state.graph.metadata.relationship_type_counts || {}).sort().forEach((type) => { const option = document.createElement("option"); option.value = type; option.textContent = relationName(type); select.appendChild(option); }); select.value = state.relationType; select.addEventListener("change", () => { state.relationType = select.value; renderGraph(); }); }
  function renderTimeline() { const timeline = $("timeline"); clear(timeline); const papers = state.graph.entities.filter((item) => item.type === "Paper" && item.year).sort((a, b) => a.year - b.year || a.name.localeCompare(b.name)); const years = new Map(); papers.forEach((paper) => { if (!years.has(paper.year)) years.set(paper.year, []); years.get(paper.year).push(paper); }); years.forEach((items, year) => { const column = el("div", "timeline-year"); column.appendChild(el("div", "timeline-year-label", year)); const dots = el("div", "timeline-papers"); items.forEach((paper) => { const dot = el("button", "timeline-paper", ""); dot.type = "button"; dot.dataset.id = paper.id; dot.setAttribute("aria-label", `Focus paper ${paper.name}`); dot.title = paper.name; dot.addEventListener("click", () => { state.mode = "paper"; state.selectedId = paper.id; state.selectedRelationId = null; updateModes(); renderGraph(); renderInspector(); focusTimeline(paper.id); $("analysis").scrollIntoView({ behavior: "smooth", block: "start" }); }); dots.appendChild(dot); }); column.appendChild(dots); timeline.appendChild(column); }); }
  function addReportDetail(report, title, key, formatter) { const details = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = title; details.appendChild(summary); const content = el("div", "report-content"); const items = Array.isArray(report[key]) ? report[key] : []; if (items.length) formatter(content, items); else content.appendChild(el("div", "item-meta", "No records available for this result.")); details.appendChild(content); return details; }
  function renderReport() { const report = state.report; const target = $("report"); clear(target); const left = el("section", "panel"); const right = el("section", "panel"); left.appendChild(el("div", "eyebrow", "evidence-backed report")); left.appendChild(el("h3", "", "What the graph found")); left.appendChild(addReportDetail(report, "Matched concepts", "matched_concepts", (box, items) => { box.appendChild(createPills(items)); })); left.appendChild(addReportDetail(report, "Related methods", "related_methods", (box, items) => { box.appendChild(createPills(items)); })); left.appendChild(addReportDetail(report, "Related papers", "related_papers", (box, items) => { items.slice(0, 8).forEach((item) => { const row = el("div", "report-path"); row.appendChild(link(item.source_url, item.title)); row.appendChild(el("div", "item-meta", `${item.year || "n.d."} · graph score ${item.score}`)); const focus = el("button", "ghost", "Focus in graph"); focus.type = "button"; focus.addEventListener("click", () => { state.mode = "paper"; state.selectedId = item.paper_id; state.selectedRelationId = null; updateModes(); renderGraph(); renderInspector(); focusTimeline(item.paper_id); $("graph-svg").scrollIntoView({ behavior: "smooth", block: "center" }); }); row.appendChild(focus); box.appendChild(row); }); })); left.appendChild(addReportDetail(report, "Multi-hop paths", "multi_hop_evidence_paths", (box, items) => { items.slice(0, 8).forEach((item) => { const row = el("button", "report-path", item.nodes.join(" → ")); row.type = "button"; row.addEventListener("click", () => { state.mode = "paths"; state.pathRelationIds = new Set(item.evidence_ids || []); state.pathNodeIds = new Set(); (item.evidence_ids || []).forEach((relId) => { const rel = state.graph.relationMap.get(relId); if (rel) { state.pathNodeIds.add(rel.source); state.pathNodeIds.add(rel.target); } }); updateModes(); renderGraph(); }); box.appendChild(row); }); })); left.appendChild(addReportDetail(report, "Evidence ledger", "evidence", (box, items) => { items.slice(0, 12).forEach((item) => { const row = el("button", "report-path", `${relationName(item.relation)} · ${entityName(item.source)} → ${entityName(item.target)}`); row.type = "button"; row.appendChild(el("div", "item-meta", `${item.evidence_type || "knowledge-state evidence"} · ${item.source_location || "source metadata"}`)); row.addEventListener("click", () => { selectRelationship(item.evidence_id); $("graph-svg").scrollIntoView({ behavior: "smooth", block: "center" }); }); box.appendChild(row); }); })); right.appendChild(el("div", "eyebrow", "research guidance")); right.appendChild(el("h3", "", "Where to look next")); right.appendChild(addReportDetail(report, "Known limitations", "known_limitations", (box, items) => { box.appendChild(createPills(items)); })); right.appendChild(addReportDetail(report, "Research directions", "research_directions", (box, items) => { box.appendChild(createPills(items)); })); right.appendChild(addReportDetail(report, "Recommended reading", "recommended_reading_order", (box, items) => { items.forEach((item) => { const row = el("div", "report-path", `${item.order}. ${item.title}`); row.appendChild(el("div", "item-meta", item.reason)); box.appendChild(row); }); })); right.appendChild(addReportDetail(report, "Uncertainty notes", "uncertainty_notes", (box, items) => { items.forEach((item) => box.appendChild(el("div", "report-path", item))); })); target.appendChild(left); target.appendChild(right); }
  function setPathHighlight(stepCount) { const paths = state.report?.multi_hop_evidence_paths || []; const chosen = paths.find((item) => item.steps && item.steps.length > 1) || paths[0]; if (!chosen) return; const ids = chosen.evidence_ids || []; state.pathRelationIds = new Set(ids.slice(0, stepCount)); state.pathNodeIds = new Set(); state.pathRelationIds.forEach((relId) => { const rel = state.graph.relationMap.get(relId); if (rel) { state.pathNodeIds.add(rel.source); state.pathNodeIds.add(rel.target); } }); renderGraph(); }
  function animatePath() { const run = ++state.animationRun; const paths = state.report?.multi_hop_evidence_paths || []; const chosen = paths.find((item) => item.steps && item.steps.length > 1) || paths[0]; if (!chosen || !chosen.evidence_ids?.length) return; state.mode = "paths"; updateModes(); state.pathRelationIds = new Set(); state.pathNodeIds = new Set(); setPathHighlight(1); if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) { setPathHighlight(chosen.evidence_ids.length); return; } chosen.evidence_ids.slice(1).forEach((_, index) => { window.setTimeout(() => { if (run !== state.animationRun) return; setPathHighlight(index + 2); }, 480 * (index + 1)); }); }
  function setZoom(value) { state.zoom = Math.max(.45, Math.min(2.2, value)); renderGraph(); }
  function resetGraph() { state.animationRun += 1; state.requestId += 1; state.zoom = 1; state.panX = 0; state.panY = 0; state.positions = new Map(); state.layoutKey = ""; state.selectedId = null; state.selectedRelationId = null; state.pathRelationIds = new Set(); state.pathNodeIds = new Set(); state.mode = "landscape"; state.expandLevel = 1; state.query = ""; state.searchMatchIds = new Set(); $("node-search").value = ""; updateModes(); renderGraph(); renderInspector(); }
  function focusSelected() { if (!state.selectedId) return; state.mode = "focus"; state.expandLevel = 1; state.layoutKey = ""; updateModes(); renderGraph(); renderInspector(); }
  function analyze() { const idea = $("idea").value.trim(); if (idea.length < 12) { setStatus("Add a little more detail so the graph can find a meaningful connection.", true); return; } if (idea.length > 5000) { setStatus("That idea is longer than the supported input limit.", true); return; } const requestId = ++state.requestId; const loading = $("loading"); loading.classList.add("visible"); loading.setAttribute("aria-hidden", "false"); const steps = [...document.querySelectorAll(".loading-step")]; steps.forEach((step) => step.classList.remove("active", "done")); const timers = steps.map((step, index) => window.setTimeout(() => { steps.forEach((candidate, candidateIndex) => { if (candidateIndex < index) candidate.classList.add("done"); }); step.classList.add("active"); }, 130 * index)); fetch("/api/reason", { method: "POST", headers: { "Content-Type": "application/json", Accept: "application/json" }, body: JSON.stringify({ idea }), cache: "no-store" }).then(async (response) => { const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "The research graph could not process that idea."); return normalizeReportPayload(payload); }).then((report) => { if (requestId !== state.requestId) return; timers.forEach(window.clearTimeout); state.report = report; state.mode = "position"; state.selectedId = null; state.selectedRelationId = null; state.pathRelationIds = new Set(); state.pathNodeIds = new Set(); state.reportEntityIds = reportEntityIds(); $("analysis-title").textContent = "Your idea, in context."; $("analysis-summary").textContent = `${report.related_papers.length} related papers · ${report.multi_hop_evidence_paths.length} evidence paths · ${report.evidence.length} evidence ledger entries`; $("analysis").classList.add("visible"); renderReport(); renderGraph(); renderInspector(); $("analysis").scrollIntoView({ behavior: "smooth", block: "start" }); window.setTimeout(animatePath, 420); const resultMessage = !reportEntityIds().size ? "No strong graph connections were found for this research idea." : !(report.multi_hop_evidence_paths || []).length ? "Relevant research was found, but no multi-hop relationship path was identified." : !(report.evidence || []).length ? "Related research found, but supporting evidence is unavailable." : "Research position ready."; setStatus(resultMessage); }).catch((error) => { if (requestId === state.requestId) setStatus(error instanceof Error ? error.message : "The request could not be completed.", true); }).finally(() => { if (requestId === state.requestId) window.setTimeout(() => { loading.classList.remove("visible"); loading.setAttribute("aria-hidden", "true"); }, 300); }); }
  async function init() { try { const [graphResponse, healthResponse] = await Promise.all([fetch("/api/graph", { cache: "no-store" }), fetch("/api/health", { cache: "no-store" })]); if (!graphResponse.ok || !healthResponse.ok) throw new Error("Knowledge state unavailable."); const rawGraph = await graphResponse.json(); const health = await healthResponse.json(); const graph = normalizeGraphPayload(rawGraph); const degrees = new Map(); graph.relationships.forEach((rel) => { degrees.set(rel.source, (degrees.get(rel.source) || 0) + 1); degrees.set(rel.target, (degrees.get(rel.target) || 0) + 1); }); state.graph = { ...graph, entityMap: new Map(graph.entities.map((item) => [item.id, item])), relationMap: new Map(graph.relationships.map((item) => [item.id, item])), degrees }; $("stat-papers").textContent = health.corpus_size; $("stat-entities").textContent = health.entity_count; $("stat-relationships").textContent = health.relationship_count; renderMiniGraph(); renderLegend(); renderFilters(); renderTimeline(); renderGraph(); setStatus("Select a node after analysis to inspect the atlas."); } catch (_) { setStatus("The knowledge state could not be loaded. Restart the local application and try again.", true); } }
  function renderMiniGraph() { const svg = $("mini-graph"); clear(svg); const nodes = state.graph.entities.filter((item) => item.type !== "Author").sort((a, b) => (state.graph.degrees.get(b.id) || 0) - (state.graph.degrees.get(a.id) || 0)).slice(0, 58); const ids = new Set(nodes.map((item) => item.id)); const positions = new Map(nodes.map((item, index) => { const ring = Math.floor(index / 18); const angle = (index % 18) / 18 * Math.PI * 2; return [item.id, { x: 350 + Math.cos(angle) * (55 + ring * 48), y: 220 + Math.sin(angle) * (55 + ring * 38) }]; })); const lines = svgEl("g", {}); const dots = svgEl("g", {}); state.graph.relationships.slice(0, 240).forEach((rel) => { if (!ids.has(rel.source) || !ids.has(rel.target)) return; const a = positions.get(rel.source); const b = positions.get(rel.target); const line = svgEl("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: TYPE_COLORS[state.graph.entityMap.get(rel.source)?.type] || "#466", "stroke-opacity": ".22", "stroke-width": "1" }); lines.appendChild(line); }); nodes.forEach((item) => { const pos = positions.get(item.id); const circle = svgEl("circle", { cx: pos.x, cy: pos.y, r: item.type === "Paper" ? 4 : 5, fill: TYPE_COLORS[item.type] || "#91a5ad", "fill-opacity": ".62" }); dots.appendChild(circle); }); svg.appendChild(lines); svg.appendChild(dots); }
  $("idea").addEventListener("input", refreshCounter); $("analyze").addEventListener("click", analyze); $("zoom-in").addEventListener("click", () => setZoom(state.zoom + .15)); $("zoom-out").addEventListener("click", () => setZoom(state.zoom - .15)); $("fit").addEventListener("click", () => { state.zoom = .9; state.panX = 0; state.panY = 0; renderGraph(); }); $("reset").addEventListener("click", resetGraph); $("focus").addEventListener("click", focusSelected); $("filters-toggle").addEventListener("click", () => { state.filtersOpen = !state.filtersOpen; $("filters").classList.toggle("visible", state.filtersOpen); }); $("reset-filters").addEventListener("click", () => { state.visibleTypes = new Set(DEFAULT_TYPES); state.relationType = "all"; renderFilters(); renderGraph(); }); $("timeline-toggle").addEventListener("click", () => { $("timeline-panel").classList.toggle("hidden"); }); $("node-search").addEventListener("input", (event) => { state.query = event.target.value; state.layoutKey = ""; if (state.query.trim()) { state.zoom = 1.12; state.panX = 0; state.panY = 0; } renderGraph(); }); document.querySelectorAll("[data-mode]").forEach((button) => button.addEventListener("click", () => { state.animationRun += 1; state.mode = button.dataset.mode; state.layoutKey = ""; if (state.mode === "paths") { state.pathRelationIds = pathRelations(); state.pathNodeIds = new Set(); state.pathRelationIds.forEach((relId) => { const rel = state.graph.relationMap.get(relId); if (rel) { state.pathNodeIds.add(rel.source); state.pathNodeIds.add(rel.target); } }); } updateModes(); renderGraph(); })); $("graph-svg").addEventListener("pointerdown", startPan); $("graph-svg").addEventListener("pointermove", moveDrag); $("graph-svg").addEventListener("pointerup", endDrag); $("graph-svg").addEventListener("pointercancel", endDrag); $("graph-svg").addEventListener("pointerleave", endDrag); $("graph-svg").addEventListener("click", () => { if (state.moved) { state.moved = false; return; } if (!state.drag) { state.animationRun += 1; state.selectedId = null; state.selectedRelationId = null; state.mode = "landscape"; state.expandLevel = 1; state.layoutKey = ""; renderGraph(); renderInspector(); } }); $("graph-svg").addEventListener("wheel", (event) => { event.preventDefault(); setZoom(state.zoom + (event.deltaY < 0 ? .08 : -.08)); }, { passive: false }); refreshCounter(); init();
})();
</script>
</body>
</html>'''


class SecureHandler(BaseHTTPRequestHandler):
    server_version = "RAGResearchOnboarding/1.0"
    sys_version = ""
    reasoner: GraphReasoner | None = None

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _headers(self, content_type: str, length: int, cache_control: str = "no-store") -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'self'")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _path(self) -> str:
        return urlsplit(self.path).path

    def do_GET(self) -> None:
        path = self._path()
        if path == "/":
            self._send_bytes(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            state = self.reasoner.state if self.reasoner else {}
            metadata = state.get("metadata", {})
            self._send_json(200, {"status": "ok", "corpus_size": metadata.get("corpus_size", 0), "entity_count": len(state.get("entities", [])), "relationship_count": len(state.get("relationships", []))})
            return
        if path == "/api/graph":
            self._send_json(200, public_graph(self.reasoner.state if self.reasoner else {}))
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self._path() != "/api/reason":
            self._send_json(404, {"error": "Not found."})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            self._send_json(415, {"error": "Content-Type must be application/json."})
            return
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header is not None else -1
        except ValueError:
            length = -1
        if length < 0:
            self._send_json(411, {"error": "Content-Length is required."})
            return
        if length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body is too large."})
            return
        raw = self.rfile.read(length)
        try:
            idea = validate_payload(raw)
            if self.reasoner is None:
                raise RuntimeError("Reasoner unavailable")
            self._send_json(200, self.reasoner.reason(idea, limit=8))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            print("Reasoning request failed.", file=sys.stderr)
            self._send_json(500, {"error": "The research graph could not process that request."})


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    handler = SecureHandler
    handler.reasoner = load_reasoner()
    server = ThreadingHTTPServer((host, port), handler)
    print(f"RAG Research Onboarding UI listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()

"""YUBI v3.0 session report — local behavioral analysis plus 1–3 targeted v3.0 passes.

``SESSION_REPORT_PASSES``:

1. compose — one generation turns the draft and measured analysis into the final brief.
2. analyze + compose — a compact JSON findings pass over the measurements, then compose.
3. + evidence audit — checks the brief against the data and applies targeted fixes.

Only compose produces a full-length document; analyze and audit return small JSON
payloads, so cost scales with the evidence rather than with repeated rewrites.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from backend.ai_client import AIClient, AIUnavailable
from backend.config import Settings
from backend.facs import CUES, EMOTION_LABELS
from backend.insight_engine import format_clock

ANALYSIS_HEADING = "## Behavioral analysis"
CHRONOLOGY_HEADING = "## Event chronology"

DISCLAIMER = (
    "> Behavioral signals are measured relative to each subject's own baseline. They are "
    "observational indicators — not evidence of deception, intent, or emotional state."
)

SYSTEM_INSTRUCTION = (
    "You are YUBI v3.0, the intelligence layer of YUBI Supervision — a real-time vision and "
    "behavioral-analysis platform. You turn session logs and on-device measurements (objects, "
    "FACS action units, micro-expression episodes, blinks, gaze, head gestures, hand-to-face "
    "contact, an experimental camera pulse estimate) into clear operational reports.\n"
    "- Report observable behavior. Never claim to know thoughts, intent, truthfulness, guilt, "
    "or any diagnosis, and never say someone is lying. Use calibrated language.\n"
    "- Every claim must trace to the supplied data (episode ids, timestamps, counts).\n"
    "- Mouth-region episodes during speech are weak evidence; say so when relevant.\n"
    "- Never mention model vendors; refer only to YUBI v3.0 and YUBI Supervision."
)

ANALYZE_PROMPT = (
    "Analyze these session measurements. Identify the patterns a careful human reviewer would "
    "miss: clusters of micro-expressions, incongruent signals (masking, leakage, suppression), "
    "shifts from baseline over time (blink rate, gaze, pulse, valence), and moments that "
    "deserve a second look.\n"
    'Return JSON: {"executive_summary": str (<= 90 words), "findings": [{"title": str, '
    '"detail": str, "evidence": [episode ids or mm:ss times], "confidence": 0.0-1.0}], '
    '"timeline_highlights": [{"at": seconds, "note": str}], "patterns": [str], '
    '"follow_ups": [str (questions or moments worth reviewing)], "caveats": [str]}. '
    "At most 7 findings and 10 highlights. Omit anything the data does not support."
)

COMPOSE_PROMPT = (
    "Write the final session report in Markdown from the draft below{findings_clause}.\n"
    "Structure: title block, **Executive summary**, **Key findings** (each with its evidence "
    "times / episode ids), **Timeline highlights**, then the draft's measured sections "
    "(keep every metric, table row, and the behavioral-analysis tables; collapse duplicate "
    "chronology rows), **Moments worth reviewing**, and **Method & limitations**. "
    "Keep the behavioral disclaimer. Footer credits YUBI Supervision Node and YUBI v3.0 only. "
    "Return the Markdown document only."
)

AUDIT_PROMPT = (
    "Audit this report against the measurements. Find statements that are unsupported, "
    "overstated (certainty about intent, deception, or emotion), or contradict the data. "
    'Return JSON: {"fixes": [{"find": exact substring of the report, "replace": corrected '
    'text, "reason": str}]}. At most 8 fixes; return an empty list if the report is sound.'
)

PASS_LABELS = {
    "analyze": "Analyzing behavioral evidence",
    "compose": "Composing intelligence brief",
    "audit": "Auditing claims against evidence",
}

_PASS_PLANS: dict[int, tuple[str, ...]] = {
    1: ("compose",),
    2: ("analyze", "compose"),
    3: ("analyze", "compose", "audit"),
}

_MAX_EVIDENCE_EPISODES = 150
_MAX_TIMELINE_POINTS = 60
_MAX_MOMENTS = 25


def pass_plan(passes: int) -> tuple[str, ...]:
    return _PASS_PLANS[max(1, min(3, int(passes)))]


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole > 0 else "—"


def _dominant_affect(subject: dict[str, Any]) -> str:
    times = (subject.get("affect") or {}).get("time_by_emotion_sec") or {}
    total = sum(times.values())
    if not total:
        return "—"
    ranked = sorted(times.items(), key=lambda item: -item[1])[:3]
    return " · ".join(f"{EMOTION_LABELS.get(k, k)} {_pct(v, total)}" for k, v in ranked)


def _pulse_text(subject: dict[str, Any]) -> str:
    pulse = subject.get("pulse")
    if not pulse:
        return "—"
    return f"{pulse['mean']:.0f} bpm ({pulse['min']:.0f}–{pulse['max']:.0f})"


def select_moments(
    episodes: list[dict[str, Any]], limit: int = _MAX_MOMENTS
) -> list[dict[str, Any]]:
    """Notable episodes first, then the most intense others; returned chronologically."""
    notable = [e for e in episodes if e.get("notable")]
    others = sorted(
        (e for e in episodes if not e.get("notable") and e.get("kind") != "behavior"),
        key=lambda e: -float(e.get("intensity", 0.0)),
    )
    if len(notable) > limit:
        notable = sorted(notable, key=lambda e: -float(e.get("intensity", 0.0)))[:limit]
    chosen = notable + others[: max(0, limit - len(notable))]
    return sorted(chosen, key=lambda e: float(e.get("at", 0.0)))


def _escape_cell(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def render_analysis_markdown(analysis: dict[str, Any] | None) -> str:
    """Deterministic Markdown for the measured analysis — no model calls."""
    if not analysis:
        return ""
    expressions = analysis.get("expressions") or {}
    subjects = expressions.get("subjects") or []
    insights = analysis.get("insights") or []
    if not subjects and not insights:
        return ""

    session = analysis.get("session") or {}
    lines = [ANALYSIS_HEADING, "", DISCLAIMER, ""]
    meta = []
    profile = (session.get("profile") or {}).get("label")
    if profile:
        meta.append(f"**Profile:** {profile}")
    if session.get("source_label"):
        meta.append(f"**Source:** {session['source_label']}")
    if session.get("duration_sec"):
        meta.append(f"**Analyzed:** {format_clock(session['duration_sec'])}")
    if session.get("media_duration_sec"):
        meta.append(f"**Media length:** {format_clock(session['media_duration_sec'])}")
    if meta:
        lines += [" · ".join(meta), ""]

    if subjects:
        lines += [
            "### Subjects",
            "",
            "| Subject | Observed | Micro | Macro | Behavior | Blinks /min | Gaze away "
            "| Nods / shakes | Self-touch | Pulse (experimental) | Affect over time |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for subject in subjects:
            counts = subject.get("episodes") or {}
            blink = subject.get("blink") or {}
            rate = blink.get("rate_per_min")
            lines.append(
                f"| {subject.get('id')} | {format_clock(subject.get('observed_sec', 0))} "
                f"| {counts.get('micro', 0)} | {counts.get('macro', 0)} "
                f"| {counts.get('behavior', 0)} "
                f"| {rate if rate is not None else '—'} "
                f"| {subject.get('gaze_away_pct', 0)}% "
                f"| {subject.get('head_nods', 0)} / {subject.get('head_shakes', 0)} "
                f"| {subject.get('self_touch', 0)} | {_pulse_text(subject)} "
                f"| {_dominant_affect(subject)} |"
            )
        lines.append("")

    moments = select_moments(expressions.get("episodes") or [])
    if moments:
        lines += [
            "### Notable moments",
            "",
            "| Time | Subject | Type | Signal | Action units | Duration | Cues |",
            "|---|---|---|---|---|---|---|",
        ]
        for episode in moments:
            speaking = " (speaking)" if episode.get("speaking") else ""
            lines.append(
                f"| {format_clock(episode.get('at', 0))} | {episode.get('subject')} "
                f"| {episode.get('kind')} | {_escape_cell(episode.get('label'))}{speaking} "
                f"| {episode.get('au_label') or '—'} | {episode.get('duration_ms', 0)} ms "
                f"| {_escape_cell(', '.join(episode.get('cue_labels') or [])) or '—'} |"
            )
        lines.append("")

    cue_counts = expressions.get("cue_counts") or {}
    if cue_counts:
        lines += [
            "### Behavioral cues",
            "",
            "| Cue | Count | What it can indicate |",
            "|---|---|---|",
        ]
        for key, count in cue_counts.items():
            cue = CUES.get(key)
            if cue is None:
                continue
            lines.append(f"| {cue.label} | {count} | {_escape_cell(cue.meaning)} |")
        lines.append("")

    if insights:
        lines += ["### YUBI v3.0 interpretations", ""]
        for item in insights:
            evidence = ", ".join(item.get("evidence") or [])
            subject = f"subject {item['subject']}, " if item.get("subject") else ""
            line = (
                f"- **[{item.get('at_clock', '')}] {item.get('title', '')}** "
                f"({subject}confidence {float(item.get('confidence', 0)):.0%}) — "
                f"{item.get('observation', '')}"
            )
            if item.get("interpretation"):
                line += f" *{item['interpretation']}*"
            if evidence:
                line += f" Evidence: {evidence}."
            lines.append(line)
        lines.append("")
    if analysis.get("running_summary"):
        lines += [f"**Session read:** {analysis['running_summary']}", ""]

    usage = analysis.get("ai_usage") or {}
    if usage.get("calls"):
        tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        lines += [
            f"*YUBI v3.0 usage: {usage['calls']} calls · {tokens / 1000:.1f}k tokens · "
            f"${float(usage.get('cost_usd', 0)):.3f}*",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def with_analysis_section(draft_markdown: str, analysis_markdown: str) -> str:
    """Place measured analysis ahead of the raw chronology (or append when absent)."""
    if not analysis_markdown or ANALYSIS_HEADING in draft_markdown:
        return draft_markdown
    section = analysis_markdown.strip() + "\n\n"
    index = draft_markdown.find(CHRONOLOGY_HEADING)
    if index >= 0:
        return draft_markdown[:index] + section + draft_markdown[index:]
    return draft_markdown.rstrip() + "\n\n" + section


def _downsample(points: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return points
    step = len(points) / limit
    return [points[int(i * step)] for i in range(limit)]


def evidence_payload(
    analysis: dict[str, Any] | None, session: dict[str, Any] | None
) -> dict[str, Any]:
    """Compact, model-facing view of the session (episodes trimmed, timeline downsampled)."""
    analysis = analysis or {}
    expressions = analysis.get("expressions") or {}
    episodes = select_moments(expressions.get("episodes") or [], _MAX_EVIDENCE_EPISODES)
    keep = ("id", "subject", "kind", "label", "at", "duration_ms", "intensity", "aus",
            "emotion", "side", "cues", "speaking")
    timeline = {
        subject: _downsample(points, _MAX_TIMELINE_POINTS)
        for subject, points in (expressions.get("timeline") or {}).items()
    }
    client_session = session or {}
    return {
        "session": analysis.get("session") or {"id": client_session.get("id")},
        "log": {
            "event_count": len(client_session.get("events") or []),
            "peak_objects": client_session.get("peakObjectCount"),
        },
        "scene_summary": analysis.get("scene_summary", ""),
        "subjects": expressions.get("subjects") or [],
        "episode_counts": expressions.get("counts") or {},
        "episodes": [{k: e[k] for k in keep if k in e and e[k] not in ("", [], None)}
                     for e in episodes],
        "cue_counts": expressions.get("cue_counts") or {},
        "affect_timeline": timeline,
        "live_insights": [
            {k: item.get(k) for k in ("id", "at", "title", "observation", "confidence")}
            for item in analysis.get("insights") or []
        ],
        "running_summary": analysis.get("running_summary", ""),
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _compact_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


def _run_analyze(ai: AIClient, evidence: dict[str, Any]) -> dict[str, Any]:
    result = ai.generate(
        feature="report",
        model=ai.insight_model,
        contents=f"{ANALYZE_PROMPT}\n\nMeasurements:\n{_compact_json(evidence)}",
        json_mode=True,
        temperature=0.2,
        thinking="medium",
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=4096,
        timeout_sec=90,
        essential=True,
    )
    return result.data if isinstance(result.data, dict) else {}


def _run_compose(
    ai: AIClient, draft: str, findings: dict[str, Any] | None, model: str
) -> str:
    clause = " and the analyst findings that follow it" if findings else ""
    prompt = COMPOSE_PROMPT.format(findings_clause=clause) + f"\n\n---\nDraft:\n\n{draft}"
    if findings:
        prompt += "\n\n---\nAnalyst findings (JSON):\n" + _compact_json(findings)
    result = ai.generate(
        feature="report",
        model=model,
        contents=prompt,
        temperature=0.3,
        thinking="low",
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=12000,
        timeout_sec=120,
        essential=True,
    )
    return _strip_fences(result.text) or draft


def apply_fixes(markdown: str, fixes: Any) -> tuple[str, int]:
    applied = 0
    if not isinstance(fixes, list):
        return markdown, 0
    for fix in fixes[:8]:
        if not isinstance(fix, dict):
            continue
        find = str(fix.get("find", ""))
        replace = str(fix.get("replace", ""))
        if len(find) >= 8 and find in markdown and find != replace:
            markdown = markdown.replace(find, replace, 1)
            applied += 1
    return markdown, applied


def _run_audit(ai: AIClient, markdown: str, evidence: dict[str, Any]) -> tuple[str, int]:
    result = ai.generate(
        feature="report",
        model=ai.insight_model,
        contents=(
            f"{AUDIT_PROMPT}\n\nMeasurements:\n{_compact_json(evidence)}"
            f"\n\n---\nReport:\n\n{markdown}"
        ),
        json_mode=True,
        temperature=0.1,
        thinking="low",
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=3000,
        timeout_sec=90,
        essential=True,
    )
    data = result.data if isinstance(result.data, dict) else {}
    return apply_fixes(markdown, data.get("fixes"))


def enrich_session_report_events(
    draft_markdown: str,
    session: dict[str, Any] | None,
    settings: Settings,
    ai: AIClient | None = None,
    analysis: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield NDJSON progress events, ending with ``done`` + the final markdown."""
    draft = with_analysis_section(draft_markdown, render_analysis_markdown(analysis))
    steps = pass_plan(settings.session_report_passes)
    yield {
        "phase": "compile",
        "message": "Compiling session data…",
        "pass": 0,
        "steps": [PASS_LABELS[step] for step in steps],
    }

    client = ai or AIClient(settings)
    if not settings.gemini_active or not client.available:
        yield {
            "phase": "done",
            "message": "YUBI v3.0 offline — exporting measured session log.",
            "markdown": draft,
            "passes_completed": 0,
            "enriched": False,
        }
        return

    evidence = evidence_payload(analysis, session)
    findings: dict[str, Any] | None = None
    markdown = draft
    completed = 0
    for index, step in enumerate(steps):
        yield {
            "phase": "pass",
            "pass": index + 1,
            "total_passes": len(steps),
            "step": step,
            "label": PASS_LABELS[step],
            "message": f"YUBI v3.0 — {PASS_LABELS[step]}…",
        }
        try:
            if step == "analyze":
                findings = _run_analyze(client, evidence)
            elif step == "compose":
                model = client.perception_model if findings else client.insight_model
                markdown = _run_compose(client, draft, findings, model)
            else:
                markdown, applied = _run_audit(client, markdown, evidence)
                yield {
                    "phase": "audit",
                    "pass": index + 1,
                    "message": f"Audit applied {applied} correction{'s' if applied != 1 else ''}.",
                    "fixes_applied": applied,
                }
            completed = index + 1
        except AIUnavailable as exc:
            yield {
                "phase": "error",
                "message": f"{PASS_LABELS[step]} failed — exporting best draft.",
                "detail": str(exc),
            }
            break

    usage = client.budget.snapshot()
    yield {
        "phase": "done",
        "message": "Session report ready.",
        "markdown": markdown,
        "passes_completed": completed,
        "enriched": completed > 0 and markdown != draft,
        "cost_usd": usage.get("features", {}).get("report", {}).get("cost_usd", 0.0),
    }

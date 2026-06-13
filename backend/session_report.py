"""YUBI v3.0 multi-pass session report enrichment."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from backend.config import Settings

PASS_PROMPTS = (
    (
        "Pass 1 — Structure and executive summary. "
        "Add a concise **Executive Summary** after the title block. "
        "Reorganize sections for scanability; keep every factual metric and table row. "
        "Do not invent events or objects not present in the draft."
    ),
    (
        "Pass 2 — Analysis and synthesis. "
        "Add **Key findings** and **Timeline highlights** sections that synthesize "
        "the event chronology into plain-language insights. "
        "Collapse duplicate or near-duplicate chronology rows; keep unique signals."
    ),
    (
        "Pass 3 — Editorial finalize. "
        "Polish tone for an operational intelligence brief. "
        "Ensure Markdown tables and headings are consistent. "
        "Footer must credit YUBI Supervision Node and YUBI v3.0 only."
    ),
)

SYSTEM_INSTRUCTION = (
    "You are YUBI v3.0, the intelligence layer of YUBI Supervision — "
    "a real-time vision supervision platform. "
    "You refine session logs into clear operational markdown reports. "
    "Never mention Gemini, Google, or underlying model vendors. "
    "Refer only to YUBI v3.0 and YUBI Supervision. "
    "Return the full revised markdown document only — no JSON wrapper."
)


def _run_v3_pass(markdown: str, pass_instruction: str, settings: Settings) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = (
        f"{pass_instruction}\n\n"
        "---\n"
        "Revise this session log markdown:\n\n"
        f"{markdown}"
    )
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        return markdown
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text or markdown


def enrich_session_report_events(
    draft_markdown: str,
    session: dict[str, Any] | None,
    settings: Settings,
) -> Iterator[dict[str, Any]]:
    """Yield progress events, ending with done + final markdown."""
    yield {
        "phase": "compile",
        "message": "Compiling session data…",
        "pass": 0,
    }

    if not settings.gemini_active:
        yield {
            "phase": "done",
            "message": "YUBI v3.0 offline — exporting raw session log.",
            "markdown": draft_markdown,
            "passes_completed": 0,
            "enriched": False,
        }
        return

    passes = max(1, min(3, settings.session_report_passes))
    markdown = draft_markdown
    completed_passes = 0
    session_hint = ""
    if session:
        session_hint = json.dumps(
            {
                "id": session.get("id"),
                "duration_hint": session.get("startedAt"),
                "event_count": len(session.get("events") or []),
                "peak_objects": session.get("peakObjectCount"),
            },
            indent=2,
        )

    for index in range(passes):
        pass_num = index + 1
        yield {
            "phase": "pass",
            "pass": pass_num,
            "total_passes": passes,
            "message": f"YUBI v3.0 pass {pass_num} of {passes}…",
        }
        instruction = PASS_PROMPTS[index]
        if session_hint and pass_num == 1:
            instruction += f"\n\nSession metadata:\n{session_hint}"
        try:
            markdown = _run_v3_pass(markdown, instruction, settings)
            completed_passes = pass_num
        except Exception as exc:
            yield {
                "phase": "error",
                "message": f"YUBI v3.0 pass {pass_num} failed — exporting best draft.",
                "detail": str(exc),
            }
            break

    yield {
        "phase": "done",
        "message": "Session report ready.",
        "markdown": markdown,
        "passes_completed": completed_passes,
        "enriched": completed_passes > 0,
    }

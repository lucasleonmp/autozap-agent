"""
Human handoff: pause AI on chat_memory even when no row exists yet.

P1 — upsert on (chat_id, workspace_id) so the first WhatsApp message still
pauses. A plain UPDATE is a no-op when chat_memory is missing; a later
INSERT in MemoryManager.save then races UNIQUE(chat_id, workspace_id)
(Postgres name may be chat_memory_chat_id_workspace_id_key or
chat_memory_chat_id_workspace_unique).

P2 — if the model claims it transferred without calling transfer_to_human,
still pause. process-message in autozapv2 can treat tools_used /
handoff_source as a secondary hook; pause itself lives here.

pause_reason stays a short code on chat_memory (P6 may move history to
handoff_events). Optional insert into handoff_events is best-effort and
must never fail the pause.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Stable codes for inbox / alerts in autozapv2. Do not put free-text here.
PAUSE_REASON_TOOL = "handoff"
PAUSE_REASON_VERBAL = "handoff_verbal"

HANDOFF_CLAIM_RE = re.compile(
    r"("
    r"vou\s+(te\s+)?(passar|transferir|encaminhar)"
    r"|estou\s+(te\s+)?(passando|transferindo|encaminhando)"
    r"|j[aá]\s+(te\s+)?(passei|transferi|encaminhei)"
    r"|transferi\s+(voc[eê]|sua\s+conversa|o\s+atendimento)"
    r"|passar\s+(voc[eê]|sua\s+conversa)\s+para\s+(um\s+)?(atendente|humano|equipe)"
    r"|encaminh(ar|ei|ando)\s+para\s+(um\s+)?(atendente|humano|nossa\s+equipe)"
    r"|um\s+(atendente|humano)\s+(vai|ir[aá]|continuar[aá])"
    r"|nossa\s+equipe\s+(humana\s+)?(vai|ir[aá])\s+(continuar|atender)"
    r"|i('?ll| will)\s+transfer"
    r"|transferr?ing\s+you"
    r")",
    re.IGNORECASE,
)

HANDOFF_NEGATION_RE = re.compile(
    r"n[aã]o\s+(vou|preciso|quero|posso|irei)\s+(te\s+)?(passar|transferir|encaminhar)",
    re.IGNORECASE,
)

CHAT_MEMORY_CONFLICT = "chat_id,workspace_id"


def claims_human_transfer(text: str | None) -> bool:
    """True when the assistant claims a human handoff in natural language."""
    if not text or not str(text).strip():
        return False
    if HANDOFF_NEGATION_RE.search(text):
        return False
    return bool(HANDOFF_CLAIM_RE.search(text))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execute(query: Any):
    return query.execute()


def pause_for_human_handoff(
    supabase: Any,
    *,
    workspace_id: str,
    lead_id: str,
    chat_id: str | None,
    reason: str,
    source: str = "tool",
) -> dict:
    """
    Pause this conversation. Always upserts chat_memory.

    source: "tool" | "verbal"
    Returns {ok, pause_reason, created, error?}
    """
    pause_reason = PAUSE_REASON_TOOL if source == "tool" else PAUSE_REASON_VERBAL
    resolved_chat_id = chat_id or lead_id
    pause_row = {
        "ai_paused": True,
        "pause_reason": pause_reason,
        "paused_at": _now_iso(),
        "paused_by": None,
    }
    created = False

    try:
        existing = (
            supabase.table("chat_memory")
            .select("id")
            .eq("workspace_id", workspace_id)
            .eq("lead_id", lead_id)
            .limit(1)
            .execute()
        )
        rows = (existing.data if existing else None) or []

        if rows:
            supabase.table("chat_memory").update(pause_row).eq(
                "id", rows[0]["id"]
            ).execute()
        else:
            created = True
            supabase.table("chat_memory").upsert(
                {
                    **pause_row,
                    "workspace_id": workspace_id,
                    "chat_id": resolved_chat_id,
                    "lead_id": lead_id,
                    "conversation_history": [],
                },
                on_conflict=CHAT_MEMORY_CONFLICT,
            ).execute()

        supabase.table("leads").update(
            {
                "ai_enabled": False,
                "updated_at": _now_iso(),
            }
        ).eq("id", lead_id).execute()

        _maybe_insert_handoff_event(
            supabase,
            workspace_id=workspace_id,
            lead_id=lead_id,
            chat_id=resolved_chat_id,
            source=source,
            reason=reason,
            pause_reason=pause_reason,
        )
        return {"ok": True, "pause_reason": pause_reason, "created": created}
    except Exception as exc:
        return {"ok": False, "pause_reason": pause_reason, "created": created, "error": str(exc)}


def _maybe_insert_handoff_event(
    supabase: Any,
    *,
    workspace_id: str,
    lead_id: str,
    chat_id: str,
    source: str,
    reason: str,
    pause_reason: str,
) -> None:
    """P6 table may not exist yet — never block the pause."""
    try:
        supabase.table("handoff_events").insert(
            {
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "chat_id": chat_id,
                "source": source,
                "reason": reason,
                "pause_reason": pause_reason,
                "created_at": _now_iso(),
            }
        ).execute()
    except Exception as exc:
        print(f"[handoff] handoff_events insert skipped: {exc}")

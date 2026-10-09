from agent.handoff import (
    PAUSE_REASON_TOOL,
    PAUSE_REASON_VERBAL,
    claims_human_transfer,
    pause_for_human_handoff,
)
from tests.fake_supabase import FakeStore


def test_claims_transfer_portuguese_and_english():
    assert claims_human_transfer("Vou te transferir para um atendente agora.")
    assert claims_human_transfer("Estou encaminhando para a nossa equipe.")
    assert claims_human_transfer("Já te passei para um humano.")
    assert claims_human_transfer("I'll transfer you to a specialist.")
    assert not claims_human_transfer("Não vou transferir, posso te ajudar aqui.")
    assert not claims_human_transfer("Qual o prazo de entrega?")
    assert not claims_human_transfer("")


def test_pause_upserts_when_no_chat_memory_row():
    db = FakeStore()
    db.tables["leads"] = [{"id": "lead-1", "ai_enabled": True}]

    result = pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="5511999999999",
        reason="cliente pediu atendente",
        source="tool",
    )

    assert result["ok"] is True
    assert result["created"] is True
    assert result["pause_reason"] == PAUSE_REASON_TOOL
    memory = db.tables["chat_memory"]
    assert len(memory) == 1
    row = memory[0]
    assert row["chat_id"] == "5511999999999"
    assert row["workspace_id"] == "ws-1"
    assert row["lead_id"] == "lead-1"
    assert row["ai_paused"] is True
    assert row["pause_reason"] == PAUSE_REASON_TOOL
    assert row["paused_at"]
    assert db.tables["leads"][0]["ai_enabled"] is False


def test_second_pause_does_not_duplicate_unique_chat_workspace():
    db = FakeStore()
    db.tables["leads"] = [{"id": "lead-1", "ai_enabled": True}]

    pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="chat-a",
        reason="primeiro",
        source="tool",
    )
    pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="chat-a",
        reason="de novo",
        source="verbal",
    )

    assert len(db.tables["chat_memory"]) == 1
    assert db.tables["chat_memory"][0]["pause_reason"] == PAUSE_REASON_VERBAL


def test_updates_existing_row_by_lead_id():
    db = FakeStore()
    db.tables["chat_memory"] = [
        {
            "id": "mem-9",
            "workspace_id": "ws-1",
            "lead_id": "lead-1",
            "chat_id": "chat-old",
            "ai_paused": False,
        }
    ]
    db.tables["leads"] = [{"id": "lead-1", "ai_enabled": True}]

    result = pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="chat-new",
        reason="tool",
        source="tool",
    )

    assert result["created"] is False
    assert len(db.tables["chat_memory"]) == 1
    assert db.tables["chat_memory"][0]["id"] == "mem-9"
    assert db.tables["chat_memory"][0]["ai_paused"] is True
    assert db.tables["chat_memory"][0]["chat_id"] == "chat-old"


def test_handoff_events_failure_does_not_block_pause():
    db = FakeStore()
    db.tables["leads"] = [{"id": "lead-1", "ai_enabled": True}]
    db.fail_tables["handoff_events"] = "relation handoff_events does not exist"

    result = pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="chat-a",
        reason="ok",
        source="tool",
    )
    assert result["ok"] is True
    assert db.tables["chat_memory"][0]["ai_paused"] is True

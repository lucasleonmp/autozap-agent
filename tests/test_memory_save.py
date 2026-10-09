import asyncio

from agent.memory import MemoryManager
from tests.fake_supabase import FakeStore


def test_save_upserts_instead_of_duplicate_insert_after_handoff():
    db = FakeStore()
    db.tables["chat_memory"] = [
        {
            "id": "mem-handoff",
            "workspace_id": "ws-1",
            "lead_id": "lead-1",
            "chat_id": "chat-a",
            "ai_paused": True,
            "pause_reason": "handoff",
            "conversation_history": [],
        }
    ]
    manager = MemoryManager(db)

    asyncio.run(
        manager.save(
            memory_id=None,
            lead_id="lead-1",
            workspace_id="ws-1",
            chat_id="chat-a",
            new_messages=[{"role": "user", "content": "quero um humano"}],
            current_memory={"history": [], "summary": None, "lead_profile": {}},
        )
    )

    assert len(db.tables["chat_memory"]) == 1
    row = db.tables["chat_memory"][0]
    assert row["ai_paused"] is True
    assert row["pause_reason"] == "handoff"
    assert row["conversation_history"][0]["content"] == "quero um humano"


def test_plain_insert_without_existing_row_still_creates():
    db = FakeStore()
    manager = MemoryManager(db)
    asyncio.run(
        manager.save(
            memory_id=None,
            lead_id="lead-1",
            workspace_id="ws-1",
            chat_id="chat-a",
            new_messages=[{"role": "assistant", "content": "oi"}],
            current_memory={"history": [], "summary": None, "lead_profile": {}},
        )
    )
    assert len(db.tables["chat_memory"]) == 1
    assert db.tables["chat_memory"][0]["chat_id"] == "chat-a"

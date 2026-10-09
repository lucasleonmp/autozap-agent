from agent.handoff import claims_human_transfer, pause_for_human_handoff
from tests.fake_supabase import FakeStore


def test_verbal_path_used_when_model_claims_without_tool():
    """Mirrors graph.py: if transfer_to_human not in tools_used, still pause."""
    response = "Claro! Vou te transferir para um atendente agora."
    tools_used: list[str] = []
    assert "transfer_to_human" not in tools_used
    assert claims_human_transfer(response)

    db = FakeStore()
    db.tables["leads"] = [{"id": "lead-1", "ai_enabled": True}]
    pause_for_human_handoff(
        db,
        workspace_id="ws-1",
        lead_id="lead-1",
        chat_id="chat-a",
        reason="verbal_claim",
        source="verbal",
    )
    assert db.tables["chat_memory"][0]["pause_reason"] == "handoff_verbal"

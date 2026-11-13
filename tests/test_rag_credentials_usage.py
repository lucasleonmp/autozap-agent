import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.rag import RAGEngine
from agent.usage import UsageCallback, record_usage


class Supabase:
    def __init__(self, results=None, fail_log=False):
        self.calls = []
        self.rows = []
        self.results = results or {}
        self.fail_log = fail_log

    def rpc(self, name, params):
        self.calls.append((name, params))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.results.get(name, [])))

    def table(self, name):
        assert name == "ai_usage_logs"
        def insert(row):
            self.rows.append(row)
            def execute():
                if self.fail_log:
                    raise RuntimeError("unavailable")
            return SimpleNamespace(execute=execute)
        return SimpleNamespace(insert=insert)


def mock_http(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return patch("agent.rag.httpx.AsyncClient", return_value=client)


@pytest.mark.asyncio
async def test_embedding_http_and_scoped_rpcs():
    db = Supabase({"match_knowledge_base": [{"id": "1", "title": "t", "content": "c"}]})
    def handle(request):
        assert request.url == "https://api.openai.com/v1/embeddings"
        assert request.headers["authorization"] == "Bearer openai-key"
        assert json.loads(request.content) == {"model": "text-embedding-3-large", "dimensions": 1536, "input": "hello"}
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1536}], "usage": {"prompt_tokens": 7}})
    with mock_http(handle):
        result = await RAGEngine(db, "openai-key").search("workspace-a", "hello", "agent-a")
    assert "c" in result
    assert [c[0] for c in db.calls] == ["match_knowledge_base", "fulltext_knowledge_base"]
    vector, text = [c[1] for c in db.calls]
    assert vector["p_workspace_id"] == text["p_workspace_id"] == "workspace-a"
    assert vector["p_agent_id"] == text["p_agent_id"] == "agent-a"
    assert len(json.loads(vector["query_embedding"])) == 1536
    assert db.rows[0]["workspace_id"] == "workspace-a"
    assert db.rows[0]["prompt_tokens"] == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("key,status", [(None, None), ("wrong", 401)])
async def test_missing_or_wrong_openai_key_keeps_fulltext(key, status):
    db = Supabase({"fulltext_knowledge_base": [{"id": "1", "title": "t", "content": "found"}]})
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status or 500)
    with mock_http(handle):
        result = await RAGEngine(db, key).search("w", "q", "a")
    assert "found" in result
    assert len(calls) == (1 if key else 0)
    assert [x[0] for x in db.calls] == ["fulltext_knowledge_base"]
    assert db.calls[0][1]["p_workspace_id"] == "w"
    assert db.calls[0][1]["p_agent_id"] == "a"


@pytest.mark.asyncio
async def test_telemetry_failure_is_non_blocking_and_usage_is_scoped():
    db = Supabase(fail_log=True)
    callback = UsageCallback(db, "workspace-a")
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok", usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}, response_metadata={"model_name": "gemini-2.5-flash-lite"}))]])
    await callback.on_llm_end(response)
    await record_usage(db, "workspace-b", "openai", "text-embedding-3-large", 10, 0)
    assert [r["workspace_id"] for r in db.rows] == ["workspace-a", "workspace-b"]
    assert db.rows[0]["completion_tokens"] == 3


def test_google_model_and_credentials(monkeypatch):
    from agent.graph import AutozapAgent
    monkeypatch.setenv("SUPABASE_URL", "https://example.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("agent.graph.create_client"), patch("agent.graph.ChatGoogleGenerativeAI") as google:
        agent = AutozapAgent("google-key", "gemini-2.5-flash")
        assert agent.rag_engine.openai_api_key is None
        assert google.call_args_list[0].kwargs["google_api_key"] == "google-key"
        with pytest.raises(ValueError, match="Gemini"):
            AutozapAgent("google-key", "gpt-4o")
        assert google.call_count == 2


@pytest.mark.asyncio
async def test_process_uses_edge_context_and_direct_fallback_without_openai(monkeypatch):
    from agent.graph import AutozapAgent
    agent = AutozapAgent.__new__(AutozapAgent)
    agent.supabase = MagicMock()
    agent.supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None
    agent.llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="resposta")))
    agent.llm_lite = MagicMock()
    agent.rag_engine = RAGEngine(Supabase(), None)
    agent.memory_manager = MagicMock()
    agent.memory_manager.load = AsyncMock(return_value={"history": []})
    agent.memory_manager.sync_missed_messages = AsyncMock(side_effect=lambda _lead, memory: memory)
    agent.memory_manager.save = AsyncMock()
    agent.memory_manager.get_chat_messages.return_value = []
    with patch("agent.graph.create_tools", return_value=[]), \
         patch("agent.graph.create_tool_calling_agent"), \
         patch("agent.graph.AgentExecutor") as executor, \
         patch("agent.graph.build_system_prompt", return_value="prompt") as prompt:
        executor.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("agent failed"))
        result = await agent.process("lead", "workspace", "oi", {"id": "agent"}, edge_knowledge_context="edge KB")
    assert result["response"] == "resposta"
    assert prompt.call_args.kwargs["knowledge_context"] == "edge KB"
    assert agent.llm.ainvoke.await_count == 1
    assert agent.memory_manager.save.await_count == 1


@pytest.mark.asyncio
async def test_internal_llm_callback_is_propagated_through_langchain():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    db = Supabase()
    response = AIMessage(content='[1, 2]', usage_metadata={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}, response_metadata={"model_name": "gemini-2.5-flash-lite"})
    llm = FakeMessagesListChatModel(responses=[response]).with_config({"callbacks": [UsageCallback(db, "w")]})
    rows = [{"id": "1", "title": "a", "content": "a"}, {"id": "2", "title": "b", "content": "b"}]
    assert len(await RAGEngine(db, None)._llm_rerank(rows, "q", llm)) == 2
    assert db.rows[0]["workspace_id"] == "w"
    assert db.rows[0]["model"] == "gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_rpc_failure_yields_empty_context():
    db = Supabase()
    db.rpc = MagicMock(side_effect=RuntimeError("RPC unavailable"))
    assert await RAGEngine(db, None).search("workspace", "q", "agent") == ""

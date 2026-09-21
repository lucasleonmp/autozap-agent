"""
============================================
Agent Graph - LangChain Agent with Memory & Tools
============================================
O cérebro do agente. Orquestra memória, RAG e ferramentas
em um loop de raciocínio (ReAct pattern).
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from supabase import create_client

from .memory import MemoryManager
from .rag import RAGEngine
from .tools import create_tools
from .prompts import build_system_prompt
from .response_guard import agent_instructs_silence_for_media, should_suppress_ai_response


class AutozapAgent:
    """Agente principal do Autozap com memória, RAG e ferramentas."""

    def __init__(self, ai_api_key: str | None = None, ai_model: str | None = None):
        supabase_url = os.environ["SUPABASE_URL"]
        supabase_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.supabase = create_client(supabase_url, supabase_key)

        # AI keys: request > env var > error
        resolved_key = ai_api_key or os.environ.get("AI_API_KEY", "")
        resolved_model = ai_model or os.environ.get("AI_MODEL", "gemini-2.5-flash")

        if not resolved_key:
            raise ValueError("AI_API_KEY must be provided via request or environment variable")

        self.llm = ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=resolved_key,
            temperature=0.3,
            max_output_tokens=800,
            convert_system_message_to_human=False,
        )

        # LLM leve para tarefas internas (resumos, extração)
        self.llm_lite = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=resolved_key,
            temperature=0.1,
            max_output_tokens=300,
        )

        self.memory_manager = MemoryManager(self.supabase)

        # OpenAI key for embeddings (RAG) — separate from Gemini LLM key
        openai_key = os.environ.get("OPENAI_API_KEY", resolved_key)
        self.rag_engine = RAGEngine(self.supabase, openai_key)

    async def process(
        self,
        lead_id: str,
        workspace_id: str,
        message: str,
        agent_config: dict,
        instance_id: str | None = None,
        chat_id: str | None = None,
        edge_knowledge_context: str | None = None,
    ) -> dict:
        """Processa uma mensagem e retorna a resposta do agente."""

        # 1. CARREGAR MEMÓRIA
        memory = await self.memory_manager.load(lead_id, workspace_id)

        # 1b. SINCRONIZAR mensagens perdidas (hands-on / pausa)
        memory = await self.memory_manager.sync_missed_messages(lead_id, memory)

        # Checar se AI está pausada
        if memory.get("ai_paused"):
            return {"response": None, "status": "ai_paused"}

        # Deterministic silence: custom instructions say not to reply to this media
        if agent_instructs_silence_for_media(agent_config.get("system_prompt"), message):
            print("[Agent] Skipping media reply (agent instructed silence)")
            return {"response": None, "status": "no_reply"}

        # 2. RAG - Buscar no knowledge base
        knowledge_context = ""
        try:
            knowledge_context = await self.rag_engine.search(
                workspace_id=workspace_id,
                query=message,
                agent_id=agent_config.get("id"),
                llm=self.llm_lite,
            )
        except Exception as e:
            print(f"[Agent] RAG search failed (non-blocking): {e}")

        # Use edge context as fallback if Python RAG returned nothing
        if not knowledge_context and edge_knowledge_context:
            print(f"[Agent] 📚 Using Edge Function KB context as fallback ({len(edge_knowledge_context)} chars)")
            knowledge_context = edge_knowledge_context
        elif knowledge_context:
            print(f"[Agent] 📚 Using Python RAG context ({len(knowledge_context)} chars)")
        else:
            print("[Agent] ⚠️ No knowledge context available from either source")

        # 3. CONSTRUIR PROMPT DINÂMICO
        lead_name = None
        try:
            lead_result = (
                self.supabase.table("leads")
                .select("name")
                .eq("id", lead_id)
                .single()
                .execute()
            )
            lead_name = lead_result.data.get("name") if lead_result.data else None
        except Exception:
            pass

        memory_context = self.memory_manager.format_for_prompt(memory)
        is_returning = len(memory.get("history", [])) > 0

        system_prompt = build_system_prompt(
            agent_config=agent_config,
            memory_context=memory_context,
            knowledge_context=knowledge_context,
            lead_name=lead_name,
            is_returning=is_returning,
        )

        # 4. CRIAR FERRAMENTAS
        enabled_tools = agent_config.get("enabled_tools", None)
        tools = create_tools(self.supabase, workspace_id, lead_id, instance_id=instance_id, enabled_tools=enabled_tools)

        # 5. MONTAR AGENTE COM LANGCHAIN
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        # Preparar histórico de chat para LangChain
        chat_history = self.memory_manager.get_chat_messages(memory, limit=20)

        # 6. EXECUTAR AGENTE
        result = None
        try:
            result = await executor.ainvoke({
                "input": message,
                "chat_history": chat_history,
            })

            ai_response = result["output"]
        except Exception as e:
            print(f"[Agent] Error during execution: {e}")
            # Fallback: chamada direta sem ferramentas
            ai_response = await self._fallback_response(
                system_prompt, chat_history, message
            )

        # 7. Converter markdown para WhatsApp
        final_response = self._convert_to_whatsapp(ai_response)

        # 7b. Never deliver meta "I'm not replying" or looped garbage
        if should_suppress_ai_response(final_response):
            print(
                f"[Agent] Suppressing no_reply/loop response: {str(final_response)[:120]!r}"
            )
            return {"response": None, "status": "no_reply"}

        # 8. SALVAR MEMÓRIA
        new_messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_response},
        ]
        await self.memory_manager.save(
            memory_id=memory.get("id"),
            lead_id=lead_id,
            workspace_id=workspace_id,
            chat_id=chat_id or lead_id,
            new_messages=new_messages,
            current_memory=memory,
            llm=self.llm_lite,
        )

        return {
            "response": final_response,
            "status": "success",
            "tools_used": [
                step[0].tool for step in result.get("intermediate_steps", [])
            ] if isinstance(result, dict) else [],
        }

    async def _fallback_response(
        self, system_prompt: str, chat_history: list, message: str
    ) -> str:
        """Resposta direta sem ferramentas (fallback de segurança)."""
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(chat_history[-10:])
        messages.append(HumanMessage(content=message))

        response = await self.llm.ainvoke(messages)
        return response.content

    @staticmethod
    def _convert_to_whatsapp(text: str) -> str:
        """Converte markdown para formatação WhatsApp."""
        if not text:
            return text

        import re

        result = text
        # **bold** → *bold*
        result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
        # __italic__ → _italic_
        result = re.sub(r"__(.+?)__", r"_\1_", result)
        # ### Header → *Header*
        result = re.sub(r"^#{1,3}\s+(.+)$", r"*\1*", result, flags=re.MULTILINE)
        # Remove code block markers
        result = re.sub(r"```[\w]*\n?", "", result)

        return result.strip()

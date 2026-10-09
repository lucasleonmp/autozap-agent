"""
============================================
Memory Manager - Long-term & Short-term Memory
============================================
Prioridade #1: Resolver o esquecimento do agente.

Três camadas de memória:
1. Buffer (curto prazo): Últimas 20 mensagens
2. Resumo (médio prazo): Resumo comprimido do histórico
3. Perfil Semântico (longo prazo): Fatos persistentes sobre o lead
"""

from datetime import datetime, timezone
from supabase import Client
from langchain_core.messages import HumanMessage, AIMessage


class MemoryManager:
    """Gerencia memória de curto, médio e longo prazo para cada lead."""

    def __init__(self, supabase: Client):
        self.supabase = supabase

    async def load(self, lead_id: str, workspace_id: str) -> dict:
        """Carrega todas as camadas de memória de um lead."""
        result = (
            self.supabase.table("chat_memory")
            .select("id, conversation_history, conversation_summary, context_flags, lead_profile, last_interaction, active_agent_id, ai_paused")
            .eq("lead_id", lead_id)
            .eq("workspace_id", workspace_id)
            .maybe_single()
            .execute()
        )

        # NOTE: with supabase-py/postgrest, `.maybe_single().execute()` returns
        # None (not a response with data=None) when there are 0 rows. Guarding
        # only `result.data` raised `'NoneType' object has no attribute 'data'`,
        # crashing every first-message conversation and forcing the Edge Function
        # to fall back to the legacy engine (no tools).
        if not result or not result.data:
            return {
                "id": None,
                "history": [],
                "summary": None,
                "context_flags": {},
                "lead_profile": {},
                "ai_paused": False,
                "last_interaction": None,
                "active_agent_id": None,
            }

        data = result.data
        return {
            "id": data.get("id"),
            "history": data.get("conversation_history", []),
            "summary": data.get("conversation_summary"),
            "context_flags": data.get("context_flags", {}),
            "lead_profile": data.get("lead_profile", {}),
            # Read ai_paused from direct column first (set by frontend toggle),
            # then fallback to context_flags for backward compatibility (set by agent tools)
            "ai_paused": data.get("ai_paused") or data.get("context_flags", {}).get("ai_paused", False),
            "last_interaction": data.get("last_interaction"),
            "active_agent_id": data.get("active_agent_id"),
        }

    async def sync_missed_messages(self, lead_id: str, memory: dict) -> dict:
        """
        Sincroniza mensagens que aconteceram durante períodos de pausa/hands-on.
        Busca na tabela 'messages' tudo que ocorreu APÓS last_interaction do chat_memory.
        Retorna o memory atualizado com as mensagens faltantes injetadas.
        """
        last_interaction = memory.get("last_interaction")
        if not last_interaction:
            # Memória nova, sem histórico — nada para sincronizar
            return memory

        try:
            result = (
                self.supabase.table("messages")
                .select("content, direction, created_at")
                .eq("lead_id", lead_id)
                .gt("created_at", last_interaction)
                .in_("direction", ["inbound", "outbound", "outbound_manual"])
                .order("created_at", desc=False)
                .limit(30)
                .execute()
            )

            missed = result.data or []
            if not missed:
                return memory

            # Converter para formato de histórico do chat_memory
            missed_history = []
            for msg in missed:
                content = msg.get("content", "")
                if not content or not content.strip():
                    continue
                role = "user" if msg.get("direction") == "inbound" else "assistant"
                missed_history.append({"role": role, "content": content})

            if missed_history:
                history = list(memory.get("history", []))
                history.extend(missed_history)
                memory["history"] = history
                print(f"[Memory] 🔄 Synced {len(missed_history)} missed messages for lead {lead_id[:8]}...")

        except Exception as e:
            print(f"[Memory] ⚠️ Error syncing missed messages: {e}")

        return memory

    async def save(
        self,
        memory_id: str | None,
        lead_id: str,
        workspace_id: str,
        new_messages: list[dict],
        current_memory: dict,
        chat_id: str | None = None,
        llm=None,
    ):
        """Salva memória atualizada com compressão inteligente."""
        history = list(current_memory.get("history", []))
        history.extend(new_messages)

        summary = current_memory.get("summary")
        lead_profile = current_memory.get("lead_profile", {})

        # --- Compressão: Se > 20 mensagens, comprimir as mais antigas ---
        if len(history) > 20 and llm:
            oldest = history[:10]
            recent = history[10:]

            summary = await self._compress_history(oldest, summary, llm)
            # Extrair fatos do lead das mensagens mais antigas
            new_facts = await self._extract_lead_facts(oldest, lead_profile, llm)
            lead_profile.update(new_facts)

            history = recent

        update_data = {
            "conversation_history": history,
            "conversation_summary": summary,
            "lead_profile": lead_profile,
            "last_interaction": datetime.now(timezone.utc).isoformat(),
        }

        if memory_id:
            # Do not send ai_paused / pause_reason — a concurrent handoff
            # must not be overwritten by a history save.
            self.supabase.table("chat_memory").update(update_data).eq(
                "id", memory_id
            ).execute()
        else:
            update_data["lead_id"] = lead_id
            update_data["workspace_id"] = workspace_id
            update_data["chat_id"] = chat_id or lead_id
            # Upsert: transfer_to_human may have created the row on the
            # first message (UNIQUE chat_id, workspace_id).
            self.supabase.table("chat_memory").upsert(
                update_data,
                on_conflict="chat_id,workspace_id",
            ).execute()

    async def _compress_history(
        self, messages: list[dict], previous_summary: str | None, llm
    ) -> str:
        """Comprime histórico antigo em um resumo conciso."""
        formatted = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
        )

        prompt = f"""Tarefa: Resumir a conversa abaixo mantendo TODOS os fatos importantes.
Resumo anterior: {previous_summary or 'Nenhum'}

Novas mensagens:
{formatted}

Instruções:
- Crie um resumo de no máximo 5 linhas.
- MANTENHA: nomes, datas, produtos, preferências, preços mencionados, decisões tomadas.
- IGNORE: saudações, "obrigado", "ok", conversa fiada.
- Retorne APENAS o texto do resumo."""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    async def _extract_lead_facts(
        self, messages: list[dict], existing_profile: dict, llm
    ) -> dict:
        """Extrai fatos persistentes sobre o lead (memória de longo prazo)."""
        formatted = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
        )

        existing_str = "\n".join(f"- {k}: {v}" for k, v in existing_profile.items()) if existing_profile else "Nenhum"

        prompt = f"""Analise as mensagens abaixo e extraia FATOS NOVOS sobre o cliente.

Perfil existente:
{existing_str}

Mensagens:
{formatted}

Retorne um JSON com APENAS fatos NOVOS ou ATUALIZADOS. Exemplos de campos:
- "nome": nome do cliente
- "preferencia_horario": "manhã", "tarde", "noite"
- "produtos_interesse": ["produto A", "produto B"]
- "tom_preferido": "informal", "formal"
- "observacoes": "viaja dia 15", "tem urgência"
- "data_aniversario": se mencionada
- "empresa": se mencionada

Se não houver fatos novos, retorne {{}}
Retorne APENAS o JSON, sem explicações."""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            import json
            content = response.content.strip()
            # Limpa possíveis markdown code blocks
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception:
            return {}

    def format_for_prompt(self, memory: dict) -> str:
        """Formata a memória para inserção no system prompt."""
        parts = []

        if memory.get("summary"):
            parts.append(f"[📝 CONTEXTO ANTERIOR]\n{memory['summary']}")

        if memory.get("lead_profile"):
            profile_str = "\n".join(
                f"- {k}: {v}" for k, v in memory["lead_profile"].items()
            )
            parts.append(f"[👤 PERFIL DO CLIENTE]\n{profile_str}")

        return "\n\n".join(parts)

    def get_chat_messages(self, memory: dict, limit: int = 20) -> list:
        """Converte histórico em objetos LangChain Message."""
        messages = []
        history = memory.get("history", [])[-limit:]

        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        return messages

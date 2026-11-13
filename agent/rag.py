"""
============================================
RAG Engine v2 - Hybrid Search + LLM Reranking
============================================
Melhorias implementadas:
1. Busca híbrida: vetorial (OpenAI) + full-text (PostgreSQL)
2. Reciprocal Rank Fusion (RRF) para mesclar resultados
3. LLM Reranking via Gemini (com fallback keyword)
4. Formatação categorizada para o prompt
"""

import json
import httpx
from supabase import Client

from .usage import record_usage


class RAGEngine:
    """Motor de RAG com busca híbrida e reranking inteligente."""

    def __init__(self, supabase: Client, openai_api_key: str | None):
        self.supabase = supabase
        key = (openai_api_key or "").strip()
        self.openai_api_key = key if key and not key.startswith("AIza") else None

    async def search(
        self,
        workspace_id: str,
        query: str,
        agent_id: str | None = None,
        top_k: int = 8,
        threshold: float = 0.40,
        llm=None,
    ) -> str:
        """Busca híbrida na knowledge base: vetor + full-text + reranking."""
        try:
            # 1. Busca vetorial (semantic)
            vector_results = await self._vector_search(
                workspace_id, query, agent_id, top_k, threshold
            )

            # 2. Busca full-text (keyword)
            fulltext_results = await self._fulltext_search(
                workspace_id, query, agent_id, top_k
            )

            # 3. Merge com Reciprocal Rank Fusion (RRF)
            merged = self._reciprocal_rank_fusion(vector_results, fulltext_results)

            if not merged:
                print("[RAG] No results from hybrid search")
                return ""

            print(f"[RAG] Hybrid search: {len(vector_results)} vector + {len(fulltext_results)} fulltext → {len(merged)} merged")

            # 4. Reranking
            top_candidates = merged[:8]
            if llm and len(top_candidates) > 1:
                try:
                    reranked = await self._llm_rerank(top_candidates, query, llm)
                except Exception:
                    print("[RAG] LLM rerank failed, using keyword fallback")
                    reranked = self._keyword_rerank(top_candidates, query)
            else:
                reranked = self._keyword_rerank(top_candidates, query)

            # 5. Formatar top-5 para o prompt
            return self._format_results(reranked[:5])

        except Exception:
            print("[RAG] Error during search")
            return ""

    # ═══════════════════════════════════════════
    # SEARCH METHODS
    # ═══════════════════════════════════════════

    async def _vector_search(
        self,
        workspace_id: str,
        query: str,
        agent_id: str | None,
        top_k: int,
        threshold: float,
    ) -> list[dict]:
        """Busca semântica via embeddings."""
        if not self.openai_api_key:
            return []
        try:
            embedding = await self._generate_embedding(query, workspace_id)
            result = self.supabase.rpc(
                "match_knowledge_base",
                {
                    "query_embedding": f"[{','.join(str(e) for e in embedding)}]",
                    "p_workspace_id": workspace_id,
                    "p_agent_id": agent_id,
                    "match_threshold": threshold,
                    "match_count": top_k,
                },
            ).execute()
            return result.data or []
        except Exception:
            print("[RAG] Vector search error")
            return []

    async def _fulltext_search(
        self,
        workspace_id: str,
        query: str,
        agent_id: str | None,
        limit: int,
    ) -> list[dict]:
        """Busca por palavras-chave via PostgreSQL tsvector."""
        try:
            result = self.supabase.rpc(
                "fulltext_knowledge_base",
                {
                    "p_query": query,
                    "p_workspace_id": workspace_id,
                    "p_agent_id": agent_id,
                    "p_limit": limit,
                },
            ).execute()
            return result.data or []
        except Exception:
            print("[RAG] Full-text search error")
            return []

    # ═══════════════════════════════════════════
    # FUSION & RERANKING
    # ═══════════════════════════════════════════

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        fulltext_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) — combina dois rankings sem bias de escala.

        RRF score = Σ 1/(k + rank_i) para cada lista em que o item aparece.
        k=60 é o valor padrão da literatura (suaviza diferenças de posição).
        """
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        for rank, item in enumerate(vector_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank)
            items[item_id] = item

        for rank, item in enumerate(fulltext_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + 1.0 / (k + rank)
            items[item_id] = item

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for item_id in sorted_ids:
            entry = items[item_id].copy()
            entry["rrf_score"] = scores[item_id]
            result.append(entry)

        return result

    async def _llm_rerank(
        self,
        results: list[dict],
        query: str,
        llm,
    ) -> list[dict]:
        """Reranking via LLM — entende sinônimos e contexto semântico."""
        from langchain_core.messages import HumanMessage

        # Build compact summary of each result
        snippets = []
        for i, item in enumerate(results):
            title = item.get("title", "")
            content = (item.get("content", "") or "")[:200]
            snippets.append(f"{i+1}. [{title}]: {content}")

        prompt = f"""Classifique a relevância de cada trecho para responder a pergunta do cliente.
Retorne APENAS um JSON array com os índices ordenados do MAIS relevante ao MENOS relevante.

Pergunta: "{query}"

Trechos:
{chr(10).join(snippets)}

Retorne APENAS o JSON array de índices, ex: [3, 1, 5, 2, 4]. Nada mais."""

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Parse the JSON array of indices
        content = content.replace("```json", "").replace("```", "").strip()
        indices = json.loads(content)

        reranked = []
        seen = set()
        for idx in indices:
            pos = idx - 1  # convert 1-based to 0-based
            if 0 <= pos < len(results) and pos not in seen:
                reranked.append(results[pos])
                seen.add(pos)

        # Add any items not ranked by LLM
        for i, item in enumerate(results):
            if i not in seen:
                reranked.append(item)

        return reranked

    def _keyword_rerank(self, results: list[dict], query: str) -> list[dict]:
        """Fallback: reranking por overlap de palavras-chave."""
        query_words = set(query.lower().split())

        for item in results:
            content = (item.get("content", "") + " " + item.get("title", "")).lower()
            content_words = set(content.split())

            rrf_score = item.get("rrf_score", 0)
            keyword_overlap = len(query_words & content_words) / max(len(query_words), 1)

            item["final_score"] = (rrf_score * 0.7) + (keyword_overlap * 0.3)

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    # ═══════════════════════════════════════════
    # OUTPUT FORMATTING
    # ═══════════════════════════════════════════

    def _format_results(self, results: list[dict]) -> str:
        """Formata resultados por categoria para inserção no prompt."""
        if not results:
            return ""

        category_map: dict[str, list[str]] = {}

        for item in results:
            category = item.get("category", "Geral")
            if category not in category_map:
                category_map[category] = []
            title = item.get("title", "")
            content = item.get("content", "")
            category_map[category].append(f"**{title}**: {content}")

        parts = []
        for category, items in category_map.items():
            parts.append(f"### {category.upper()}")
            parts.extend(items)
            parts.append("")

        return "\n".join(parts)

    # ═══════════════════════════════════════════
    # EMBEDDING GENERATION
    # ═══════════════════════════════════════════

    async def _generate_embedding(self, text: str, workspace_id: str) -> list[float]:
        """Gera embedding usando OpenAI text-embedding-3-large (1536 dims)."""
        url = "https://api.openai.com/v1/embeddings"

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "text-embedding-3-large",
                    "dimensions": 1536,
                    "input": text[:8000],
                },
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            if "prompt_tokens" in usage:
                await record_usage(self.supabase, workspace_id, "openai", "text-embedding-3-large", usage["prompt_tokens"], 0)
            return data["data"][0]["embedding"]

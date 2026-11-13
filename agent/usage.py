"""Best-effort accounting for successful provider calls."""
import asyncio
import logging

from langchain_core.callbacks import AsyncCallbackHandler

log = logging.getLogger(__name__)
PRICES = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "text-embedding-3-large": (0.13, 0.0),
}


async def record_usage(supabase, workspace_id, provider, model, prompt_tokens, completion_tokens):
    """Write only provider-reported usage; telemetry must never break a response."""
    try:
        input_price, output_price = PRICES.get(model, (0, 0))
        row = {
            "workspace_id": workspace_id, "function_name": "python-agent",
            "provider": provider, "model": model,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "estimated_cost_usd": (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000,
        }
        await asyncio.to_thread(lambda: supabase.table("ai_usage_logs").insert(row).execute())
    except Exception:
        log.warning("Failed to record AI usage", exc_info=True)


class UsageCallback(AsyncCallbackHandler):
    def __init__(self, supabase, workspace_id):
        self.supabase = supabase
        self.workspace_id = workspace_id

    async def on_llm_end(self, response, **kwargs):
        try:
            for generation_list in response.generations:
                for generation in generation_list:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None) or {}
                    metadata = getattr(message, "response_metadata", None) or {}
                    model = metadata.get("model_name") or (generation.generation_info or {}).get("model_name") or (response.llm_output or {}).get("model_name")
                    if model and "input_tokens" in usage and "output_tokens" in usage:
                        await record_usage(self.supabase, self.workspace_id, "gemini", model,
                                           usage["input_tokens"], usage["output_tokens"])
        except Exception:
            log.warning("Failed to read LLM usage", exc_info=True)

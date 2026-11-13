"""Best-effort accounting for successful provider calls."""
import asyncio
import logging
import concurrent.futures
import threading

from langchain_core.callbacks import AsyncCallbackHandler

log = logging.getLogger(__name__)
PRICES = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "text-embedding-3-large": (0.13, 0.0),
}


# A stuck synchronous client consumes at most one of these daemon workers.
_USAGE_SLOTS = threading.BoundedSemaphore(4)
_USAGE_DEADLINE = 0.25


async def record_usage(supabase, workspace_id, provider, model, prompt_tokens, completion_tokens):
    """Write only provider-reported usage; telemetry must never break a response."""
    try:
        price_model = model.removeprefix("models/").removeprefix("google/")
        prices = PRICES.get(price_model)
        cost = ((prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000
                if prices is not None else None)
        row = {
            "workspace_id": workspace_id, "function_name": "python-agent",
            "provider": provider, "model": model,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "estimated_cost_usd": cost,
        }
        if not _USAGE_SLOTS.acquire(blocking=False):
            log.warning("AI usage logger at capacity; dropping event")
            return
        future = concurrent.futures.Future()

        def write():
            try:
                supabase.table("ai_usage_logs").insert(row).execute()
                future.set_result(True)
            except Exception:
                future.set_result(False)
            finally:
                _USAGE_SLOTS.release()

        try:
            threading.Thread(target=write, daemon=True, name="ai-usage").start()
        except Exception:
            _USAGE_SLOTS.release()
            raise
        if not await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), _USAGE_DEADLINE):
            log.warning("Failed to record AI usage")
    except asyncio.TimeoutError:
        log.warning("AI usage logger timed out")
    except Exception:
        log.warning("Failed to record AI usage")


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
            log.warning("Failed to read LLM usage")

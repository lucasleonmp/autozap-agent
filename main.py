"""
============================================
Autozap Agent Service - FastAPI Server
============================================
"""

import os
import sys
import time
import logging
import traceback
import asyncio

# Load .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("autozap")

# ─── App ───
app = FastAPI(title="Autozap Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───
class ProcessRequest(BaseModel):
    lead_id: str
    chat_id: str | None = None
    workspace_id: str
    message: str
    agent_config: dict = {}
    instance_id: str | None = None
    ai_api_key: str | None = None
    ai_model: str | None = None
    knowledge_context: str | None = None


class ProcessResponse(BaseModel):
    response: str | None
    status: str
    tools_used: list[str] = []
    latency_ms: float = 0


# ─── Agent (lazy) ───
_agents: dict = {}
_agent_class = None


def _get_agent_class():
    global _agent_class
    if _agent_class is None:
        logger.info("Importing AutozapAgent...")
        from agent.graph import AutozapAgent
        _agent_class = AutozapAgent
        logger.info("AutozapAgent imported OK")
    return _agent_class


def get_agent(ai_api_key=None, ai_model=None):
    AgentClass = _get_agent_class()
    key = f"{ai_api_key or 'env'}:{ai_model or 'default'}"
    if key not in _agents:
        _agents[key] = AgentClass(ai_api_key=ai_api_key, ai_model=ai_model)
    return _agents[key]


# ─── Routes ───
@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "autozap-agent", "version": "1.0.0"}


def _require_agent_secret(request: Request) -> None:
    agent_secret = os.environ.get("AGENT_SECRET", "")
    if not agent_secret:
        return
    auth = request.headers.get("authorization", "")
    if auth.replace("Bearer ", "") != agent_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/transcode/voice")
async def transcode_voice_route(request: Request):
    _require_agent_secret(request)
    from agent.voice_transcode import VoiceTranscodeError, transcode_voice

    content_type = (request.headers.get("content-type") or "").lower()
    mime_type = request.headers.get("content-type")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if hasattr(upload, "read"):
            raw = await upload.read()
            mime_type = getattr(upload, "content_type", None) or form.get("mime_type") or mime_type
        else:
            raw = b""
    else:
        raw = await request.body()

    try:
        output = await asyncio.to_thread(transcode_voice, raw, mime_type)
    except VoiceTranscodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    return Response(content=output, media_type="audio/ogg")


@app.post("/process", response_model=ProcessResponse)
async def process_message(req: ProcessRequest, request: Request):
    _require_agent_secret(request)

    start = time.time()
    try:
        agent_instance = get_agent(req.ai_api_key, req.ai_model)
        result = await agent_instance.process(
            lead_id=req.lead_id,
            chat_id=req.chat_id,
            workspace_id=req.workspace_id,
            message=req.message,
            agent_config=req.agent_config,
            instance_id=req.instance_id,
            edge_knowledge_context=req.knowledge_context,
        )
        ms = (time.time() - start) * 1000
        logger.info(f"lead={req.lead_id[:8]}... tools={result.get('tools_used',[])} {ms:.0f}ms")
        return ProcessResponse(
            response=result.get("response"),
            status=result.get("status", "success"),
            tools_used=result.get("tools_used", []),
            latency_ms=round(ms, 1),
        )
    except Exception as e:
        logger.error(f"Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Startup ───
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)

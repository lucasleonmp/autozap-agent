from fastapi.testclient import TestClient
import asyncio
import time

import httpx

from main import app


def test_route_rejects_missing_bearer(monkeypatch):
    monkeypatch.setenv("AGENT_SECRET", "secret")
    client = TestClient(app)
    res = client.post(
        "/transcode/voice",
        content=b"OggS" + b"\x00" * 8,
        headers={"content-type": "audio/ogg"},
    )
    assert res.status_code == 401


def test_route_ogg_passthrough_with_bearer(monkeypatch):
    monkeypatch.setenv("AGENT_SECRET", "secret")
    client = TestClient(app)
    raw = b"OggS" + b"\x00" * 8
    res = client.post(
        "/transcode/voice",
        content=raw,
        headers={
            "content-type": "audio/ogg",
            "authorization": "Bearer secret",
        },
    )
    assert res.status_code == 200
    assert res.content == raw
    assert res.headers["content-type"].startswith("audio/ogg")


def test_route_garbage_fails_closed(monkeypatch):
    monkeypatch.setenv("AGENT_SECRET", "secret")
    client = TestClient(app)
    res = client.post(
        "/transcode/voice",
        content=b"not-an-audio-file",
        headers={
            "content-type": "audio/mp4",
            "authorization": "Bearer secret",
        },
    )
    assert res.status_code == 422
    body = res.json()
    assert body["detail"]["code"] == "AZ4_VOICE_TRANSCODE_FAILED"


def test_health_completes_while_transcode_runs(monkeypatch):
    monkeypatch.setenv("AGENT_SECRET", "secret")

    def slow(_raw, _mime=None):
        time.sleep(0.4)
        return b"OggS" + b"\x00" * 8

    monkeypatch.setattr("agent.voice_transcode.transcode_voice", slow)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            transcode = asyncio.create_task(
                client.post(
                    "/transcode/voice",
                    content=b"not-ogg",
                    headers={
                        "content-type": "audio/mp4",
                        "authorization": "Bearer secret",
                    },
                )
            )
            await asyncio.sleep(0.05)
            started = time.perf_counter()
            health = await client.get("/health")
            elapsed_ms = (time.perf_counter() - started) * 1000
            assert health.status_code == 200
            assert elapsed_ms < 200, elapsed_ms
            transcoded = await transcode
            assert transcoded.status_code == 200

    asyncio.run(run())

import shutil
import subprocess

import pytest

from agent.voice_transcode import (
    MAX_DURATION_SECONDS,
    MAX_INPUT_BYTES,
    VoiceTranscodeError,
    transcode_voice,
)


HAS_FFMPEG = shutil.which("ffmpeg") is not None
HAS_FFPROBE = shutil.which("ffprobe") is not None


class DummyProc:
    def __init__(self, returncode=0, stderr=b""):
        self.returncode = returncode
        self._stderr = stderr

    def communicate(self, timeout=None):
        return b"", self._stderr

    def send_signal(self, _sig):
        return None

    def wait(self):
        return self.returncode


def first_ogg_payload(ogg: bytes) -> bytes:
    nsegs = ogg[26]
    header_size = 27 + nsegs
    payload_size = sum(ogg[27:header_size])
    return ogg[header_size:header_size + payload_size]


def ffmpeg_bytes(fmt: str, codec: str) -> bytes:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/in.{fmt}"
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=0.3",
                "-c:a",
                codec,
                "-ac",
                "1",
                "-f",
                fmt,
                path,
            ],
            check=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        with open(path, "rb") as handle:
            return handle.read()


def probe_codec(path: str) -> tuple[str, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels",
            "-of",
            "csv=p=0",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    codec, channels = result.stdout.strip().split(",")
    return codec, channels


def test_ogg_passthrough_does_not_run_ffmpeg(monkeypatch):
    def fake_popen(*_args, **_kwargs):
        raise AssertionError("ffmpeg must not run for OggS input")

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    raw = b"OggS" + b"\x00" * 16
    out = transcode_voice(raw, "audio/ogg")
    assert out == raw


def test_webm_becomes_ogg_opus_mono(monkeypatch):
    captured = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        open(cmd[-1], "wb").write(b"OggS" + b"\x00" * 20 + b"OpusHead")
        return DummyProc()

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 10.0)
    out = transcode_voice(b"webm-bytes", "audio/webm")
    assert out.startswith(b"OggS")
    assert "libopus" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-ac") + 1] == "1"
    assert captured["cmd"][captured["cmd"].index("-f") + 1] == "ogg"


def test_safari_mp4_uses_mp4_suffix(monkeypatch):
    captured = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        open(cmd[-1], "wb").write(b"OggS" + b"\x00" * 20 + b"OpusHead")
        return DummyProc()

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 8.0)
    out = transcode_voice(b"mp4-bytes", "audio/mp4")
    assert out.startswith(b"OggS")
    assert captured["cmd"][captured["cmd"].index("-i") + 1].endswith(".mp4")


def test_wav_uses_wav_suffix(monkeypatch):
    captured = {}

    def fake_popen(cmd, stdout=None, stderr=None):
        captured["cmd"] = cmd
        open(cmd[-1], "wb").write(b"OggS" + b"\x00" * 20 + b"OpusHead")
        return DummyProc()

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 4.0)
    out = transcode_voice(b"RIFF....WAVEfmt ", "audio/wav")
    assert out.startswith(b"OggS")
    assert captured["cmd"][captured["cmd"].index("-i") + 1].endswith(".wav")


def test_garbage_input_fails_closed(monkeypatch):
    def fake_popen(cmd, stdout=None, stderr=None):
        return DummyProc(returncode=1, stderr=b"invalid data")

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 1.0)
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"not-an-audio-file", "audio/mp4")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"


def test_non_ogg_encoder_output_fails_closed(monkeypatch):
    def fake_popen(cmd, stdout=None, stderr=None):
        open(cmd[-1], "wb").write(b"ID3not-ogg")
        return DummyProc()

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 1.0)
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"webm-bytes", "audio/webm")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"


def test_20mb_file_is_rejected():
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"x" * (MAX_INPUT_BYTES + 1), "audio/webm")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"


def test_400_second_audio_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "agent.voice_transcode._probe_duration",
        lambda _path: MAX_DURATION_SECONDS + 100,
    )

    def fake_popen(*_args, **_kwargs):
        raise AssertionError("ffmpeg must not run")

    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", fake_popen)
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"short", "audio/webm")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"


def test_encoder_timeout_returns_failed_code(monkeypatch):
    import subprocess as sp

    class SlowProc(DummyProc):
        def communicate(self, timeout=None):
            raise sp.TimeoutExpired(cmd="ffmpeg", timeout=timeout)

    monkeypatch.setattr("agent.voice_transcode._probe_duration", lambda _path: 2.0)
    monkeypatch.setattr("agent.voice_transcode.subprocess.Popen", lambda *a, **k: SlowProc())
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"bytes", "audio/webm")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    "mime,fmt,codec",
    [
        ("audio/webm", "webm", "libopus"),
        ("audio/mp4", "mp4", "aac"),
        ("audio/wav", "wav", "pcm_s16le"),
    ],
)
def test_real_ffmpeg_emits_ogg_opus(mime, fmt, codec, tmp_path):
    raw = ffmpeg_bytes(fmt, codec)
    out = transcode_voice(raw, mime)
    assert out.startswith(b"OggS")
    assert first_ogg_payload(out).startswith(b"OpusHead")
    if not HAS_FFPROBE:
        return
    path = tmp_path / "out.ogg"
    path.write_bytes(out)
    probed, channels = probe_codec(str(path))
    assert probed == "opus"
    assert channels == "1"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_ogg_passthrough_keeps_bytes():
    raw = ffmpeg_bytes("ogg", "libopus")
    assert raw.startswith(b"OggS")
    out = transcode_voice(raw, "audio/ogg")
    assert out == raw


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_real_garbage_fails_closed():
    with pytest.raises(VoiceTranscodeError) as err:
        transcode_voice(b"this is not audio", "audio/mp4")
    assert err.value.code == "AZ4_VOICE_TRANSCODE_FAILED"

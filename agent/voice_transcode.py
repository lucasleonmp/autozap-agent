import signal
import subprocess
import tempfile
from pathlib import Path

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_DURATION_SECONDS = 300
ENCODER_TIMEOUT_SECONDS = 20
OGG_MAGIC = b"OggS"

FFMPEG_ARGS = [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
]


class VoiceTranscodeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def looks_like_ogg(raw: bytes) -> bool:
    return len(raw) >= 4 and raw[:4] == OGG_MAGIC


def _probe_duration(path: str) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return None


def _input_suffix(mime_type: str | None) -> str:
    lowered = (mime_type or "").lower()
    if "mp4" in lowered or "m4a" in lowered or "aac" in lowered:
        return ".mp4"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "wav" in lowered:
        return ".wav"
    return ".webm"


def transcode_voice(raw: bytes, mime_type: str | None = None) -> bytes:
    if len(raw) > MAX_INPUT_BYTES:
        raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", "input exceeds 16 MB")
    if looks_like_ogg(raw):
        return raw

    suffix = _input_suffix(mime_type)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.ogg"
        src.write_bytes(raw)
        duration = _probe_duration(str(src))
        if duration is not None and duration > MAX_DURATION_SECONDS:
            raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", "audio exceeds 300 seconds")

        cmd = [
            "ffmpeg",
            *FFMPEG_ARGS,
            "-i",
            str(src),
            "-c:a",
            "libopus",
            "-ac",
            "1",
            "-b:a",
            "24k",
            "-application",
            "voip",
            "-f",
            "ogg",
            str(dst),
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                _stdout, stderr = proc.communicate(timeout=ENCODER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGKILL)
                proc.wait()
                raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", "encoder timeout") from None
        except FileNotFoundError as exc:
            raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", "ffmpeg missing") from exc

        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            detail = (stderr or b"").decode("utf-8", errors="replace")[:180]
            raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", detail or "empty encoder output")

        output = dst.read_bytes()
        if not looks_like_ogg(output):
            raise VoiceTranscodeError("AZ4_VOICE_TRANSCODE_FAILED", "encoder output is not Ogg")
        return output

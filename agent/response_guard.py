"""
Guards against AI "silence narration" and looped output.

When a customer instructs the agent not to reply to images/media, models often
emit meta text like "(Sem resposta, conforme instrução...)" — sometimes looped.
That must never be delivered to WhatsApp.
"""

from __future__ import annotations

import re

MEDIA_PLACEHOLDER_RE = re.compile(
    r"^\[(Imagem|Image|Áudio|Audio|V[íi]deo|Video|Sticker|Figurinha|Documento|Arquivo|Document)[^\]]*\]$",
    re.IGNORECASE,
)

NO_REPLY_TOKEN_RE = re.compile(r"^\[NO_REPLY\]$", re.IGNORECASE)

SILENCE_META_RE = re.compile(
    r"\(?\s*sem\s+resposta\b|"
    r"conforme\s+instru[cç][aã]o.{0,80}n[aã]o\s+responder|"
    r"n[aã]o\s+(vou|devo|posso|irei)\s+responder.{0,40}"
    r"(imagem|image|m[íi]dia|media|áudio|audio|v[íi]deo|video|sticker)",
    re.IGNORECASE | re.DOTALL,
)

IGNORE_IMAGE_INSTRUCTION_RE = re.compile(
    r"n[aã]o\s+respon(der|da|de).{0,40}(a\s+)?(imagem|imagens|foto|fotos)|"
    r"n[aã]o\s+responder?\s+a\s+imagens|"
    r"(imagem|imagens|foto|fotos).{0,40}n[aã]o\s+respon|"
    r"(ignore|ignorar|skip).{0,20}(imagem|image|foto)|"
    r"do\s+not\s+respond.{0,40}image|"
    r"never\s+respond.{0,40}image",
    re.IGNORECASE | re.DOTALL,
)

IGNORE_MEDIA_INSTRUCTION_RE = re.compile(
    r"n[aã]o\s+respon(der|da|de).{0,40}(m[íi]dia|media|sticker|figurinha|áudio|audio|v[íi]deo|video)|"
    r"(ignore|ignorar|skip).{0,20}(m[íi]dia|media)|"
    r"do\s+not\s+respond.{0,40}media",
    re.IGNORECASE | re.DOTALL,
)

REPEATED_CHUNK_RE = re.compile(r"^(.{12,160}?)\1{2,}", re.DOTALL)
PHRASE_CHUNK_RE = re.compile(r"\([^)]{10,120}\)|[^.!?\n]{15,120}")


def is_media_only_placeholder(content: str | None) -> bool:
    if not content or not content.strip():
        return False
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return False
    return all(MEDIA_PLACEHOLDER_RE.match(line) for line in lines)


def media_kind_from_placeholder(content: str) -> str:
    sample = content.lower()
    if re.search(r"imagem|image|foto", sample):
        return "image"
    if re.search(r"áudio|audio", sample):
        return "audio"
    if re.search(r"v[íi]deo|video", sample):
        return "video"
    if re.search(r"sticker|figurinha", sample):
        return "sticker"
    if re.search(r"documento|arquivo|document", sample):
        return "document"
    return "media"


def agent_instructs_silence_for_media(system_prompt: str | None, message_content: str) -> bool:
    if not system_prompt or not system_prompt.strip():
        return False
    if not is_media_only_placeholder(message_content):
        return False
    kind = media_kind_from_placeholder(message_content)
    if kind == "image":
        return bool(
            IGNORE_IMAGE_INSTRUCTION_RE.search(system_prompt)
            or IGNORE_MEDIA_INSTRUCTION_RE.search(system_prompt)
        )
    return bool(
        IGNORE_MEDIA_INSTRUCTION_RE.search(system_prompt)
        or IGNORE_IMAGE_INSTRUCTION_RE.search(system_prompt)
    )


def is_no_reply_meta_response(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip()
    if NO_REPLY_TOKEN_RE.match(t):
        return True
    if SILENCE_META_RE.search(t):
        return True
    return False


def is_highly_repetitive_response(text: str | None) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < 60:
        return False
    if REPEATED_CHUNK_RE.match(t):
        return True
    counts: dict[str, int] = {}
    for raw in PHRASE_CHUNK_RE.findall(t):
        key = raw.strip().lower()
        if len(key) < 12:
            continue
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= 4:
            return True
    return False


def should_suppress_ai_response(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    if is_no_reply_meta_response(text):
        return True
    if is_highly_repetitive_response(text):
        return True
    return False

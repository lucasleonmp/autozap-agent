from agent.response_guard import (
    agent_instructs_silence_for_media,
    is_highly_repetitive_response,
    is_media_only_placeholder,
    is_no_reply_meta_response,
    should_suppress_ai_response,
)


def test_media_only_placeholder():
    assert is_media_only_placeholder("[Imagem 📷]")
    assert is_media_only_placeholder("[Imagem 📷]\n[Imagem 📷]")
    assert not is_media_only_placeholder("[Imagem 📷]\nOlha isso")
    assert not is_media_only_placeholder("oi")


def test_agent_instructs_silence_for_images():
    prompt = "Atenda clientes. Não responder a imagens. Seja objetiva."
    assert agent_instructs_silence_for_media(prompt, "[Imagem 📷]")
    assert not agent_instructs_silence_for_media(prompt, "quanto custa?")
    assert not agent_instructs_silence_for_media("Atenda clientes.", "[Imagem 📷]")


def test_no_reply_meta_and_loop():
    meta = "(Sem resposta, conforme instrução para não responder a imagens.)"
    assert is_no_reply_meta_response(meta)
    assert should_suppress_ai_response(meta)
    assert should_suppress_ai_response("[NO_REPLY]")
    looped = meta * 20
    assert is_highly_repetitive_response(looped)
    assert should_suppress_ai_response(looped)
    assert not should_suppress_ai_response("Oi! Como posso te ajudar?")

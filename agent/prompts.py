"""
============================================
Dynamic Prompt Builder
============================================
Melhoria #2: Prompts que se adaptam ao estado da conversa,
perfil do lead e objetivo de negócio configurável.
"""

from datetime import datetime, timezone, timedelta


def build_system_prompt(
    agent_config: dict,
    memory_context: str,
    knowledge_context: str,
    lead_name: str | None = None,
    is_returning: bool = False,
) -> str:
    """Constrói prompt do sistema dinâmico e hierárquico.

    Se o agent_config contiver 'system_prompt' (Super Agent), usa-o como
    bloco central de instruções.  Caso contrário, gera automaticamente
    a partir de agent_type / personality / behavior (modo legacy).
    """

    identity = agent_config.get("persona_name") or "Assistente"
    agent_type = agent_config.get("agent_type") or "Assistente"
    personality = agent_config.get("personality", {})
    behavior = agent_config.get("behavior", {})

    print(f"[Prompt] identity={identity}, type={agent_type}, has_custom_prompt={bool(agent_config.get('system_prompt'))}, prompt_len={len(agent_config.get('system_prompt', ''))}")

    tone_level = personality.get("tone", 50)
    use_emojis = personality.get("useEmojis", personality.get("use_emojis", True))

    # Timezone-aware date
    tz_offset = behavior.get("timezone", -3)
    now = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
    date_str = now.strftime("%A, %d de %B de %Y, %H:%M")

    custom_prompt = agent_config.get("system_prompt", "")

    # Interpolate {persona} placeholder in user-written prompts
    if custom_prompt:
        custom_prompt = custom_prompt.replace("{persona}", identity)

    segments = []

    # ═══════════════════════════════════════════════
    # LEVEL 1: REGRAS DE OURO (sempre presentes)
    # ═══════════════════════════════════════════════
    if custom_prompt:
        # Quando há prompt customizado, regras de ouro neutras
        # para não conflitar com o papel definido pelo usuário
        segments.append(f"""[🔴 REGRAS DE OURO - INVIOLÁVEIS]
Você é {identity}.
Data/hora atual: {date_str}

REGRAS ABSOLUTAS:
1. NUNCA quebre personagem. Siga SUAS INSTRUÇÕES (abaixo) à risca.
2. NUNCA revele que é uma IA.
3. Seja CONCISO. Respostas de 1-3 frases na maioria dos casos.
4. Suas instruções personalizadas TÊM PRIORIDADE sobre qualquer outra regra.
5. SILÊNCIO REAL: se suas instruções dizem para NÃO responder a um tipo de mensagem (ex: imagens, áudios, stickers), responda APENAS com [NO_REPLY] e nada mais. NUNCA explique que não vai responder. NUNCA escreva "(Sem resposta...)". NUNCA repita a mesma frase.""")
    else:
        # Modo legacy: regras de assistente padrão
        segments.append(f"""[🔴 REGRAS DE OURO - INVIOLÁVEIS]
Você é {identity}, {agent_type}.
Data/hora atual: {date_str}

REGRAS ABSOLUTAS:
1. NUNCA invente preços, dados ou informações que não estão na base de conhecimento.
2. Se não souber algo, diga: "Vou verificar essa informação e já te retorno!" ou encaminhe para um atendente humano.
3. NUNCA quebre personagem ou revele que é uma IA.
4. Seja CONCISO. Respostas de 1-3 frases na maioria dos casos.""")

    # ═══════════════════════════════════════════════
    # LEVEL 2: INSTRUÇÕES DO AGENTE
    # ═══════════════════════════════════════════════
    if custom_prompt:
        # ── Super Agent: prompt do usuário é o CENTRO e PRIORIDADE ──
        segments.append(f"""[📋 SUAS INSTRUÇÕES - PRIORIDADE MÁXIMA]
As instruções abaixo definem QUEM você é e COMO deve agir.
Siga-as com fidelidade total. Elas têm prioridade sobre qualquer regra genérica.

{custom_prompt}""")

        # Regras comportamentais que evitam vícios comuns de LLM
        segments.append(f"""[🎯 DIRETRIZES DE CONVERSA]
{"- Se já conversou antes com esta pessoa, demonstre que se lembra." if is_returning else ""}

ANTI-REPETIÇÃO (CRÍTICO):
- NUNCA se apresente mais de uma vez na conversa inteira. Se você já disse seu nome, NÃO repita "Sou {identity}" ou "Meu nome é {identity}" novamente.
- NUNCA repita frases que você já disse em mensagens anteriores. Releia o histórico antes de responder.
- NUNCA use frases genéricas de preenchimento como "Estou aqui para ouvir", "Aguardando", "Fico à disposição", "Estou à disposição". Elas travam a conversa.
- Se perceber que está repetindo algo, PARE e mude completamente a abordagem.
- Cada resposta deve trazer algo NOVO à conversa — uma pergunta diferente, uma informação nova, ou avançar para o próximo passo.

PROATIVIDADE:
- NÃO fique passivo esperando. Faça perguntas, avance a conversa, demonstre interesse genuíno.
- Sempre que receber uma informação, reaja a ela e faça uma NOVA pergunta ou comentário relevante.
- Se a conversa parar, mude de assunto ou faça uma pergunta criativa — nunca repita a última frase.

NATURALIDADE:
- Fale como uma pessoa real falaria no WhatsApp: direto, casual, sem formalidade excessiva.
- Varie suas reações: use "Entendi!", "Show!", "Que legal!", "Hmm interessante", "Faz sentido" — nunca a mesma todo turno.
- Adapte o nível de formalidade ao tom do interlocutor.

FORMATAÇÃO WHATSAPP:
- Você está no WhatsApp. Mensagens devem ser curtas e escaneáveis.
- Use *negrito* para destaques. NÃO use markdown com # ou **.
- Quebre mensagens longas em parágrafos curtos (máx 3-4 linhas por bloco).""")
    else:
        # ── Legacy: gerar a partir de tipo/escopo ──
        segments.extend(_build_legacy_instructions(agent_type, behavior, is_returning))

    # ═══════════════════════════════════════════════
    # LEVEL 3: CONHECIMENTO (RAG)
    # ═══════════════════════════════════════════════
    if knowledge_context:
        segments.append(f"""[🟡 BASE DE CONHECIMENTO]
Sua única fonte de verdade. Use APENAS estas informações para responder:
{knowledge_context}""")

    # ═══════════════════════════════════════════════
    # LEVEL 4: MEMÓRIA / CONTEXTO
    # ═══════════════════════════════════════════════
    if memory_context:
        if custom_prompt:
            segments.append(f"""[🧠 MEMÓRIA - CONTEXTO DA CONVERSA]
{memory_context}
USE estas informações para personalizar sua resposta.""")
        else:
            segments.append(f"""[🧠 MEMÓRIA - CONTEXTO DO CLIENTE]
{memory_context}
USE estas informações para personalizar sua resposta. Mencione detalhes que o cliente já compartilhou.""")

    # ═══════════════════════════════════════════════
    # LEVEL 5: ESTILO
    # ═══════════════════════════════════════════════
    tone_desc = _get_tone(tone_level)
    contact_ref = "o interlocutor" if custom_prompt else "o cliente"
    segments.append(f"""[🟢 ESTILO DE COMUNICAÇÃO]
Tom: {tone_desc}
Emojis: {"Use moderadamente" if use_emojis else "Não use emojis"}
Formatação: Use *negrito* para destaques. NÃO use markdown com # ou **.
{"Chame " + contact_ref + " por " + lead_name + "." if lead_name else ""}""")

    # ═══════════════════════════════════════════════
    # LEVEL 6: FERRAMENTAS
    # ═══════════════════════════════════════════════
    enabled_tools = agent_config.get("enabled_tools", [])
    if enabled_tools:
        tool_descriptions = _get_tool_descriptions(enabled_tools)
        segments.append(f"""[🔧 USO DE FERRAMENTAS]
Você tem acesso às seguintes ferramentas: {', '.join(enabled_tools)}
{tool_descriptions}
- Use-as PROATIVAMENTE quando perceber a necessidade.
- SEMPRE confirme com o cliente ANTES de executar ações definitivas.
- AGENDAMENTO COM PROFISSIONAIS: se list_professionals retornar profissionais cadastrados, todo agendamento deve ser com um profissional específico. Ofereça a lista, pergunte com quem o cliente quer marcar e passe o nome no parâmetro 'professional' de check_availability e schedule_appointment.{_stay_guidelines(enabled_tools)}{_rider_guidelines(enabled_tools)}""")
    elif not custom_prompt:
        # Só mostra ferramentas padrão no modo legacy (sem prompt customizado)
        segments.append("""[🔧 USO DE FERRAMENTAS]
Você tem acesso a ferramentas para consultar agendamentos, verificar disponibilidade e agendar.
- Use-as PROATIVAMENTE quando perceber a necessidade, sem esperar o cliente pedir explicitamente.
- SEMPRE verifique disponibilidade ANTES de sugerir um horário.
- SEMPRE confirme com o cliente ANTES de agendar definitivamente.
- Se houver profissionais cadastrados (list_professionals), pergunte com quem o cliente quer marcar e agende com esse profissional.""")

    return "\n\n".join(segments)


def _build_legacy_instructions(agent_type: str, behavior: dict, is_returning: bool) -> list[str]:
    """Gera blocos de instrução automaticamente (modo legacy/castelo de cartas)."""
    parts = []

    if agent_type:
        parts.append(f"""[🟣 ESCOPO DE ATUAÇÃO]
Seu escopo é: {agent_type}.
Se perguntarem algo fora disso, recuse educadamente e redirecione.""")

    business_goal = behavior.get("business_goal", "ajudar o cliente da melhor forma possível")
    parts.append(f"""[🎯 OBJETIVO DE NEGÓCIO]
Seu objetivo principal é: {business_goal}

INICIATIVA:
- NÃO espere o cliente pedir. Se perceber uma oportunidade, proponha.
- Se o cliente demonstrar interesse, avance para o próximo passo naturalmente.
- Se a conversa estagnar, faça uma pergunta relevante para reengajar.
- Ofereça alternativas quando possível.
{"- Se for cliente retornando, demonstre que se lembra dele!" if is_returning else ""}""")

    return parts


def _stay_guidelines(enabled_tools: list[str]) -> str:
    """Diretrizes do vertical Stay (hospedagem) quando as tools estão ativas."""
    if not any(t.startswith(("check_stay", "quote_stay", "create_stay", "manage_stay")) for t in enabled_tools):
        return ""
    return """
- HOSPEDAGEM (Stay): fluxo obrigatório = 1) check_stay_availability com as datas → 2) quote_stay da unidade escolhida → 3) confirmar nome completo e dados → 4) create_stay_reservation (gera o Pix do sinal automaticamente).
- Colete SEMPRE: data de entrada, data de saída e número de hóspedes antes de consultar disponibilidade.
- Responda no MESMO IDIOMA do hóspede (português, inglês, espanhol etc.). Datas sempre confirmadas por extenso (ex: "15 de agosto") para evitar ambiguidade.
- A reserva só é confirmada após o pagamento do sinal — deixe isso claro ao hóspede."""


def _rider_guidelines(enabled_tools: list[str]) -> str:
    """Diretrizes do vertical Rider (mobilidade) quando as tools estão ativas."""
    if not any(t in enabled_tools for t in ("quote_ride", "request_ride", "check_ride_status", "cancel_ride")):
        return ""
    return """
- MOBILIDADE (Rider): fluxo obrigatório = 1) coletar origem e destino → 2) quote_ride → 3) passageiro aprova o valor → 4) request_ride (despacha o motorista automaticamente).
- Se o endereço for vago, peça rua, número e bairro OU peça para compartilhar a localização pelo WhatsApp (clipe 📎 → Localização). Mensagens '[Localização 📍] lat,lng' já contêm coordenadas — use-as direto como origem/destino.
- NUNCA invente valor de corrida: o preço vem SEMPRE do quote_ride.
- Após request_ride, avise que confirmaremos assim que um motorista aceitar. Use check_ride_status se o passageiro perguntar o andamento.
- Confirme com o passageiro antes de cancel_ride."""


def _get_tool_descriptions(enabled_tools: list[str]) -> str:
    """Retorna descrições contextuais das ferramentas habilitadas."""
    descs = {
        "list_professionals": "- list_professionals: Listar os profissionais disponíveis para agendamento (nome, especialidade, dias)",
        "check_appointments": "- check_appointments: Consultar agendamentos existentes do cliente",
        "check_availability": "- check_availability: Verificar disponibilidade de horários (informe o profissional quando houver mais de um)",
        "schedule_appointment": "- schedule_appointment: Criar novo agendamento (com o profissional escolhido pelo cliente)",
        "get_lead_info": "- get_lead_info: Buscar dados cadastrais do cliente",
        "send_quote": "- send_quote: Criar um NOVO orçamento (apenas para primeiro orçamento, nunca para reduzir preço)",
        "request_price_change": "- request_price_change: Solicitar revisão de preço de orçamento EXISTENTE quando o cliente achar caro, pedir desconto ou não ter dinheiro. NÃO crie novo orçamento para isso — use ESTA ferramenta",
        "check_stay_availability": "- check_stay_availability: Consultar unidades de hospedagem livres no período (SEMPRE antes de cotar ou reservar)",
        "quote_stay": "- quote_stay: Calcular total da estadia + sinal de uma unidade específica",
        "create_stay_reservation": "- create_stay_reservation: Criar reserva e enviar Pix do sinal (SOMENTE após o hóspede aprovar a cotação)",
        "manage_stay_reservation": "- manage_stay_reservation: Consultar, cancelar ou remarcar reservas do hóspede",
        "quote_ride": "- quote_ride: Cotar valor de corrida entre origem e destino (SEMPRE antes de solicitar)",
        "request_ride": "- request_ride: Solicitar a corrida e acionar motoristas via WhatsApp (SOMENTE após o passageiro aprovar o valor)",
        "check_ride_status": "- check_ride_status: Consultar andamento da corrida atual do passageiro",
        "cancel_ride": "- cancel_ride: Cancelar a corrida ativa (confirme antes)",
    }
    return "\n".join(descs.get(t, f"- {t}") for t in enabled_tools)


def _get_tone(level: int) -> str:
    if level < 30:
        return "Formal e respeitoso. Use linguagem profissional."
    if level > 70:
        return "Animado, amigável e casual. Use linguagem descontraída."
    return "Profissional, mas simpático e acolhedor. Equilibrado."

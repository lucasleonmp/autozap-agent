"""
============================================
Agent Tools - Autonomous Tool Usage
============================================
Ferramentas que o agente pode usar autonomamente.

O agente decide QUANDO usar cada ferramenta baseado no contexto
da conversa, sem precisar de keyword matching.
"""

from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from supabase import Client
import json
import os
import httpx

from .stay_tools import create_stay_tools, workspace_has_stay, STAY_TOOL_NAMES
from .rider_tools import create_rider_tools, workspace_has_rider, RIDER_TOOL_NAMES


def create_tools(supabase: Client, workspace_id: str, lead_id: str, instance_id: str | None = None, enabled_tools: list[str] | None = None):
    """Cria ferramentas contextualizadas para o agente.

    Args:
        enabled_tools: Lista de nomes de ferramentas habilitadas. Se None, retorna todas.
        instance_id: ID da instância WhatsApp (para associar agendamentos).
    """

    # ─────────────────────────────────────────────
    # CORE TOOLS (always active)
    # ─────────────────────────────────────────────

    @tool
    def get_lead_info() -> str:
        """Busca informações cadastrais do lead/cliente atual.
        Use quando precisar do nome, telefone, tags ou classificação do cliente.
        """
        try:
            result = (
                supabase.table("leads")
                .select("name, phone, email, status, score, ai_enabled, metadata, created_at, instagram, linkedin, facebook")
                .eq("id", lead_id)
                .single()
                .execute()
            )

            if not result.data:
                return "Informações do lead não encontradas."

            lead = result.data
            created = datetime.fromisoformat(lead["created_at"].replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - created).days
            meta = lead.get("metadata") or {}

            lines = [
                f"Nome: {lead.get('name', 'Desconhecido')}",
                f"Telefone: {lead.get('phone', 'N/A')}",
                f"Email: {lead.get('email', 'N/A')}",
                f"Status: {lead.get('status', 'N/A')}",
                f"Score: {lead.get('score', 0)}/100",
                f"Cliente há: {days_since} dias",
            ]

            if lead.get("instagram"):
                lines.append(f"Instagram: {lead['instagram']}")
            if lead.get("linkedin"):
                lines.append(f"LinkedIn: {lead['linkedin']}")
            if lead.get("facebook"):
                lines.append(f"Facebook: {lead['facebook']}")
            if meta.get("company"):
                lines.append(f"Empresa: {meta['company']}")
            if meta.get("notes"):
                lines.append(f"Observações: {meta['notes']}")

            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao buscar info do lead: {e}"

    @tool
    def create_update_lead(name: str = "", email: str = "", status: str = "", notes: str = "", company: str = "") -> str:
        """Captura e atualiza dados do cliente automaticamente.
        Use quando descobrir o nome, email, empresa ou qualquer informação relevante do cliente durante a conversa.
        NÃO precisa esperar o cliente pedir — capture proativamente.

        Args:
            name: Nome do cliente (se souber)
            email: Email do cliente (se fornecido)
            status: Status do lead (new, contacted, qualified, proposal, negotiation, won, lost)
            notes: Observações adicionais sobre o cliente
            company: Nome da empresa do cliente
        """
        try:
            update_data = {}
            if name:
                update_data["name"] = name
            if email:
                update_data["email"] = email
            if status and status in ["new", "contacted", "qualified", "proposal", "negotiation", "won", "lost"]:
                update_data["status"] = status

            # Handle metadata fields (notes, company)
            if notes or company:
                current = supabase.table("leads").select("metadata").eq("id", lead_id).single().execute()
                meta = (current.data or {}).get("metadata") or {}
                if notes:
                    existing_notes = meta.get("notes", "")
                    meta["notes"] = f"{existing_notes}\n{notes}".strip() if existing_notes else notes
                if company:
                    meta["company"] = company
                update_data["metadata"] = meta

            if not update_data:
                return "Nenhum dado para atualizar."

            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            supabase.table("leads").update(update_data).eq("id", lead_id).execute()

            updated_fields = [k for k in update_data if k not in ("updated_at", "metadata")]
            if notes:
                updated_fields.append("observações")
            if company:
                updated_fields.append("empresa")

            return f"✅ Lead atualizado: {', '.join(updated_fields)}"
        except Exception as e:
            return f"Erro ao atualizar lead: {e}"

    @tool
    def transfer_to_human(reason: str = "Solicitação do cliente") -> str:
        """Escala a conversa para atendimento humano.
        Use quando o cliente pedir para falar com um atendente, quando não souber responder,
        ou quando a situação exigir intervenção humana.

        Args:
            reason: Motivo da transferência
        """
        try:
            # Pause AI on chat_memory
            result = (
                supabase.table("chat_memory")
                .select("id")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .limit(1)
                .execute()
            )

            if result.data:
                supabase.table("chat_memory").update({
                    "ai_paused": True,
                    "pause_reason": f"Transferência: {reason}",
                    "paused_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", result.data[0]["id"]).execute()

            # Also disable AI on the lead
            supabase.table("leads").update({
                "ai_enabled": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", lead_id).execute()

            return f"✅ Conversa transferida para atendimento humano. Motivo: {reason}. A IA foi pausada."
        except Exception as e:
            return f"Erro ao transferir: {e}. INFORME AO CLIENTE QUE HOUVE UM ERRO NA TRANSFERÊNCIA."

    @tool
    def search_knowledge_base(query: str) -> str:
        """Consulta a base de conhecimento para respostas precisas.
        Use quando o cliente fizer perguntas sobre produtos, serviços, políticas, ou qualquer informação
        que possa estar documentada na base de conhecimento.

        Args:
            query: Pergunta ou termos de busca
        """
        try:
            # Search by keyword matching in title, content, and keywords
            search_term = f"%{query}%"
            result = (
                supabase.table("knowledge_base")
                .select("title, content, category, keywords, priority")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                .order("priority", desc=True)
                .limit(5)
                .execute()
            )

            if not result.data:
                return f"Nenhuma informação encontrada na base de conhecimento para: '{query}'."

            items = []
            for kb in result.data:
                items.append(
                    f"📖 {kb['title']} [{kb.get('category', 'geral')}]\n{kb['content'][:500]}"
                )

            return f"Base de conhecimento ({len(result.data)} resultado(s)):\n\n" + "\n\n---\n\n".join(items)
        except Exception as e:
            return f"Erro ao consultar base de conhecimento: {e}"

    @tool
    def register_note(note: str, category: str = "observação") -> str:
        """Adiciona observações e notas ao perfil do cliente.
        Use para registrar informações importantes mencionadas durante a conversa,
        como preferências, reclamações, feedback, etc.

        Args:
            note: Conteúdo da nota
            category: Categoria da nota (observação, preferência, reclamação, feedback)
        """
        try:
            # Store in lead metadata
            current = supabase.table("leads").select("metadata").eq("id", lead_id).single().execute()
            meta = (current.data or {}).get("metadata") or {}

            agent_notes = meta.get("agent_notes", [])
            agent_notes.append({
                "note": note,
                "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            meta["agent_notes"] = agent_notes

            supabase.table("leads").update({
                "metadata": meta,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", lead_id).execute()

            return f"✅ Nota registrada: [{category}] {note}"
        except Exception as e:
            return f"Erro ao registrar nota: {e}"

    # ─────────────────────────────────────────────
    # OPTIONAL TOOLS (toggleable per agent)
    # ─────────────────────────────────────────────

    # ── Professionals helpers (multi-professional agenda) ──
    WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    WEEKDAY_LABELS = {
        "mon": "seg", "tue": "ter", "wed": "qua", "thu": "qui",
        "fri": "sex", "sat": "sáb", "sun": "dom",
    }

    def _get_professionals() -> list[dict]:
        """Bookable professionals of the workspace ([] = single shared agenda)."""
        try:
            result = supabase.rpc(
                "get_bookable_professionals", {"p_workspace_id": workspace_id}
            ).execute()
            return result.data or []
        except Exception:
            return []

    def _format_professionals(professionals: list[dict]) -> str:
        lines = []
        for p in professionals:
            wh = p.get("working_hours") or {}
            days = [WEEKDAY_LABELS[k] for k in WEEKDAY_KEYS if wh.get(k)]
            parts = [p.get("name", "Profissional")]
            if p.get("specialty"):
                parts.append(f"({p['specialty']})")
            if days:
                parts.append(f"— atende: {', '.join(days)}")
            lines.append(f"- {' '.join(parts)}")
        return "\n".join(lines)

    def _resolve_professional(name: str, professionals: list[dict]):
        """Match a professional by (partial) name or specialty.

        Returns (professional, error_message). Exactly one of them is set,
        except when there are no professionals at all (both None = legacy mode).
        """
        if not professionals:
            return None, None
        if not name:
            if len(professionals) == 1:
                return professionals[0], None
            return None, (
                "Há mais de um profissional disponível. Pergunte ao cliente com quem "
                "ele quer agendar. Profissionais:\n" + _format_professionals(professionals)
            )

        n = name.lower().strip()
        matches = [
            p for p in professionals
            if n in (p.get("name") or "").lower() or n in (p.get("specialty") or "").lower()
        ]
        if not matches:
            # Token-level match (e.g. "dra ana" vs "Ana Souza")
            for p in professionals:
                tokens = (p.get("name") or "").lower().split()
                query_tokens = [t for t in n.split() if len(t) > 2]
                if any(qt in tokens or any(t.startswith(qt) for t in tokens) for qt in query_tokens):
                    matches.append(p)
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, (
                f"Mais de um profissional corresponde a '{name}'. Peça ao cliente para "
                "especificar. Opções:\n" + _format_professionals(matches)
            )
        return None, (
            f"Não encontrei o profissional '{name}'. Profissionais disponíveis:\n"
            + _format_professionals(professionals)
        )

    def _working_windows(professional: dict | None, date: str):
        """UTC (start, end) windows the professional attends on this date.

        Returns None when there is no configured schedule (fall back to the
        default 08-18 window); [] when the professional does not attend that day.
        """
        if not professional:
            return None
        wh = professional.get("working_hours") or {}
        if not wh:
            return None
        key = WEEKDAY_KEYS[datetime.fromisoformat(date).weekday()]
        windows = []
        for w in wh.get(key) or []:
            try:
                s = datetime.fromisoformat(f"{date}T{w['start']}:00-03:00").astimezone(timezone.utc)
                e = datetime.fromisoformat(f"{date}T{w['end']}:00-03:00").astimezone(timezone.utc)
                if e > s:
                    windows.append((s, e))
            except (KeyError, ValueError):
                continue
        return windows

    def _appointment_conflicts(start_utc, end_utc, professional: dict | None) -> bool:
        query = (
            supabase.table("appointments")
            .select("id")
            .eq("workspace_id", workspace_id)
            .neq("status", "cancelled")
            .lt("start_time", end_utc.isoformat())
            .gt("end_time", start_utc.isoformat())
            .limit(1)
        )
        if professional:
            query = query.eq("professional_member_id", professional["member_id"])
        return bool(query.execute().data)

    @tool
    def list_professionals() -> str:
        """Lista os profissionais disponíveis para agendamento (nome, especialidade, dias de atendimento).
        Use quando o cliente quiser agendar e você precisar oferecer as opções de profissionais,
        ou quando ele perguntar quem atende na clínica/empresa.
        """
        try:
            professionals = _get_professionals()
            if not professionals:
                return "Não há profissionais cadastrados — a agenda é única (geral)."
            return (
                f"Profissionais disponíveis ({len(professionals)}):\n"
                + _format_professionals(professionals)
            )
        except Exception as e:
            return f"Erro ao listar profissionais: {e}"

    @tool
    def check_appointments(query: str = "") -> str:
        """Consulta os agendamentos existentes do cliente.
        Use quando o cliente perguntar sobre horários, consultas marcadas,
        ou quando precisar verificar disponibilidade.

        Args:
            query: Descrição do que procurar (ex: "próximos agendamentos")
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = (
                supabase.table("appointments")
                .select("title, start_time, end_time, status")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .neq("status", "cancelled")
                .gte("start_time", now)
                .order("start_time", desc=False)
                .limit(5)
                .execute()
            )

            if not result.data:
                return "O cliente não tem agendamentos futuros."

            appointments = []
            for a in result.data:
                start = datetime.fromisoformat(a["start_time"].replace("Z", "+00:00"))
                local_start = start - timedelta(hours=3)  # UTC-3
                appointments.append(
                    f"- {a['title']}: {local_start.strftime('%d/%m às %H:%M')} ({a['status']})"
                )

            return f"Agendamentos do cliente:\n" + "\n".join(appointments)

        except Exception as e:
            return f"Erro ao consultar agendamentos: {e}"

    @tool
    def check_availability(date: str, time: str = "", professional: str = "") -> str:
        """Verifica se um horário está disponível para agendamento.
        Use quando o cliente quiser marcar algo ou perguntar se tem vaga.
        Se houver profissionais cadastrados, informe com QUEM o cliente quer agendar
        (use list_professionals para ver as opções).

        Args:
            date: Data no formato YYYY-MM-DD
            time: Horário no formato HH:MM (opcional)
            professional: Nome do profissional desejado (obrigatório quando há mais de um)
        """
        import re
        try:
            professionals = _get_professionals()
            prof, err = _resolve_professional(professional, professionals)
            if err:
                return err

            duration = int(prof["appointment_duration_minutes"]) if prof else 60
            windows = _working_windows(prof, date)
            prof_label = f" com {prof['name']}" if prof else ""

            if windows is not None and not windows:
                days = [
                    WEEKDAY_LABELS[k]
                    for k in WEEKDAY_KEYS
                    if (prof.get("working_hours") or {}).get(k)
                ]
                days_txt = ", ".join(days) if days else "nenhum dia configurado"
                return (
                    f"{prof['name']} não atende em {date}. Dias de atendimento: {days_txt}."
                )

            if time:
                # Robust time cleaning (e.g., '10h' -> '10:00', '10:00:00' -> '10:00')
                clean_time = re.sub(r'[^0-9:]', '', time)
                if len(clean_time) == 1 or len(clean_time) == 2:
                    clean_time = f"{clean_time.zfill(2)}:00"
                elif len(clean_time) >= 5 and clean_time.count(':') >= 1:
                    clean_time = clean_time[:5]
                elif len(clean_time) == 4 and ':' not in clean_time:
                    clean_time = f"{clean_time[:2]}:{clean_time[2:]}"

                try:
                    start_dt = datetime.fromisoformat(f"{date}T{clean_time}:00-03:00")
                except ValueError:
                    return f"[ERRO] Formato de data/hora inválido. Recebido date='{date}' e time='{time}'. Corrija e tente novamente."

                start_utc = start_dt.astimezone(timezone.utc)
                end_utc = start_utc + timedelta(minutes=duration)

                if windows is not None and not any(
                    s <= start_utc and end_utc <= e for s, e in windows
                ):
                    hours_txt = ", ".join(
                        f"{(s + timedelta(hours=-3)).strftime('%H:%M')}–{(e + timedelta(hours=-3)).strftime('%H:%M')}"
                        for s, e in windows
                    )
                    return (
                        f"{date} às {clean_time} está FORA do horário de atendimento"
                        f"{prof_label}. Horários nesse dia: {hours_txt}."
                    )

                if _appointment_conflicts(start_utc, end_utc, prof):
                    return f"O horário {date} às {clean_time}{prof_label} já está OCUPADO."
                return f"O horário {date} às {clean_time}{prof_label} está DISPONÍVEL! ✅"

            # No specific time: list free slots (or busy times in legacy mode)
            if windows is None:
                windows = [(
                    datetime.fromisoformat(f"{date}T08:00:00-03:00").astimezone(timezone.utc),
                    datetime.fromisoformat(f"{date}T18:00:00-03:00").astimezone(timezone.utc),
                )]

            day_start = min(w[0] for w in windows)
            day_end = max(w[1] for w in windows)

            busy_query = (
                supabase.table("appointments")
                .select("start_time, end_time")
                .eq("workspace_id", workspace_id)
                .neq("status", "cancelled")
                .lt("start_time", day_end.isoformat())
                .gt("end_time", day_start.isoformat())
                .order("start_time", desc=False)
            )
            if prof:
                busy_query = busy_query.eq("professional_member_id", prof["member_id"])
            busy = [
                (
                    datetime.fromisoformat(a["start_time"].replace("Z", "+00:00")),
                    datetime.fromisoformat(a["end_time"].replace("Z", "+00:00")),
                )
                for a in (busy_query.execute().data or [])
            ]

            free_slots = []
            for w_start, w_end in windows:
                cursor = w_start
                while cursor + timedelta(minutes=duration) <= w_end:
                    slot_end = cursor + timedelta(minutes=duration)
                    if not any(bs < slot_end and be > cursor for bs, be in busy):
                        free_slots.append((cursor + timedelta(hours=-3)).strftime("%H:%M"))
                    cursor = slot_end

            if not free_slots:
                return f"Não há horários livres em {date}{prof_label}."
            shown = free_slots[:12]
            more = f" (e mais {len(free_slots) - 12})" if len(free_slots) > 12 else ""
            return (
                f"Horários DISPONÍVEIS em {date}{prof_label} "
                f"(duração {duration}min): {', '.join(shown)}{more}"
            )

        except Exception as e:
            return f"[ERRO] Falha ao verificar disponibilidade: {e}. INFORME AO CLIENTE QUE OCORREU UM ERRO."

    @tool
    def schedule_appointment(date: str, time: str, purpose: str = "Agendamento via WhatsApp", professional: str = "") -> str:
        """Cria um novo agendamento para o cliente.
        Use SOMENTE após confirmar com o cliente que ele deseja marcar.
        SEMPRE verifique disponibilidade antes de agendar.
        Se houver profissionais cadastrados, o agendamento DEVE ser com um profissional
        específico — pergunte ao cliente com quem ele quer marcar.

        Args:
            date: Data no formato YYYY-MM-DD
            time: Horário no formato HH:MM
            purpose: Descrição do motivo do agendamento
            professional: Nome do profissional com quem agendar (obrigatório quando há mais de um)
        """
        import re
        try:
            professionals = _get_professionals()
            prof, err = _resolve_professional(professional, professionals)
            if err:
                return f"{err}\nO agendamento NÃO foi criado."

            duration = int(prof["appointment_duration_minutes"]) if prof else 60

            # Robust time cleaning
            clean_time = re.sub(r'[^0-9:]', '', time)
            if len(clean_time) == 1 or len(clean_time) == 2:
                clean_time = f"{clean_time.zfill(2)}:00"
            elif len(clean_time) >= 5 and clean_time.count(':') >= 1:
                clean_time = clean_time[:5]
            elif len(clean_time) == 4 and ':' not in clean_time:
                clean_time = f"{clean_time[:2]}:{clean_time[2:]}"

            try:
                start_dt = datetime.fromisoformat(f"{date}T{clean_time}:00-03:00")
            except ValueError:
                return f"[ERRO CRÍTICO] O formato de hora '{time}' ou data '{date}' é inválido. O agendamento NÃO foi criado."

            start_utc = start_dt.astimezone(timezone.utc)
            end_utc = start_utc + timedelta(minutes=duration)

            windows = _working_windows(prof, date)
            if windows is not None:
                if not windows:
                    return (
                        f"{prof['name']} não atende em {date}. O agendamento NÃO foi criado. "
                        f"Use check_availability para achar outro dia."
                    )
                if not any(s <= start_utc and end_utc <= e for s, e in windows):
                    hours_txt = ", ".join(
                        f"{(s + timedelta(hours=-3)).strftime('%H:%M')}–{(e + timedelta(hours=-3)).strftime('%H:%M')}"
                        for s, e in windows
                    )
                    return (
                        f"{date} às {clean_time} está fora do horário de {prof['name']} "
                        f"({hours_txt}). O agendamento NÃO foi criado."
                    )

            if _appointment_conflicts(start_utc, end_utc, prof):
                who = f" com {prof['name']}" if prof else ""
                return (
                    f"O horário {date} às {clean_time}{who} já está OCUPADO. "
                    f"O agendamento NÃO foi criado. Ofereça outro horário."
                )

            metadata = {"source": "whatsapp_agent"}
            title = purpose
            if prof:
                metadata["professional_name"] = prof["name"]
                title = f"{purpose} — {prof['name']}"

            insert_data = {
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "title": title,
                "start_time": start_utc.isoformat(),
                "end_time": end_utc.isoformat(),
                "status": "scheduled",
                "metadata": json.dumps(metadata),
            }
            if instance_id:
                insert_data["instance_id"] = instance_id
            if prof:
                insert_data["professional_member_id"] = prof["member_id"]

            supabase.table("appointments").insert(insert_data).execute()

            who = f" com {prof['name']}" if prof else ""
            return f"✅ Agendamento criado com sucesso: {purpose}{who} em {date} às {clean_time}."

        except Exception as e:
            return f"[ERRO CRÍTICO] O agendamento FALHOU. Motivo: {e}. NÃO DIGA QUE FOI CONFIRMADO."

    @tool
    def cancel_reschedule(action: str, appointment_title: str = "", new_date: str = "", new_time: str = "") -> str:
        """Cancela ou reagenda um agendamento existente do cliente.
        SEMPRE confirme com o cliente antes de cancelar ou mover.

        Args:
            action: 'cancel' para cancelar, 'reschedule' para reagendar
            appointment_title: Título ou descrição do agendamento (para identificar)
            new_date: Nova data (YYYY-MM-DD) - obrigatório para reagendamento
            new_time: Novo horário (HH:MM) - obrigatório para reagendamento
        """
        import re
        try:
            # Find the appointment
            now = datetime.now(timezone.utc).isoformat()
            query = (
                supabase.table("appointments")
                .select("id, title, start_time, end_time")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .neq("status", "cancelled")
                .gte("start_time", now)
                .order("start_time", desc=False)
                .limit(5)
                .execute()
            )

            if not query.data:
                return "O cliente não tem agendamentos futuros para alterar."

            # Find best match
            target = query.data[0]  # Default to next appointment
            if appointment_title:
                for a in query.data:
                    if appointment_title.lower() in a["title"].lower():
                        target = a
                        break

            if action == "cancel":
                supabase.table("appointments").update({
                    "status": "cancelled",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", target["id"]).execute()

                start = datetime.fromisoformat(target["start_time"].replace("Z", "+00:00")) - timedelta(hours=3)
                return f"✅ Agendamento cancelado: {target['title']} ({start.strftime('%d/%m às %H:%M')})"

            elif action == "reschedule":
                if not new_date or not new_time:
                    return "Para reagendar, informe a nova data (new_date) e horário (new_time)."

                clean_time = re.sub(r'[^0-9:]', '', new_time)
                if len(clean_time) <= 2:
                    clean_time = f"{clean_time.zfill(2)}:00"
                elif len(clean_time) >= 5:
                    clean_time = clean_time[:5]

                try:
                    new_start = datetime.fromisoformat(f"{new_date}T{clean_time}:00-03:00").astimezone(timezone.utc)
                except ValueError:
                    return f"[ERRO] Data/hora inválida: {new_date} {new_time}"

                new_end = new_start + timedelta(hours=1)

                supabase.table("appointments").update({
                    "start_time": new_start.isoformat(),
                    "end_time": new_end.isoformat(),
                    "status": "rescheduled",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", target["id"]).execute()

                return f"✅ Agendamento reagendado: {target['title']} → {new_date} às {clean_time}"
            else:
                return "Ação inválida. Use 'cancel' ou 'reschedule'."

        except Exception as e:
            return f"Erro ao alterar agendamento: {e}"

    def _recalc_deal_value() -> None:
        """Recalcula e persiste o deal_value do lead a partir dos quotes ativos."""
        try:
            quotes_res = (
                supabase.table("quotes")
                .select("estimated_value")
                .eq("lead_id", lead_id)
                .in_("status", ["pending", "negotiating", "accepted"])
                .execute()
            )
            total = sum(q.get("estimated_value", 0) or 0 for q in (quotes_res.data or []))
            supabase.table("leads").update({
                "deal_value": total,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", lead_id).execute()
        except Exception as e:
            print(f"[tools] Error recalculating deal_value: {e}")

    def _get_lead_phone() -> str:
        """Busca o telefone do lead para montar o chat_id."""
        try:
            res = supabase.table("leads").select("phone").eq("id", lead_id).single().execute()
            return (res.data or {}).get("phone", "")
        except Exception:
            return ""

    @tool
    def send_quote(description: str, amount: float, due_days: int = 7) -> str:
        """Gera e salva um orçamento com base nos serviços discutidos.
        Use quando o cliente pedir um orçamento ou quando os serviços e valores forem acordados na conversa.
        O orçamento é criado automaticamente e fica visível na página de Orçamentos para o admin.

        Args:
            description: Descrição dos serviços/produtos incluídos no orçamento
            amount: Valor total em reais (R$)
            due_days: Dias de validade do orçamento (padrão: 7)
        """
        try:
            due_date = (datetime.now(timezone.utc) + timedelta(days=due_days)).isoformat()
            phone = _get_lead_phone()
            chat_id = f"{phone.replace('+', '')}@c.us" if phone else ""

            result = supabase.table("quotes").insert({
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "chat_id": chat_id,
                "status": "pending",
                "estimated_value": amount,
                "ai_summary": description,
                "source": "agent",
                "valid_until": due_date,
            }).execute()

            # Update lead score +10
            try:
                lead_res = supabase.table("leads").select("score").eq("id", lead_id).single().execute()
                current_score = (lead_res.data or {}).get("score", 0) or 0
                supabase.table("leads").update({
                    "score": max(0, current_score + 10),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", lead_id).execute()
            except Exception:
                pass

            # Recalculate deal_value
            _recalc_deal_value()

            formatted = f"R$ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            return (
                f"✅ Orçamento criado com sucesso!\n"
                f"Descrição: {description}\n"
                f"Valor: {formatted}\n"
                f"O orçamento foi salvo e pode ser visualizado pelo administrador."
            )
        except Exception as e:
            return f"Erro ao criar orçamento: {e}"

    @tool
    def check_lead_quotes() -> str:
        """Busca os orçamentos existentes do cliente atual.
        Use para verificar se já existe orçamento antes de criar um novo,
        ou quando o cliente perguntar sobre orçamentos anteriores.
        """
        try:
            result = (
                supabase.table("quotes")
                .select("id, status, estimated_value, ai_summary, created_at, valid_until")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )

            if not result.data:
                return "O cliente não tem orçamentos."

            status_labels = {
                "pending": "⏳ Pendente",
                "negotiating": "🔄 Em Negociação",
                "accepted": "✅ Aceito",
                "rejected": "❌ Rejeitado",
                "completed": "🏁 Concluído",
            }

            lines = []
            for q in result.data:
                val = q.get("estimated_value") or 0
                formatted_val = f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                status = status_labels.get(q["status"], q["status"])
                desc = q.get("ai_summary", "Sem descrição") or "Sem descrição"
                dt = q["created_at"][:10] if q.get("created_at") else "N/A"
                lines.append(f"- {desc[:80]} | {formatted_val} | {status} | {dt}")

            return f"Orçamentos do cliente ({len(result.data)}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Erro ao buscar orçamentos: {e}"

    @tool
    def manage_quote_status(new_status: str, reason: str = "") -> str:
        """Atualiza o status do orçamento mais recente do cliente.
        Use quando o cliente indicar na conversa que aceita, rejeita ou quer negociar o orçamento.
        IMPORTANTE: Quando o cliente ACEITAR ("fechado", "aceito", "vamos em frente"),
        marque como 'negotiating' para o admin confirmar a aprovação final.

        Args:
            new_status: Novo status — use 'negotiating' quando o cliente aceitar, 'rejected' se recusar
            reason: Motivo da mudança (ex: "Cliente aceitou o valor proposto")
        """
        try:
            valid_statuses = ["negotiating", "rejected"]
            if new_status not in valid_statuses:
                return f"Status inválido. Use um destes: {', '.join(valid_statuses)}. Para 'accepted' e 'completed', o admin deve confirmar manualmente."

            # Get most recent active quote
            result = (
                supabase.table("quotes")
                .select("id, status, estimated_value, ai_summary")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .in_("status", ["pending", "negotiating"])
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return "Nenhum orçamento ativo encontrado para este cliente."

            quote = result.data[0]
            update_data = {
                "status": new_status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            supabase.table("quotes").update(update_data).eq("id", quote["id"]).execute()

            # Update lead score
            score_change = -5 if new_status == "rejected" else 0
            if score_change != 0:
                try:
                    lead_res = supabase.table("leads").select("score").eq("id", lead_id).single().execute()
                    current_score = (lead_res.data or {}).get("score", 0) or 0
                    supabase.table("leads").update({
                        "score": max(0, current_score + score_change),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }).eq("id", lead_id).execute()
                except Exception:
                    pass

            # Recalculate deal_value
            _recalc_deal_value()

            status_labels = {
                "negotiating": "Em Negociação (aguardando confirmação do admin)",
                "rejected": "Rejeitado",
            }
            label = status_labels.get(new_status, new_status)
            val = quote.get("estimated_value", 0) or 0
            formatted_val = f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            return (
                f"✅ Orçamento atualizado para: {label}\n"
                f"Valor: {formatted_val}\n"
                f"Motivo: {reason or 'N/A'}\n"
                f"O administrador foi notificado."
            )
        except Exception as e:
            return f"Erro ao atualizar status do orçamento: {e}"

    @tool
    def request_price_change(customer_message: str, requested_value: float = 0) -> str:
        """Cria uma solicitação de revisão de preço quando o cliente pede desconto.
        Use SEMPRE que o cliente disser que está caro, pedir desconto, ou sugerir um valor menor.
        NUNCA crie um novo orçamento para reduzir preço. Use ESTA ferramenta.
        A solicitação fica pendente para o admin aprovar ou rejeitar.

        Args:
            customer_message: Mensagem/motivo do cliente para o pedido de desconto
            requested_value: Valor sugerido pelo cliente em reais (0 se não especificou)
        """
        try:
            # Get most recent active quote
            result = (
                supabase.table("quotes")
                .select("id, estimated_value, status")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .in_("status", ["pending", "negotiating"])
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                return "Nenhum orçamento ativo encontrado para solicitar revisão de preço."

            quote = result.data[0]
            original_value = quote.get("estimated_value", 0) or 0

            # Create price change request
            supabase.table("price_change_requests").insert({
                "workspace_id": workspace_id,
                "quote_id": quote["id"],
                "lead_id": lead_id,
                "original_value": original_value,
                "requested_value": requested_value if requested_value > 0 else None,
                "customer_message": customer_message,
                "status": "pending",
            }).execute()

            # Update quote to negotiating
            supabase.table("quotes").update({
                "status": "negotiating",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", quote["id"]).execute()

            formatted_orig = f"R$ {original_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

            msg = (
                f"✅ Solicitação de revisão de preço registrada!\n"
                f"Valor original: {formatted_orig}\n"
            )
            if requested_value > 0:
                formatted_req = f"R$ {requested_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                msg += f"Valor sugerido pelo cliente: {formatted_req}\n"
            msg += (
                f"Motivo: {customer_message}\n"
                f"O administrador será notificado e decidirá sobre a revisão.\n"
                f"IMPORTANTE: Informe ao cliente que você vai SOLICITAR A REVISÃO DE PREÇO "
                f"com a equipe e peça para ele AGUARDAR. NÃO prometa desconto. "
                f"NÃO crie um novo orçamento."
            )
            return msg
        except Exception as e:
            return f"Erro ao solicitar revisão de preço: {e}"

    @tool
    def query_products(search: str) -> str:
        """Busca no catálogo de produtos (preço, disponibilidade, descrição).
        Use quando o cliente perguntar sobre produtos, preços ou disponibilidade.

        Args:
            search: Nome ou descrição do produto a buscar
        """
        try:
            # Search in knowledge_base with category 'products' or 'pricing'
            search_term = f"%{search}%"
            result = (
                supabase.table("knowledge_base")
                .select("title, content, keywords")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .in_("category", ["products", "pricing", "services"])
                .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                .limit(5)
                .execute()
            )

            if not result.data:
                # Fallback: search all knowledge base
                result = (
                    supabase.table("knowledge_base")
                    .select("title, content")
                    .eq("workspace_id", workspace_id)
                    .eq("is_active", True)
                    .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                    .limit(3)
                    .execute()
                )

            if not result.data:
                return f"Nenhum produto encontrado para '{search}'. Consulte a base de conhecimento ou o administrador."

            items = []
            for p in result.data:
                items.append(f"📦 {p['title']}\n{p['content'][:300]}")

            return f"Produtos encontrados ({len(result.data)}):\n\n" + "\n\n---\n\n".join(items)
        except Exception as e:
            return f"Erro ao consultar produtos: {e}"

    @tool
    def check_conversation_history(query: str = "") -> str:
        """Busca conversas anteriores do cliente para contexto.
        Use quando precisar de contexto sobre interações passadas,
        ou quando o cliente mencionar algo discutido anteriormente.

        Args:
            query: Termo opcional para filtrar nas mensagens
        """
        try:
            # Get recent messages for this lead
            builder = (
                supabase.table("messages")
                .select("content, direction, message_type, created_at")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .order("created_at", desc=True)
                .limit(30)
            )

            result = builder.execute()

            if not result.data:
                return "Sem histórico de conversas para este cliente."

            messages = list(reversed(result.data))

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                messages = [m for m in messages if m.get("content") and query_lower in m["content"].lower()]
                if not messages:
                    return f"Nenhuma mensagem encontrada com o termo '{query}'."

            history = []
            for m in messages[-20:]:  # Last 20 relevant messages
                direction = "👤 Cliente" if m["direction"] == "inbound" else "🤖 Agente"
                content = (m.get("content") or "[mídia]")[:200]
                dt = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
                history.append(f"[{dt.strftime('%d/%m %H:%M')}] {direction}: {content}")

            return f"Histórico ({len(history)} mensagens):\n" + "\n".join(history)
        except Exception as e:
            return f"Erro ao buscar histórico: {e}"

    @tool
    def check_order_status(order_reference: str = "") -> str:
        """Consulta status de pedidos/cobranças para e-commerce/delivery.
        Use quando o cliente perguntar sobre status de pedido, pagamento, entrega ou envio.

        Args:
            order_reference: Número ou referência do pedido (opcional)
        """
        try:
            builder = (
                supabase.table("invoices")
                .select(
                    "id, description, amount, status, due_date, created_at, paid_at, "
                    "payment_url, pix_code, provider, payment_method"
                )
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .order("created_at", desc=True)
                .limit(5)
            )

            result = builder.execute()

            if not result.data:
                return "Nenhum pedido/cobrança encontrado para este cliente."

            orders = []
            status_map = {
                "pending": "⏳ Pendente de pagamento",
                "sent": "📤 Cobrança enviada — aguardando pagamento",
                "paid": "✅ Pago / confirmado",
                "overdue": "⚠️ Vencido",
                "canceled": "❌ Cancelado",
                "refunded": "↩️ Estornado",
            }

            for o in result.data:
                if order_reference and order_reference.lower() not in (
                    (o.get("description") or "").lower() + o.get("id", "")
                ):
                    continue
                dt = datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
                status_label = status_map.get(o["status"], o["status"])
                line = (
                    f"- {o['description'] or 'Sem descrição'} | R$ {o['amount']:.2f} | "
                    f"{status_label} | {dt.strftime('%d/%m/%Y')}"
                )
                if o.get("payment_url") and o["status"] in ("pending", "sent", "overdue"):
                    line += f"\n  Link: {o['payment_url']}"
                if o.get("paid_at") and o["status"] == "paid":
                    line += "\n  Informe ao cliente que o pagamento já foi confirmado."
                orders.append(line)

            if not orders:
                return "Nenhuma cobrança correspondente à referência informada."

            return "Pedidos/Cobranças do cliente:\n" + "\n".join(orders)
        except Exception as e:
            return f"Erro ao consultar pedidos: {e}"

    @tool
    def send_payment_link(amount: float, description: str = "Pagamento") -> str:
        """Gera cobrança real com link/PIX (gateway do workspace ou PIX estático).
        Use quando o cliente quiser pagar ou quando um orçamento for aprovado.
        Envie o link ou código PIX retornado na resposta ao cliente.

        Args:
            amount: Valor em reais
            description: Descrição do pagamento
        """
        try:
            supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
            service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            due_date = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")

            if supabase_url and service_key:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(
                        f"{supabase_url}/functions/v1/tenant-payments",
                        headers={
                            "Authorization": f"Bearer {service_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "action": "create_charge",
                            "workspace_id": workspace_id,
                            "lead_id": lead_id,
                            "amount": float(amount),
                            "description": description,
                            "due_date": due_date,
                            "payment_method": "pix",
                            "source": "agent",
                            "send_now": True,
                        },
                    )
                    data = resp.json() if resp.content else {}
                    if resp.is_success and data.get("invoice"):
                        inv = data["invoice"]
                        parts = [
                            "✅ Cobrança criada e enviada!",
                            f"Valor: R$ {float(amount):.2f}",
                            f"Provedor: {data.get('provider') or inv.get('provider') or 'pix'}",
                        ]
                        if inv.get("payment_url"):
                            parts.append(f"Link de pagamento: {inv['payment_url']}")
                            parts.append("Envie este link ao cliente para ele pagar.")
                        if inv.get("pix_code"):
                            parts.append(f"PIX Copia e Cola: {inv['pix_code']}")
                            parts.append("Envie o código PIX ao cliente.")
                        if not inv.get("payment_url") and not inv.get("pix_code"):
                            parts.append("Cobrança registrada; confirme o envio no WhatsApp.")
                        return "\n".join(parts)
                    err = data.get("error") or resp.text
                    # fall through to static PIX below
                    print(f"[send_payment_link] tenant-payments failed: {err}")

            # Fallback: static PIX key (legacy)
            pix_config = (
                supabase.table("pix_config")
                .select("pix_key, pix_key_type, receiver_name, receiver_city, is_active")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .limit(1)
                .execute()
            )

            if pix_config.data:
                config = pix_config.data[0]
                supabase.table("invoices").insert({
                    "workspace_id": workspace_id,
                    "lead_id": lead_id,
                    "amount": amount,
                    "description": description,
                    "due_date": due_date,
                    "status": "sent",
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "source": "agent",
                    "provider": "static_pix",
                    "payment_method": "pix",
                }).execute()

                return (
                    f"✅ Cobrança criada!\n"
                    f"Valor: R$ {amount:.2f}\n"
                    f"Chave PIX ({config['pix_key_type']}): {config['pix_key']}\n"
                    f"Beneficiário: {config['receiver_name']}\n"
                    f"Informe a chave PIX ao cliente para pagamento."
                )

            return (
                "⚠️ Nenhum gateway de pagamento nem PIX configurado neste workspace.\n"
                "Peça ao administrador para conectar um meio de pagamento em Configurações → Cobranças."
            )
        except Exception as e:
            return f"Erro ao gerar link de pagamento: {e}"

    @tool
    def calculate_shipping(zip_code: str = "", product_description: str = "") -> str:
        """Calcula frete e prazo de entrega.
        Use quando o cliente perguntar sobre frete, envio ou prazo de entrega.

        Args:
            zip_code: CEP de destino
            product_description: Descrição do produto para cálculo
        """
        try:
            # Search in knowledge base for shipping info
            result = (
                supabase.table("knowledge_base")
                .select("title, content")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .in_("category", ["shipping", "delivery", "logistics", "services"])
                .limit(3)
                .execute()
            )

            if result.data:
                info = []
                for kb in result.data:
                    info.append(f"📦 {kb['title']}: {kb['content'][:300]}")
                return (
                    f"Informações de frete/entrega encontradas:\n\n"
                    + "\n\n".join(info)
                    + f"\n\nCEP informado: {zip_code or 'não informado'}"
                )

            return (
                f"ℹ️ Informações de frete não configuradas na base de conhecimento.\n"
                f"CEP: {zip_code or 'não informado'}\n"
                f"Produto: {product_description or 'não especificado'}\n"
                f"Consulte o administrador para detalhes de frete e envio."
            )
        except Exception as e:
            return f"Erro ao calcular frete: {e}"

    @tool
    def summarize_conversation() -> str:
        """Gera um resumo da conversa para handoff humano.
        Use antes de transferir para um atendente humano, ou quando solicitado.
        O resumo ajuda o atendente a entender rapidamente o contexto.
        """
        try:
            # Get recent messages
            result = (
                supabase.table("messages")
                .select("content, direction, created_at")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .eq("message_type", "text")
                .order("created_at", desc=True)
                .limit(30)
                .execute()
            )

            if not result.data:
                return "Sem histórico para resumir."

            messages = list(reversed(result.data))

            # Get lead info for context
            lead_result = (
                supabase.table("leads")
                .select("name, phone, status, score")
                .eq("id", lead_id)
                .single()
                .execute()
            )

            lead_info = lead_result.data or {}

            # Build summary context
            total_msgs = len(messages)
            client_msgs = [m for m in messages if m["direction"] == "inbound"]
            agent_msgs = [m for m in messages if m["direction"] != "inbound"]

            first_msg = datetime.fromisoformat(messages[0]["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
            last_msg = datetime.fromisoformat(messages[-1]["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)

            # Extract key messages (first and last from each side)
            key_topics = []
            for m in client_msgs[-5:]:
                if m.get("content"):
                    key_topics.append(m["content"][:100])

            summary = (
                f"📋 RESUMO DA CONVERSA\n"
                f"Cliente: {lead_info.get('name', 'Desconhecido')} ({lead_info.get('phone', 'N/A')})\n"
                f"Status: {lead_info.get('status', 'N/A')} | Score: {lead_info.get('score', 0)}\n"
                f"Período: {first_msg.strftime('%d/%m %H:%M')} → {last_msg.strftime('%d/%m %H:%M')}\n"
                f"Mensagens: {total_msgs} total ({len(client_msgs)} do cliente, {len(agent_msgs)} do agente)\n"
                f"\nÚltimas mensagens do cliente:\n"
                + "\n".join(f"  • {t}" for t in key_topics)
            )

            return summary
        except Exception as e:
            return f"Erro ao gerar resumo: {e}"

    @tool
    def create_followup_task(description: str, days_from_now: int = 1, time: str = "09:00") -> str:
        """Agenda um lembrete para recontatar o cliente.
        Use quando for necessário fazer follow-up futuro,
        ou quando o cliente pedir para ser contatado em outro momento.

        Args:
            description: O que precisa ser feito no follow-up
            days_from_now: Daqui a quantos dias (padrão: 1 = amanhã)
            time: Horário do lembrete (padrão: 09:00)
        """
        import re
        try:
            clean_time = re.sub(r'[^0-9:]', '', time)
            if len(clean_time) <= 2:
                clean_time = f"{clean_time.zfill(2)}:00"
            elif len(clean_time) >= 5:
                clean_time = clean_time[:5]

            target_date = (datetime.now(timezone.utc) - timedelta(hours=3) + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

            try:
                start_dt = datetime.fromisoformat(f"{target_date}T{clean_time}:00-03:00").astimezone(timezone.utc)
            except ValueError:
                return f"[ERRO] Horário inválido: {time}"

            end_dt = start_dt + timedelta(minutes=30)

            insert_data = {
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "title": f"📌 Follow-up: {description}",
                "description": f"Tarefa de follow-up criada pelo agente IA.\nMotivo: {description}",
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "status": "scheduled",
                "metadata": json.dumps({"type": "followup", "created_by": "agent"}),
            }
            if instance_id:
                insert_data["instance_id"] = instance_id

            supabase.table("appointments").insert(insert_data).execute()

            return (
                f"✅ Follow-up agendado!\n"
                f"Data: {target_date} às {clean_time}\n"
                f"Tarefa: {description}"
            )
        except Exception as e:
            return f"Erro ao criar follow-up: {e}"

    # ─────────────────────────────────────────────
    # TOOL REGISTRY
    # ─────────────────────────────────────────────

    all_tools = {
        # Core (always active)
        "get_lead_info": get_lead_info,
        "create_update_lead": create_update_lead,
        "transfer_to_human": transfer_to_human,
        "search_knowledge_base": search_knowledge_base,
        "register_note": register_note,
        # Optional (toggleable)
        "list_professionals": list_professionals,
        "check_appointments": check_appointments,
        "check_availability": check_availability,
        "schedule_appointment": schedule_appointment,
        "cancel_reschedule": cancel_reschedule,
        "send_quote": send_quote,
        "check_lead_quotes": check_lead_quotes,
        "manage_quote_status": manage_quote_status,
        "request_price_change": request_price_change,
        "query_products": query_products,
        "check_conversation_history": check_conversation_history,
        "check_order_status": check_order_status,
        "send_payment_link": send_payment_link,
        "calculate_shipping": calculate_shipping,
        "summarize_conversation": summarize_conversation,
        "create_followup_task": create_followup_task,
    }

    # Vertical Stay (hospedagem) — habilitadas via enabled_tools quando o
    # workspace tem o add-on Stay ativo
    all_tools.update(create_stay_tools(supabase, workspace_id, lead_id))

    # Vertical Rider (mobilidade) — mesma lógica de gating do Stay
    all_tools.update(create_rider_tools(supabase, workspace_id, lead_id))

    if enabled_tools:
        names = list(enabled_tools)
        # Scheduling requires knowing who the professionals are — always pair them.
        if "list_professionals" not in names and any(
            n in names for n in ("check_availability", "schedule_appointment", "check_appointments")
        ):
            names.append("list_professionals")
        # Stay vertical requires the add-on (or trial) — backend gate.
        if any(n in names for n in STAY_TOOL_NAMES) and not workspace_has_stay(supabase, workspace_id):
            names = [n for n in names if n not in STAY_TOOL_NAMES]
        # Rider vertical requires the add-on (or trial) — backend gate.
        if any(n in names for n in RIDER_TOOL_NAMES) and not workspace_has_rider(supabase, workspace_id):
            names = [n for n in names if n not in RIDER_TOOL_NAMES]
        return [all_tools[name] for name in names if name in all_tools]
    return list(all_tools.values())

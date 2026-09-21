"""
============================================
AutoZap Stay - Hospitality Tools
============================================
Ferramentas do vertical de hospedagem: disponibilidade,
cotação e reserva com sinal via Pix (reutiliza tenant-payments).

Habilitadas apenas quando o workspace tem o add-on Stay ativo
(gating no wizard/enabled_tools do agente).
"""

from datetime import datetime, timezone, timedelta, date
from langchain_core.tools import tool
from supabase import Client
import os
import httpx

# Percentual padrão do sinal para confirmar a reserva
DEPOSIT_PERCENT = 30
# Prazo para pagamento do sinal antes da unidade ser liberada
HOLD_HOURS = 24

ACTIVE_STATUSES = ("pending", "confirmed", "checked_in")

STAY_TOOL_NAMES = (
    "check_stay_availability",
    "quote_stay",
    "create_stay_reservation",
    "manage_stay_reservation",
)


def workspace_has_stay(supabase: Client, workspace_id: str) -> bool:
    """Entitlement do add-on Stay (trial libera para avaliação).

    Fail-open quando não há linha de assinatura para o workspace
    (ex.: sub-workspaces cujo billing vive no workspace principal).
    """
    try:
        result = (
            supabase.table("subscriptions")
            .select("addon_stay, plan_type")
            .eq("workspace_id", workspace_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return True
        sub = result.data[0]
        return bool(sub.get("addon_stay")) or sub.get("plan_type") == "trial"
    except Exception as e:
        print(f"[stay_tools] entitlement check failed (fail-open): {e}")
        return True


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip()[:10])
    except (ValueError, AttributeError):
        return None


def _nightly_rates(unit: dict, overrides: list[dict], check_in: date, check_out: date) -> list[float]:
    """Tarifa por noite: override sazonal quando existir, senão a diária base."""
    rates = []
    night = check_in
    while night < check_out:
        rate = float(unit.get("base_rate") or 0)
        for ov in overrides:
            start = _parse_date(ov["start_date"])
            end = _parse_date(ov["end_date"])
            if start and end and start <= night <= end:
                rate = float(ov["rate"])
                break
        rates.append(rate)
        night += timedelta(days=1)
    return rates


def create_stay_tools(supabase: Client, workspace_id: str, lead_id: str) -> dict:
    """Cria as ferramentas Stay contextualizadas. Retorna {nome: tool}."""

    def _active_units() -> list[dict]:
        result = (
            supabase.table("stay_units")
            .select("id, name, description, capacity, base_rate, min_nights, property_id, stay_properties(name, check_in_time, check_out_time, policies, active)")
            .eq("workspace_id", workspace_id)
            .eq("active", True)
            .execute()
        )
        return [
            u for u in (result.data or [])
            if (u.get("stay_properties") or {}).get("active", True)
        ]

    def _unit_is_free(unit_id: str, check_in: date, check_out: date, ignore_reservation_id: str | None = None) -> bool:
        """Livre = sem reserva ativa nem bloqueio sobrepondo o intervalo [check_in, check_out)."""
        res = (
            supabase.table("stay_reservations")
            .select("id, check_in, check_out, status")
            .eq("unit_id", unit_id)
            .in_("status", list(ACTIVE_STATUSES))
            .lt("check_in", check_out.isoformat())
            .gt("check_out", check_in.isoformat())
            .execute()
        )
        for r in res.data or []:
            if ignore_reservation_id and r["id"] == ignore_reservation_id:
                continue
            return False

        blocks = (
            supabase.table("stay_blocks")
            .select("id")
            .eq("unit_id", unit_id)
            .lte("start_date", (check_out - timedelta(days=1)).isoformat())
            .gte("end_date", check_in.isoformat())
            .limit(1)
            .execute()
        )
        return not (blocks.data or [])

    def _overrides_for_unit(unit_id: str, check_in: date, check_out: date) -> list[dict]:
        result = (
            supabase.table("stay_rate_overrides")
            .select("start_date, end_date, rate, min_nights")
            .eq("unit_id", unit_id)
            .lte("start_date", (check_out - timedelta(days=1)).isoformat())
            .gte("end_date", check_in.isoformat())
            .execute()
        )
        return result.data or []

    def _find_unit_by_name(unit_name: str) -> dict | None:
        units = _active_units()
        needle = unit_name.strip().lower()
        for u in units:
            if u["name"].strip().lower() == needle:
                return u
        for u in units:
            if needle in u["name"].strip().lower():
                return u
        return None

    @tool
    def check_stay_availability(check_in: str, check_out: str, guests: int = 1) -> str:
        """Consulta unidades/quartos disponíveis para hospedagem no período.
        Use quando o hóspede perguntar sobre disponibilidade, datas ou quiser reservar.
        SEMPRE verifique disponibilidade antes de cotar ou criar reserva.

        Args:
            check_in: Data de entrada no formato YYYY-MM-DD
            check_out: Data de saída no formato YYYY-MM-DD
            guests: Número de hóspedes (padrão: 1)
        """
        try:
            ci, co = _parse_date(check_in), _parse_date(check_out)
            if not ci or not co:
                return "[ERRO] Datas inválidas. Use o formato YYYY-MM-DD (ex: 2026-08-15)."
            if co <= ci:
                return "[ERRO] A data de saída deve ser depois da data de entrada."
            if ci < date.today():
                return "[ERRO] A data de entrada já passou. Peça datas futuras ao hóspede."

            nights = (co - ci).days
            units = _active_units()
            if not units:
                return "Nenhuma unidade de hospedagem cadastrada neste workspace."

            available = []
            for u in units:
                if int(u.get("capacity") or 1) < guests:
                    continue
                if not _unit_is_free(u["id"], ci, co):
                    continue
                overrides = _overrides_for_unit(u["id"], ci, co)
                min_nights = max(
                    int(u.get("min_nights") or 1),
                    max([int(ov["min_nights"]) for ov in overrides if ov.get("min_nights")] or [1]),
                )
                if nights < min_nights:
                    available.append(
                        f"- {u['name']}: exige mínimo de {min_nights} noites (período pedido: {nights})"
                    )
                    continue
                rates = _nightly_rates(u, overrides, ci, co)
                total = round(sum(rates), 2)
                prop = u.get("stay_properties") or {}
                line = (
                    f"- {u['name']} | até {u.get('capacity', 1)} hóspedes | "
                    f"{nights} noite(s) | Total: R$ {total:.2f}"
                )
                if u.get("description"):
                    line += f"\n  {u['description'][:120]}"
                if prop.get("check_in_time"):
                    line += f"\n  Check-in a partir de {str(prop['check_in_time'])[:5]}, check-out até {str(prop.get('check_out_time', ''))[:5]}"
                available.append(line)

            if not available:
                return (
                    f"Nenhuma unidade disponível para {guests} hóspede(s) "
                    f"de {ci.strftime('%d/%m/%Y')} a {co.strftime('%d/%m/%Y')}. "
                    "Ofereça datas alternativas ao hóspede."
                )

            return (
                f"Disponibilidade de {ci.strftime('%d/%m/%Y')} a {co.strftime('%d/%m/%Y')} "
                f"({nights} noite(s), {guests} hóspede(s)):\n" + "\n".join(available)
            )
        except Exception as e:
            return f"Erro ao consultar disponibilidade: {e}"

    @tool
    def quote_stay(unit_name: str, check_in: str, check_out: str, guests: int = 1) -> str:
        """Calcula o valor total da estadia em uma unidade específica.
        Use depois de verificar disponibilidade, quando o hóspede escolher uma unidade.
        Informe o total, o valor do sinal e as condições ao hóspede.

        Args:
            unit_name: Nome da unidade/quarto escolhido
            check_in: Data de entrada (YYYY-MM-DD)
            check_out: Data de saída (YYYY-MM-DD)
            guests: Número de hóspedes
        """
        try:
            ci, co = _parse_date(check_in), _parse_date(check_out)
            if not ci or not co or co <= ci:
                return "[ERRO] Datas inválidas. Use o formato YYYY-MM-DD e saída depois da entrada."

            unit = _find_unit_by_name(unit_name)
            if not unit:
                return f"Unidade '{unit_name}' não encontrada. Use check_stay_availability para listar as unidades."

            if int(unit.get("capacity") or 1) < guests:
                return f"A unidade {unit['name']} comporta até {unit.get('capacity', 1)} hóspede(s) — não atende {guests}."

            if not _unit_is_free(unit["id"], ci, co):
                return f"A unidade {unit['name']} NÃO está disponível nesse período. Verifique outras datas ou unidades."

            nights = (co - ci).days
            overrides = _overrides_for_unit(unit["id"], ci, co)
            rates = _nightly_rates(unit, overrides, ci, co)
            total = round(sum(rates), 2)
            deposit = round(total * DEPOSIT_PERCENT / 100, 2)
            avg = round(total / nights, 2) if nights else 0

            prop = unit.get("stay_properties") or {}
            lines = [
                f"Cotação — {unit['name']}",
                f"Período: {ci.strftime('%d/%m/%Y')} → {co.strftime('%d/%m/%Y')} ({nights} noite(s))",
                f"Hóspedes: {guests}",
                f"Diária média: R$ {avg:.2f}",
                f"TOTAL: R$ {total:.2f}",
                f"Sinal para confirmar ({DEPOSIT_PERCENT}%): R$ {deposit:.2f} via Pix",
                f"Restante (R$ {total - deposit:.2f}) pago no check-in.",
            ]
            if prop.get("policies"):
                lines.append(f"Políticas: {prop['policies'][:200]}")
            lines.append("Se o hóspede aprovar, use create_stay_reservation para reservar e enviar o Pix do sinal.")
            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao cotar estadia: {e}"

    @tool
    def create_stay_reservation(unit_name: str, check_in: str, check_out: str, guest_name: str, guests: int = 1, notes: str = "") -> str:
        """Cria a reserva e gera a cobrança do sinal via Pix.
        Use SOMENTE depois que o hóspede aprovar a cotação e confirmar os dados.
        A unidade fica reservada por 24h aguardando o pagamento do sinal.

        Args:
            unit_name: Nome da unidade/quarto
            check_in: Data de entrada (YYYY-MM-DD)
            check_out: Data de saída (YYYY-MM-DD)
            guest_name: Nome completo do hóspede
            guests: Número de hóspedes
            notes: Observações (horário de chegada, pedidos especiais)
        """
        try:
            ci, co = _parse_date(check_in), _parse_date(check_out)
            if not ci or not co or co <= ci:
                return "[ERRO] Datas inválidas. Use o formato YYYY-MM-DD e saída depois da entrada."
            if not guest_name.strip():
                return "[ERRO] Informe o nome do hóspede antes de reservar."

            unit = _find_unit_by_name(unit_name)
            if not unit:
                return f"Unidade '{unit_name}' não encontrada."

            if not _unit_is_free(unit["id"], ci, co):
                return f"A unidade {unit['name']} acabou de ficar indisponível nesse período. Ofereça alternativas."

            nights = (co - ci).days
            overrides = _overrides_for_unit(unit["id"], ci, co)
            rates = _nightly_rates(unit, overrides, ci, co)
            total = round(sum(rates), 2)
            deposit = round(total * DEPOSIT_PERCENT / 100, 2)
            avg = round(total / nights, 2) if nights else 0

            lead_phone = None
            try:
                lead = supabase.table("leads").select("phone").eq("id", lead_id).single().execute()
                lead_phone = (lead.data or {}).get("phone")
            except Exception:
                pass

            hold_expires = datetime.now(timezone.utc) + timedelta(hours=HOLD_HOURS)
            insert = (
                supabase.table("stay_reservations")
                .insert({
                    "workspace_id": workspace_id,
                    "unit_id": unit["id"],
                    "lead_id": lead_id,
                    "guest_name": guest_name.strip(),
                    "guest_phone": lead_phone,
                    "check_in": ci.isoformat(),
                    "check_out": co.isoformat(),
                    "guests": guests,
                    "nightly_rate": avg,
                    "total_amount": total,
                    "deposit_amount": deposit,
                    "deposit_status": "pending",
                    "status": "pending",
                    "hold_expires_at": hold_expires.isoformat(),
                    "notes": notes.strip() or None,
                    "source": "agent",
                })
                .execute()
            )
            reservation = (insert.data or [{}])[0]
            reservation_id = reservation.get("id")
            if not reservation_id:
                return "[ERRO] Não foi possível criar a reserva. Tente novamente."

            # ── Sinal via Pix (tenant-payments, mesmo fluxo do send_payment_link) ──
            payment_msg = ""
            supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
            service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            description = (
                f"Sinal reserva {unit['name']} — "
                f"{ci.strftime('%d/%m')} a {co.strftime('%d/%m')} ({guest_name.strip()})"
            )
            if supabase_url and service_key:
                try:
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
                                "amount": deposit,
                                "description": description,
                                "due_date": hold_expires.strftime("%Y-%m-%d"),
                                "payment_method": "pix",
                                "source": "agent",
                                "send_now": True,
                            },
                        )
                        data = resp.json() if resp.content else {}
                        inv = data.get("invoice") or {}
                        if resp.is_success and inv.get("id"):
                            supabase.table("stay_reservations").update(
                                {"payment_id": inv["id"]}
                            ).eq("id", reservation_id).execute()
                            if inv.get("payment_url"):
                                payment_msg = f"Link de pagamento do sinal: {inv['payment_url']}"
                            elif inv.get("pix_code"):
                                payment_msg = f"PIX Copia e Cola do sinal: {inv['pix_code']}"
                            else:
                                payment_msg = "Cobrança do sinal registrada e enviada ao hóspede."
                        else:
                            payment_msg = (
                                "⚠️ Não foi possível gerar o Pix automaticamente. "
                                "Peça ao administrador para verificar o gateway de pagamento."
                            )
                except Exception as pay_err:
                    payment_msg = f"⚠️ Falha ao gerar cobrança do sinal: {pay_err}"

            return (
                f"✅ Reserva criada (aguardando sinal)!\n"
                f"Unidade: {unit['name']}\n"
                f"Hóspede: {guest_name.strip()} ({guests} pessoa(s))\n"
                f"Período: {ci.strftime('%d/%m/%Y')} → {co.strftime('%d/%m/%Y')} ({nights} noite(s))\n"
                f"Total: R$ {total:.2f} | Sinal: R$ {deposit:.2f}\n"
                f"{payment_msg}\n"
                f"A unidade fica reservada por {HOLD_HOURS}h aguardando o pagamento do sinal. "
                f"Após a confirmação do Pix a reserva é confirmada automaticamente."
            )
        except Exception as e:
            return f"Erro ao criar reserva: {e}"

    @tool
    def manage_stay_reservation(action: str, new_check_in: str = "", new_check_out: str = "") -> str:
        """Consulta, cancela ou reagenda as reservas de hospedagem do hóspede atual.
        Use quando o hóspede perguntar sobre a reserva dele, quiser cancelar ou mudar as datas.

        Args:
            action: 'list' (consultar), 'cancel' (cancelar) ou 'reschedule' (mudar datas)
            new_check_in: Nova data de entrada (YYYY-MM-DD, apenas para reschedule)
            new_check_out: Nova data de saída (YYYY-MM-DD, apenas para reschedule)
        """
        try:
            result = (
                supabase.table("stay_reservations")
                .select("id, check_in, check_out, guests, total_amount, deposit_amount, deposit_status, status, unit_id, stay_units(name)")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .in_("status", ["pending", "confirmed", "checked_in"])
                .order("check_in", desc=False)
                .limit(5)
                .execute()
            )
            reservations = result.data or []
            if not reservations:
                return "Este hóspede não tem reservas ativas."

            status_map = {
                "pending": "⏳ Aguardando sinal",
                "confirmed": "✅ Confirmada",
                "checked_in": "🏠 Hospedado",
            }

            if action == "list":
                lines = []
                for r in reservations:
                    unit_name = (r.get("stay_units") or {}).get("name", "Unidade")
                    ci = _parse_date(r["check_in"])
                    co = _parse_date(r["check_out"])
                    lines.append(
                        f"- {unit_name} | {ci.strftime('%d/%m/%Y')} → {co.strftime('%d/%m/%Y')} | "
                        f"{r['guests']} hóspede(s) | R$ {float(r['total_amount'] or 0):.2f} | "
                        f"{status_map.get(r['status'], r['status'])}"
                    )
                return "Reservas do hóspede:\n" + "\n".join(lines)

            # cancel/reschedule atuam na próxima reserva ativa
            target = reservations[0]
            unit_name = (target.get("stay_units") or {}).get("name", "Unidade")

            if action == "cancel":
                supabase.table("stay_reservations").update(
                    {"status": "cancelled"}
                ).eq("id", target["id"]).execute()
                return (
                    f"✅ Reserva na unidade {unit_name} "
                    f"({_parse_date(target['check_in']).strftime('%d/%m/%Y')}) cancelada. "
                    "Se o sinal já foi pago, oriente o hóspede sobre o reembolso conforme a política da casa."
                )

            if action == "reschedule":
                ci, co = _parse_date(new_check_in), _parse_date(new_check_out)
                if not ci or not co or co <= ci:
                    return "[ERRO] Informe as novas datas no formato YYYY-MM-DD (saída depois da entrada)."
                if not _unit_is_free(target["unit_id"], ci, co, ignore_reservation_id=target["id"]):
                    return f"A unidade {unit_name} não está livre no novo período. Ofereça outras datas."

                unit = (
                    supabase.table("stay_units")
                    .select("id, name, base_rate, min_nights")
                    .eq("id", target["unit_id"])
                    .single()
                    .execute()
                ).data
                overrides = _overrides_for_unit(target["unit_id"], ci, co)
                rates = _nightly_rates(unit, overrides, ci, co)
                total = round(sum(rates), 2)
                nights = (co - ci).days

                supabase.table("stay_reservations").update({
                    "check_in": ci.isoformat(),
                    "check_out": co.isoformat(),
                    "total_amount": total,
                    "nightly_rate": round(total / nights, 2) if nights else 0,
                }).eq("id", target["id"]).execute()

                return (
                    f"✅ Reserva remarcada!\n"
                    f"Unidade: {unit_name}\n"
                    f"Novo período: {ci.strftime('%d/%m/%Y')} → {co.strftime('%d/%m/%Y')} ({nights} noite(s))\n"
                    f"Novo total: R$ {total:.2f}"
                    + (
                        " (o sinal já pago permanece válido)."
                        if target.get("deposit_status") == "paid"
                        else "."
                    )
                )

            return "[ERRO] Ação inválida. Use 'list', 'cancel' ou 'reschedule'."
        except Exception as e:
            return f"Erro ao gerenciar reserva: {e}"

    return {
        "check_stay_availability": check_stay_availability,
        "quote_stay": quote_stay,
        "create_stay_reservation": create_stay_reservation,
        "manage_stay_reservation": manage_stay_reservation,
    }

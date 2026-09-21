"""
============================================
AutoZap Rider - Mobility Tools
============================================
Ferramentas do vertical de mobilidade: cotação de corrida,
solicitação com despacho de motorista via WhatsApp e cobrança Pix.

Distância: Google Maps Distance Matrix (GOOGLE_MAPS_API_KEY) com
fallback Haversine quando o passageiro compartilha localização.
"""

import math
import os
import re
from datetime import datetime, timezone, timedelta

import httpx
from langchain_core.tools import tool
from supabase import Client

RIDER_TOOL_NAMES = (
    "quote_ride",
    "request_ride",
    "check_ride_status",
    "cancel_ride",
)

VEHICLE_LABELS = {"car": "Carro", "moto": "Moto", "van": "Van"}

# Fallback quando o workspace não configurou regras de preço
DEFAULT_PRICING = {"base_fare": 5.0, "per_km": 2.5, "minimum_fare": 10.0, "night_multiplier": 1.0}

_COORDS_RE = re.compile(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)")


def workspace_has_rider(supabase: Client, workspace_id: str) -> bool:
    """Entitlement do add-on Rider (trial libera para avaliação). Fail-open."""
    try:
        result = (
            supabase.table("subscriptions")
            .select("addon_rider, plan_type")
            .eq("workspace_id", workspace_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            return True
        sub = result.data[0]
        return bool(sub.get("addon_rider")) or sub.get("plan_type") == "trial"
    except Exception as e:
        print(f"[rider_tools] entitlement check failed (fail-open): {e}")
        return True


def _parse_coords(text: str) -> tuple[float, float] | None:
    """Extrai 'lat,lng' de textos como '[Localização 📍] -23.55,-46.63 (...)'."""
    match = _COORDS_RE.search(text or "")
    if not match:
        return None
    lat, lng = float(match.group(1)), float(match.group(2))
    if abs(lat) > 90 or abs(lng) > 180:
        return None
    return lat, lng


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lng1 = a
    lat2, lng2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _distance(origin: str, destination: str) -> tuple[float, int] | None:
    """Distância (km) e duração (min) entre origem e destino.

    1. Google Maps Distance Matrix (aceita endereço ou 'lat,lng')
    2. Fallback Haversine × 1.4 (fator de rota) quando ambos são coordenadas
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    o_coords = _parse_coords(origin)
    d_coords = _parse_coords(destination)

    if api_key:
        try:
            o_param = f"{o_coords[0]},{o_coords[1]}" if o_coords else origin
            d_param = f"{d_coords[0]},{d_coords[1]}" if d_coords else destination
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    "https://maps.googleapis.com/maps/api/distancematrix/json",
                    params={
                        "origins": o_param,
                        "destinations": d_param,
                        "mode": "driving",
                        "key": api_key,
                    },
                )
                data = resp.json()
                element = data["rows"][0]["elements"][0]
                if element.get("status") == "OK":
                    km = element["distance"]["value"] / 1000
                    minutes = round(element["duration"]["value"] / 60)
                    return round(km, 2), minutes
        except Exception as e:
            print(f"[rider_tools] Distance Matrix failed: {e}")

    if o_coords and d_coords:
        km = _haversine_km(o_coords, d_coords) * 1.4  # fator de rota urbana
        minutes = max(round(km / 30 * 60), 5)  # ~30 km/h médio urbano
        return round(km, 2), minutes

    return None


def create_rider_tools(supabase: Client, workspace_id: str, lead_id: str) -> dict:
    """Cria as ferramentas Rider contextualizadas. Retorna {nome: tool}."""

    def _pricing(vehicle_type: str) -> dict:
        try:
            result = (
                supabase.table("rider_pricing_rules")
                .select("base_fare, per_km, minimum_fare, night_multiplier, night_start, night_end")
                .eq("workspace_id", workspace_id)
                .eq("vehicle_type", vehicle_type)
                .eq("active", True)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]
        except Exception:
            pass
        return dict(DEFAULT_PRICING)

    def _price_for(km: float, vehicle_type: str) -> float:
        rule = _pricing(vehicle_type)
        price = float(rule.get("base_fare") or 0) + km * float(rule.get("per_km") or 0)
        price = max(price, float(rule.get("minimum_fare") or 0))

        multiplier = float(rule.get("night_multiplier") or 1)
        if multiplier > 1:
            # Horário local (Brasília)
            local = datetime.now(timezone.utc) - timedelta(hours=3)
            start = str(rule.get("night_start") or "22:00")[:5]
            end = str(rule.get("night_end") or "06:00")[:5]
            now_hm = local.strftime("%H:%M")
            in_night = now_hm >= start or now_hm < end if start > end else start <= now_hm < end
            if in_night:
                price *= multiplier
        return round(price, 2)

    def _active_ride() -> dict | None:
        result = (
            supabase.table("rider_rides")
            .select("*, rider_drivers!rider_rides_driver_id_fkey(name, phone, vehicle_description, plate)")
            .eq("workspace_id", workspace_id)
            .eq("lead_id", lead_id)
            .in_("status", ["searching", "assigned", "started"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    @tool
    def quote_ride(origin: str, destination: str, vehicle_type: str = "car") -> str:
        """Cota o valor de uma corrida entre origem e destino.
        Use quando o passageiro pedir uma corrida ou perguntar o preço.
        Aceita endereços por escrito ou coordenadas de localização compartilhada.

        Args:
            origin: Endereço ou coordenadas de origem (ex: "Rua X, 100" ou "-23.55,-46.63")
            destination: Endereço ou coordenadas de destino
            vehicle_type: Tipo de veículo: 'car', 'moto' ou 'van' (padrão: car)
        """
        try:
            if vehicle_type not in VEHICLE_LABELS:
                vehicle_type = "car"
            if not origin.strip() or not destination.strip():
                return "[ERRO] Informe origem e destino para cotar a corrida."

            dist = _distance(origin, destination)
            if not dist:
                return (
                    "Não consegui calcular a distância entre esses pontos. "
                    "Peça ao passageiro um endereço mais completo (rua, número e bairro) "
                    "ou que compartilhe a localização pelo WhatsApp (clipe 📎 → Localização)."
                )

            km, minutes = dist
            price = _price_for(km, vehicle_type)

            return (
                f"Cotação da corrida ({VEHICLE_LABELS[vehicle_type]}):\n"
                f"Origem: {origin}\n"
                f"Destino: {destination}\n"
                f"Distância: {km:.1f} km | Tempo estimado: ~{minutes} min\n"
                f"VALOR: R$ {price:.2f}\n"
                f"Se o passageiro aprovar, use request_ride para chamar um motorista."
            )
        except Exception as e:
            return f"Erro ao cotar corrida: {e}"

    @tool
    def request_ride(origin: str, destination: str, vehicle_type: str = "car", payment_method: str = "cash") -> str:
        """Solicita a corrida e despacha para os motoristas disponíveis via WhatsApp.
        Use SOMENTE depois que o passageiro aprovar a cotação.

        Args:
            origin: Endereço ou coordenadas de origem
            destination: Endereço ou coordenadas de destino
            vehicle_type: 'car', 'moto' ou 'van'
            payment_method: 'cash' (dinheiro na hora), 'pix' (antecipado) ou 'card'
        """
        try:
            if vehicle_type not in VEHICLE_LABELS:
                vehicle_type = "car"
            if payment_method not in ("cash", "pix", "card"):
                payment_method = "cash"

            existing = _active_ride()
            if existing:
                return (
                    "Este passageiro já tem uma corrida em andamento "
                    f"(status: {existing['status']}). Use check_ride_status ou cancel_ride antes de pedir outra."
                )

            dist = _distance(origin, destination)
            km, minutes = dist if dist else (None, None)
            price = _price_for(km, vehicle_type) if km is not None else None

            # Motoristas online do tipo certo
            drivers = (
                supabase.table("rider_drivers")
                .select("id")
                .eq("workspace_id", workspace_id)
                .eq("vehicle_type", vehicle_type)
                .eq("is_online", True)
                .eq("active", True)
                .execute()
            )
            if not (drivers.data or []):
                return (
                    f"⚠️ Nenhum motorista de {VEHICLE_LABELS[vehicle_type]} online agora. "
                    "Informe o passageiro e sugira tentar mais tarde."
                )

            lead = {}
            try:
                lead = (
                    supabase.table("leads").select("name, phone").eq("id", lead_id).single().execute()
                ).data or {}
            except Exception:
                pass

            o_coords = _parse_coords(origin)
            d_coords = _parse_coords(destination)
            insert = (
                supabase.table("rider_rides")
                .insert({
                    "workspace_id": workspace_id,
                    "lead_id": lead_id,
                    "passenger_name": lead.get("name"),
                    "passenger_phone": lead.get("phone"),
                    "vehicle_type": vehicle_type,
                    "origin_text": origin,
                    "origin_lat": o_coords[0] if o_coords else None,
                    "origin_lng": o_coords[1] if o_coords else None,
                    "destination_text": destination,
                    "destination_lat": d_coords[0] if d_coords else None,
                    "destination_lng": d_coords[1] if d_coords else None,
                    "distance_km": km,
                    "duration_min": minutes,
                    "price": price,
                    "status": "searching",
                    "payment_method": payment_method,
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                    "source": "agent",
                })
                .execute()
            )
            ride = (insert.data or [{}])[0]
            ride_id = ride.get("id")
            if not ride_id:
                return "[ERRO] Não foi possível registrar a corrida. Tente novamente."

            # Dispara o despacho (notifica o primeiro motorista da fila)
            supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
            service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            dispatch_msg = ""
            if supabase_url and service_key:
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(
                            f"{supabase_url}/functions/v1/rider-dispatch",
                            headers={
                                "Authorization": f"Bearer {service_key}",
                                "Content-Type": "application/json",
                            },
                            json={"action": "dispatch", "ride_id": ride_id},
                        )
                        data = resp.json() if resp.content else {}
                        if resp.is_success and data.get("dispatched"):
                            dispatch_msg = "Motorista sendo acionado agora."
                        else:
                            dispatch_msg = data.get("error") or "Despacho em processamento."
                except Exception as e:
                    dispatch_msg = f"⚠️ Falha ao acionar despacho: {e}"

            pix_msg = ""
            if payment_method == "pix" and price:
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
                                "amount": price,
                                "description": f"Corrida {origin[:40]} → {destination[:40]}",
                                "due_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                "payment_method": "pix",
                                "source": "agent",
                                "send_now": True,
                            },
                        )
                        data = resp.json() if resp.content else {}
                        inv = data.get("invoice") or {}
                        if resp.is_success and inv.get("id"):
                            supabase.table("rider_rides").update(
                                {"payment_id": inv["id"]}
                            ).eq("id", ride_id).execute()
                            if inv.get("pix_code"):
                                pix_msg = f"\nPIX Copia e Cola: {inv['pix_code']}"
                            elif inv.get("payment_url"):
                                pix_msg = f"\nLink de pagamento: {inv['payment_url']}"
                except Exception as e:
                    pix_msg = f"\n⚠️ Falha ao gerar Pix: {e}"

            price_line = f"Valor: R$ {price:.2f}\n" if price else ""
            return (
                f"🚗 Corrida solicitada!\n"
                f"Origem: {origin}\n"
                f"Destino: {destination}\n"
                f"{price_line}"
                f"{dispatch_msg}\n"
                f"Avise o passageiro que confirmaremos assim que um motorista aceitar."
                f"{pix_msg}"
            )
        except Exception as e:
            return f"Erro ao solicitar corrida: {e}"

    @tool
    def check_ride_status() -> str:
        """Consulta o status da corrida atual do passageiro.
        Use quando o passageiro perguntar onde está o motorista ou o andamento da corrida.
        """
        try:
            ride = _active_ride()
            if not ride:
                last = (
                    supabase.table("rider_rides")
                    .select("status, origin_text, destination_text, price, created_at")
                    .eq("workspace_id", workspace_id)
                    .eq("lead_id", lead_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if last.data:
                    r = last.data[0]
                    return (
                        f"Nenhuma corrida ativa. Última corrida: {r['origin_text']} → {r['destination_text']} "
                        f"(status: {r['status']})."
                    )
                return "Este passageiro não tem corridas registradas."

            status_map = {
                "searching": "🔎 Procurando motorista",
                "assigned": "✅ Motorista a caminho",
                "started": "🚗 Corrida em andamento",
            }
            lines = [
                f"Status: {status_map.get(ride['status'], ride['status'])}",
                f"Trajeto: {ride['origin_text']} → {ride['destination_text']}",
            ]
            if ride.get("price"):
                lines.append(f"Valor: R$ {float(ride['price']):.2f}")
            driver = ride.get("rider_drivers")
            if driver:
                d_desc = " - ".join(filter(None, [driver.get("vehicle_description"), driver.get("plate")]))
                lines.append(f"Motorista: {driver.get('name')}{f' ({d_desc})' if d_desc else ''}")
            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao consultar corrida: {e}"

    @tool
    def cancel_ride() -> str:
        """Cancela a corrida ativa do passageiro.
        Use quando o passageiro desistir da corrida. Confirme antes de cancelar.
        """
        try:
            ride = _active_ride()
            if not ride:
                return "Não há corrida ativa para cancelar."

            supabase.table("rider_rides").update({
                "status": "cancelled",
                "cancelled_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", ride["id"]).execute()

            # Avisa o motorista se já estava atribuído
            driver = ride.get("rider_drivers")
            if driver and driver.get("phone"):
                supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
                service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
                if supabase_url and service_key:
                    try:
                        with httpx.Client(timeout=20.0) as client:
                            client.post(
                                f"{supabase_url}/functions/v1/rider-dispatch",
                                headers={
                                    "Authorization": f"Bearer {service_key}",
                                    "Content-Type": "application/json",
                                },
                                json={"action": "notify_cancelled", "ride_id": ride["id"]},
                            )
                    except Exception:
                        pass

            return "✅ Corrida cancelada. Informe o passageiro."
        except Exception as e:
            return f"Erro ao cancelar corrida: {e}"

    return {
        "quote_ride": quote_ride,
        "request_ride": request_ride,
        "check_ride_status": check_ride_status,
        "cancel_ride": cancel_ride,
    }

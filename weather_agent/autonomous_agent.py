import logging
from typing import Any

from .memory_store import (
    append_conversation,
    get_recent_conversation,
    get_user_profile,
    normalize_user_id,
    upsert_user_profile,
)
from .personas import apply_persona_style, resolve_persona
from .weather_service import (
    WEATHER_LIVE_DATA_UNAVAILABLE,
    build_weather_answer_from_tool,
    decode_weather_tool_payload,
    get_weather_forecast,
    infer_city_from_text,
)

LOGGER = logging.getLogger("weather_agent.autonomous")


def _ensure_city_in_input(user_input: str, city_name: str) -> str:
    if city_name.lower() in user_input.lower():
        return user_input
    return f"{user_input} in {city_name}".strip()


def _trace_step(trace: list[dict[str, Any]], phase: str, detail: dict[str, Any]) -> None:
    item = {
        "step": len(trace) + 1,
        "phase": phase,
        "detail": detail,
    }
    trace.append(item)
    LOGGER.info("agent_trace step=%s phase=%s detail=%s", item["step"], phase, detail)


def _status_from_payload(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "service_unavailable"
    return str(payload.get("status") or "service_unavailable").strip().lower()


def run_autonomous_weather_agent(
    user_input: str,
    city_hint: str | None,
    user_id: str | None,
    persona_id: str | None,
    preference_updates: dict[str, Any] | None,
    remember_memory: bool,
    max_steps: int = 4,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    safe_user_id = normalize_user_id(user_id)
    profile = get_user_profile(safe_user_id)
    updates = dict(preference_updates or {})

    effective_persona_id = str(persona_id or profile.get("persona_id") or "professional")
    effective_units = str(updates.get("units") or profile.get("units") or "metric").strip().lower()
    if effective_units not in {"metric", "imperial"}:
        effective_units = "metric"

    effective_style = str(updates.get("response_style") or profile.get("response_style") or "balanced").strip().lower()
    if effective_style not in {"brief", "balanced", "detailed"}:
        effective_style = "balanced"

    preferred_city = updates.get("city") or city_hint or profile.get("preferred_city")
    inferred_city = infer_city_from_text(user_input) or None
    plan_city = inferred_city or preferred_city
    persona = resolve_persona(effective_persona_id)

    _trace_step(
        trace,
        "plan",
        {
            "user_id": safe_user_id,
            "persona_id": persona.get("id"),
            "units": effective_units,
            "response_style": effective_style,
            "city_from_memory": profile.get("preferred_city"),
            "city_for_plan": plan_city,
            "max_steps": max_steps,
        },
    )

    query = str(user_input or "").strip()
    if plan_city:
        query = _ensure_city_in_input(query, str(plan_city))

    tool_payload: dict[str, Any] = {"status": "service_unavailable", "message": WEATHER_LIVE_DATA_UNAVAILABLE}
    for attempt in range(1, max_steps + 1):
        _trace_step(trace, "tool-call", {"attempt": attempt, "tool": "get_weather_forecast", "query": query})

        raw_payload = get_weather_forecast(query)
        parsed_payload = decode_weather_tool_payload(raw_payload)
        if isinstance(parsed_payload, dict):
            tool_payload = parsed_payload
        else:
            tool_payload = {"status": "service_unavailable", "message": WEATHER_LIVE_DATA_UNAVAILABLE}

        status = _status_from_payload(tool_payload)
        location = tool_payload.get("location")
        _trace_step(
            trace,
            "observe",
            {
                "attempt": attempt,
                "status": status,
                "location": location,
            },
        )

        if status == "ok":
            _trace_step(trace, "reflect", {"attempt": attempt, "decision": "stop", "reason": "data_sufficient"})
            break

        if status == "needs_location":
            city_from_memory = profile.get("preferred_city")
            if isinstance(city_from_memory, str) and city_from_memory.strip():
                query = _ensure_city_in_input(user_input, city_from_memory.strip())
                _trace_step(
                    trace,
                    "reflect",
                    {
                        "attempt": attempt,
                        "decision": "continue",
                        "reason": "retry_with_memory_city",
                        "memory_city": city_from_memory,
                    },
                )
                continue
            _trace_step(
                trace,
                "reflect",
                {"attempt": attempt, "decision": "stop", "reason": "missing_city_and_no_memory_city"},
            )
            break

        if status in {"ambiguous_location"}:
            _trace_step(trace, "reflect", {"attempt": attempt, "decision": "stop", "reason": status})
            break

        if attempt < max_steps:
            _trace_step(
                trace,
                "reflect",
                {"attempt": attempt, "decision": "continue", "reason": f"status_{status}_retry"},
            )
            continue

        _trace_step(trace, "reflect", {"attempt": attempt, "decision": "stop", "reason": f"status_{status}"})
        break

    resolved_city = str(
        city_hint
        or inferred_city
        or tool_payload.get("location")
        or profile.get("preferred_city")
        or "unknown"
    )
    base_answer = build_weather_answer_from_tool(user_input, tool_payload, units=effective_units)

    recent_memory = get_recent_conversation(safe_user_id, limit=3)
    context_bits = []
    if recent_memory:
        context_bits.append(f"memory_messages={len(recent_memory)}")
    if profile.get("preferred_city"):
        context_bits.append(f"preferred_city={profile.get('preferred_city')}")
    context_summary = ", ".join(context_bits) if context_bits else None

    final_answer = apply_persona_style(
        base_answer,
        persona=persona,
        response_style=effective_style,
        include_context=context_summary if effective_style == "detailed" else None,
    )
    _trace_step(
        trace,
        "final-answer",
        {
            "persona_id": persona.get("id"),
            "response_style": effective_style,
            "units": effective_units,
        },
    )

    discovered_city = tool_payload.get("location")
    if remember_memory:
        profile = upsert_user_profile(
            safe_user_id,
            persona_id=str(persona.get("id") or effective_persona_id),
            preferred_city=str(discovered_city or preferred_city or profile.get("preferred_city") or "").strip() or None,
            units=effective_units,
            response_style=effective_style,
        )
        append_conversation(safe_user_id, "user", user_input)
        append_conversation(safe_user_id, "assistant", final_answer)

    return {
        "response_text": final_answer,
        "tool_payload": tool_payload,
        "resolved_city": resolved_city,
        "trace": trace,
        "profile": profile,
        "persona_id": str(persona.get("id")),
        "units": effective_units,
        "response_style": effective_style,
    }

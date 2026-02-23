import logging
from typing import Any

from .agent_service import invoke_llm_weather_agent
from .memory_store import (
    append_conversation,
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

AGENT_FAILURE_MARKERS = (
    "unable to fetch",
    "technical issue",
    "knowledge cutoff",
    "don't have real-time",
    "do not have real-time",
    "cannot access",
    "can't access",
    "cannot confirm",
    "cannot execute tools",
    "current limitation",
    "check a weather website",
    "provide the data",
    "service unavailable",
)


def _default_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "persona_id": "professional",
        "preferred_city": None,
        "units": "metric",
        "response_style": "brief",
        "updated_at": None,
    }


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


def _looks_like_agent_failure(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in AGENT_FAILURE_MARKERS)


def _looks_like_unfocused_agent_response(text: str) -> bool:
    lowered = str(text or "").lower()
    if "http://" in lowered or "https://" in lowered:
        return True
    if "for example" in lowered:
        return True
    asks_followup = "?" in lowered and ("do you need" in lowered or "is" in lowered)
    return asks_followup


def _looks_too_verbose(text: str) -> bool:
    payload = str(text or "")
    if len(payload) > 520:
        return True
    if "\n-" in payload or "\n1." in payload:
        return True
    return False


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
    profile = get_user_profile(safe_user_id) if remember_memory else _default_profile(safe_user_id)
    updates = dict(preference_updates or {})

    effective_persona_id = str(persona_id or profile.get("persona_id") or "professional")
    effective_units = str(updates.get("units") or profile.get("units") or "metric").strip().lower()
    if effective_units not in {"metric", "imperial"}:
        effective_units = "metric"

    effective_style = str(updates.get("response_style") or profile.get("response_style") or "brief").strip().lower()
    if effective_style not in {"brief", "balanced", "detailed"}:
        effective_style = "brief"

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

    llm_response = ""
    llm_tool_payload: dict[str, Any] | None = None
    llm_result = invoke_llm_weather_agent(
        user_input=query,
        persona=persona,
        response_style=effective_style,
        memory_city=profile.get("preferred_city") if remember_memory else None,
    )
    if isinstance(llm_result, dict):
        _trace_step(
            trace,
            "thought",
            {
                "engine": "llm-agent",
                "persona_id": persona.get("id"),
                "response_style": effective_style,
            },
        )
        llm_error = llm_result.get("error")
        if isinstance(llm_error, str) and llm_error.strip():
            _trace_step(
                trace,
                "observation",
                {
                    "engine": "llm-agent",
                    "status": "error",
                    "reason": llm_error[:180],
                },
            )
        llm_steps = llm_result.get("steps") if isinstance(llm_result.get("steps"), list) else []
        for index, step in enumerate(llm_steps, start=1):
            if not isinstance(step, dict):
                continue
            _trace_step(
                trace,
                "action",
                {
                    "engine": "llm-agent",
                    "index": index,
                    "tool": str(step.get("tool") or "unknown"),
                },
            )
            _trace_step(
                trace,
                "observation",
                {
                    "engine": "llm-agent",
                    "index": index,
                    "tool_status": str(step.get("status") or "unknown"),
                    "observation_preview": str(step.get("observation_preview") or "")[:140],
                },
            )

        llm_response = str(llm_result.get("output") or "").strip()
        payload_candidate = llm_result.get("tool_payload")
        if isinstance(payload_candidate, dict):
            llm_tool_payload = payload_candidate

    tool_payload: dict[str, Any] = {"status": "service_unavailable", "message": WEATHER_LIVE_DATA_UNAVAILABLE}
    if isinstance(llm_tool_payload, dict):
        tool_payload = llm_tool_payload
    else:
        _trace_step(
            trace,
            "reflect",
            {
                "decision": "continue",
                "reason": "llm_missing_tool_payload_fallback_to_direct_tool",
            },
        )
    if not isinstance(llm_tool_payload, dict):
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
                    {
                        "attempt": attempt,
                        "decision": "stop",
                        "reason": "missing_city_and_no_memory_city",
                    },
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
    status = _status_from_payload(tool_payload)
    use_llm_response = (
        bool(llm_response)
        and status == "ok"
        and not _looks_like_agent_failure(llm_response)
        and not _looks_like_unfocused_agent_response(llm_response)
        and not _looks_too_verbose(llm_response)
        and any(char.isdigit() for char in llm_response)
    )
    selected_answer = llm_response if use_llm_response else base_answer

    final_answer = apply_persona_style(
        selected_answer,
        persona=persona,
        response_style=effective_style,
        include_context=None,
    )
    _trace_step(
        trace,
        "final-answer",
        {
            "persona_id": persona.get("id"),
            "response_style": effective_style,
            "units": effective_units,
            "source": "llm" if use_llm_response else "tool-fallback",
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
        "profile": profile if remember_memory else None,
        "persona_id": str(persona.get("id")),
        "units": effective_units,
        "response_style": effective_style,
    }

import logging
import re
from typing import Any

from .agent_service import invoke_llm_weather_agent
from .memory_store import (
    append_conversation,
    get_user_profile,
    normalize_user_id,
    retrieve_relevant_memories,
    upsert_memory_fact,
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

ACTIVITY_KEYWORDS = (
    "photography",
    "running",
    "jogging",
    "cycling",
    "hiking",
    "trekking",
    "cricket",
    "football",
    "badminton",
    "camping",
    "fishing",
    "picnic",
    "travel",
    "walking",
)

SCHEDULE_KEYWORDS = (
    "weekend",
    "weekday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "morning",
    "afternoon",
    "evening",
    "night",
)

WEATHER_PREFERENCE_PATTERNS = (
    r"\b(no rain|avoid rain|clear sky|cool weather|less humidity|low wind)\b",
    r"\b(good for|best for)\s+([A-Za-z\s]{2,40})\b",
)


def _default_profile(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "persona_id": "friendly",
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


def _extract_durable_memory_facts(
    user_input: str,
    tool_payload: dict[str, Any],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    lowered = str(user_input or "").lower()

    location = tool_payload.get("location")
    if isinstance(location, str) and location.strip():
        facts.append(
            {
                "memory_type": "preferred_city",
                "value": location.strip(),
                "importance": 2.6,
                "source_message": user_input,
            }
        )
        facts.append(
            {
                "memory_type": "location_preference",
                "value": location.strip(),
                "importance": 2.0,
                "source_message": user_input,
            }
        )

    for keyword in ACTIVITY_KEYWORDS:
        if keyword in lowered:
            facts.append(
                {
                    "memory_type": "activity_interest",
                    "value": keyword,
                    "importance": 1.6,
                    "source_message": user_input,
                }
            )

    for keyword in SCHEDULE_KEYWORDS:
        if keyword in lowered:
            facts.append(
                {
                    "memory_type": "schedule_pattern",
                    "value": keyword,
                    "importance": 1.3,
                    "source_message": user_input,
                }
            )

    for pattern in WEATHER_PREFERENCE_PATTERNS:
        for match in re.findall(pattern, lowered):
            value = ""
            if isinstance(match, tuple):
                value = " ".join(part for part in match if isinstance(part, str) and part.strip()).strip()
            elif isinstance(match, str):
                value = match.strip()
            if value:
                facts.append(
                    {
                        "memory_type": "weather_preference",
                        "value": value,
                        "importance": 1.4,
                        "source_message": user_input,
                    }
                )

    preferred_city = profile.get("preferred_city")
    if isinstance(preferred_city, str) and preferred_city.strip():
        facts.append(
            {
                "memory_type": "preferred_city",
                "value": preferred_city.strip(),
                "importance": 2.0,
                "source_message": "profile",
            }
        )
    return facts


def _memory_snippets_for_prompt(relevant_memories: list[dict[str, Any]]) -> list[str]:
    snippets: list[str] = []
    for item in relevant_memories[:8]:
        if not isinstance(item, dict):
            continue
        memory_type = str(item.get("memory_type") or "").strip()
        value = str(item.get("value") or "").strip()
        if not memory_type or not value:
            continue
        snippets.append(f"{memory_type}={value}")
    return snippets


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

    relevant_memories: list[dict[str, Any]] = []
    if remember_memory:
        relevant_memories = retrieve_relevant_memories(safe_user_id, query=user_input, limit=6)
        _trace_step(
            trace,
            "memory-retrieval",
            {
                "retrieved": len(relevant_memories),
                "memory_types": [str(item.get("memory_type")) for item in relevant_memories[:6] if isinstance(item, dict)],
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
        profile_summary=profile if remember_memory else None,
        memory_snippets=_memory_snippets_for_prompt(relevant_memories) if remember_memory else [],
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

    context_label = None
    if remember_memory and relevant_memories and effective_style == "detailed":
        context_label = ", ".join(_memory_snippets_for_prompt(relevant_memories)[:3])

    final_answer = apply_persona_style(
        selected_answer,
        persona=persona,
        response_style=effective_style,
        include_context=context_label,
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
    memory_profile: dict[str, Any] | None = None
    if remember_memory:
        memory_profile = upsert_user_profile(
            safe_user_id,
            persona_id=str(persona.get("id") or effective_persona_id),
            preferred_city=str(discovered_city or preferred_city or profile.get("preferred_city") or "").strip() or None,
            units=effective_units,
            response_style=effective_style,
        )
        source_turn = append_conversation(safe_user_id, "user", user_input)
        append_conversation(safe_user_id, "assistant", final_answer)

        for fact in _extract_durable_memory_facts(user_input, tool_payload, memory_profile):
            upsert_memory_fact(
                safe_user_id,
                memory_type=str(fact.get("memory_type") or ""),
                value=str(fact.get("value") or ""),
                importance=float(fact.get("importance") or 1.0),
                source_turn=source_turn,
                source_message=str(fact.get("source_message") or user_input),
            )

    return {
        "response_text": final_answer,
        "tool_payload": tool_payload,
        "resolved_city": resolved_city,
        "trace": trace,
        "profile": memory_profile if remember_memory else None,
        "persona_id": str(persona.get("id")),
        "units": effective_units,
        "response_style": effective_style,
    }

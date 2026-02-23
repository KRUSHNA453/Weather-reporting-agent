from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .agent_service import agent_executor
from .config import FRONTEND_PATH
from .schemas import ChatRequest, ChatResponse
from .weather_service import (
    WEATHER_LIVE_DATA_UNAVAILABLE,
    build_weather_answer_from_tool,
    chat_fields_from_tool_payload,
    decode_weather_tool_payload,
    get_weather_forecast,
    infer_city_from_text,
)

app = FastAPI()

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
    "since i cannot",
    "current limitation",
    "check a weather website",
    "provide the data",
    "provide the output",
    "so i can assist further",
    "if you share",
    "clarification",
    "weather not found",
    "service unavailable",
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "weather-agent",
        "status": "ok",
        "routes": {
            "chat_post": "/chat",
            "chat_get": "/chat?city=Chennai",
            "ui": "/ui",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(FRONTEND_PATH)


def _clean_response(text: str) -> str:
    cleaned = str(text).replace("```", "").replace("**", "").strip()
    return cleaned


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


def _resolve_city(city_hint: str | None, user_input: str) -> str | None:
    if city_hint and city_hint.strip():
        return city_hint.strip()
    return infer_city_from_text(user_input)


def _ensure_city_in_input(user_input: str, city_name: str) -> str:
    if city_name.lower() in user_input.lower():
        return user_input
    return f"{user_input} in {city_name}".strip()


def _extract_tool_payload_from_steps(result: dict[str, Any]) -> dict[str, Any] | None:
    steps = result.get("intermediate_steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, tuple) or len(step) != 2:
            continue
        action, observation = step
        tool_name = str(getattr(action, "tool", "")).strip()
        if tool_name != "get_weather_forecast":
            continue
        return decode_weather_tool_payload(str(observation))
    return None


def _tool_payload_for_query(user_input: str) -> dict[str, Any]:
    payload = decode_weather_tool_payload(get_weather_forecast(user_input))
    if isinstance(payload, dict):
        return payload
    return {"status": "service_unavailable", "message": WEATHER_LIVE_DATA_UNAVAILABLE}


def _build_chat_response(response_text: str, tool_payload: dict[str, Any], fallback_city: str) -> ChatResponse:
    status = str(tool_payload.get("status") or "").lower()
    if status == "ok":
        fields = chat_fields_from_tool_payload(tool_payload)
        return ChatResponse(
            city=str(fields.get("city") or fallback_city or "unknown"),
            response=response_text,
            temperature_c=fields.get("temperature_c"),
            humidity_percent=fields.get("humidity_percent"),
            wind_speed_mps=fields.get("wind_speed_mps"),
            forecast=fields.get("forecast") or [],
        )
    return ChatResponse(city=fallback_city or "unknown", response=response_text, forecast=[])


def _chat_logic(user_input: str, city_hint: str | None) -> tuple[str, dict[str, Any], str]:
    resolved_city = _resolve_city(city_hint, user_input)
    enriched_input = _ensure_city_in_input(user_input, resolved_city) if resolved_city else user_input
    llm_response = ""
    tool_payload: dict[str, Any] | None = None

    if agent_executor is not None:
        try:
            result = agent_executor.invoke({"input": enriched_input})
            if isinstance(result, dict):
                llm_response = _clean_response(str(result.get("output") or ""))
                tool_payload = _extract_tool_payload_from_steps(result)
            else:
                llm_response = _clean_response(str(result))
        except Exception:
            llm_response = ""

    if not isinstance(tool_payload, dict):
        tool_payload = _tool_payload_for_query(enriched_input)

    fallback_response = build_weather_answer_from_tool(enriched_input, tool_payload)
    if not resolved_city:
        fallback_city = str(tool_payload.get("location") or "unknown")
    else:
        fallback_city = resolved_city
    if not llm_response:
        return fallback_response, tool_payload, fallback_city
    if _looks_like_agent_failure(llm_response):
        return fallback_response, tool_payload, fallback_city
    if _looks_like_unfocused_agent_response(llm_response):
        return fallback_response, tool_payload, fallback_city
    if _looks_too_verbose(llm_response):
        return fallback_response, tool_payload, fallback_city
    if str(tool_payload.get("status") or "").lower() != "ok":
        return fallback_response, tool_payload, fallback_city
    if not any(char.isdigit() for char in llm_response):
        return fallback_response, tool_payload, fallback_city
    return llm_response, tool_payload, fallback_city


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    city = (payload.city or "").strip()
    message = (payload.message or "").strip()
    if not city and not message:
        raise HTTPException(status_code=400, detail="Provide message or city")

    user_input = message or f"What is the weather in {city}?"
    response_text, tool_payload, resolved_city = _chat_logic(user_input=user_input, city_hint=city or None)
    return _build_chat_response(response_text, tool_payload, resolved_city or city or "unknown")


@app.get("/chat", response_model=ChatResponse)
def chat_legacy(city: str = Query(min_length=1, max_length=100)) -> ChatResponse:
    return chat(ChatRequest(city=city, message=f"What is the weather in {city}?"))

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .autonomous_agent import run_autonomous_weather_agent
from .config import FRONTEND_PATH
from .memory_store import get_recent_conversation, get_user_profile, normalize_user_id
from .personas import list_personas
from .schemas import ChatRequest, ChatResponse
from .weather_service import chat_fields_from_tool_payload

app = FastAPI()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "weather-agent",
        "status": "ok",
        "routes": {
            "chat_post": "/chat",
            "chat_get": "/chat?city=Chennai",
            "personas": "/personas",
            "user_profile": "/users/{user_id}/profile",
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


@app.get("/personas")
def personas() -> dict[str, Any]:
    return {"personas": list_personas()}


@app.get("/users/{user_id}/profile")
def user_profile(user_id: str) -> dict[str, Any]:
    safe_user_id = normalize_user_id(user_id)
    return {
        "profile": get_user_profile(safe_user_id),
        "recent_conversation": get_recent_conversation(safe_user_id, limit=10),
    }


def _build_chat_response(
    response_text: str,
    tool_payload: dict[str, Any],
    fallback_city: str,
    user_id: str,
    persona_id: str,
    units: str,
    memory_profile: dict[str, Any] | None,
    include_trace: bool,
    trace: list[dict[str, Any]],
) -> ChatResponse:
    status = str(tool_payload.get("status") or "").lower()
    if status == "ok":
        fields = chat_fields_from_tool_payload(tool_payload, units=units)
        return ChatResponse(
            city=str(fields.get("city") or fallback_city or "unknown"),
            response=response_text,
            user_id=user_id,
            persona_id=persona_id,
            units=str(fields.get("units") or units),
            temperature_unit=str(fields.get("temperature_unit") or ("F" if units == "imperial" else "C")),
            wind_speed_unit=str(fields.get("wind_speed_unit") or ("mph" if units == "imperial" else "m/s")),
            temperature_c=fields.get("temperature_c"),
            temperature=fields.get("temperature"),
            humidity_percent=fields.get("humidity_percent"),
            wind_speed_mps=fields.get("wind_speed_mps"),
            wind_speed=fields.get("wind_speed"),
            forecast=fields.get("forecast") or [],
            agent_trace=trace if include_trace else [],
            memory_profile=memory_profile,
        )
    return ChatResponse(
        city=fallback_city or "unknown",
        response=response_text,
        user_id=user_id,
        persona_id=persona_id,
        units=units,
        temperature_unit="F" if units == "imperial" else "C",
        wind_speed_unit="mph" if units == "imperial" else "m/s",
        forecast=[],
        agent_trace=trace if include_trace else [],
        memory_profile=memory_profile,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    preferences = payload.preferences
    city = (payload.city or "").strip()
    if not city and preferences and isinstance(preferences.city, str):
        city = preferences.city.strip()
    message = (payload.message or "").strip()
    if not city and not message:
        raise HTTPException(status_code=400, detail="Provide message or city")

    safe_user_id = normalize_user_id(payload.user_id)
    user_input = message or f"What is the weather in {city}?"

    preference_updates: dict[str, Any] = {}
    if preferences:
        if isinstance(preferences.units, str) and preferences.units.strip():
            preference_updates["units"] = preferences.units.strip().lower()
        if isinstance(preferences.response_style, str) and preferences.response_style.strip():
            preference_updates["response_style"] = preferences.response_style.strip().lower()
        if isinstance(preferences.city, str) and preferences.city.strip():
            preference_updates["city"] = preferences.city.strip()
    if city:
        preference_updates["city"] = city

    result = run_autonomous_weather_agent(
        user_input=user_input,
        city_hint=city or None,
        user_id=safe_user_id,
        persona_id=payload.persona_id,
        preference_updates=preference_updates,
        remember_memory=bool(payload.remember_memory),
    )
    return _build_chat_response(
        response_text=str(result.get("response_text") or ""),
        tool_payload=result.get("tool_payload") if isinstance(result.get("tool_payload"), dict) else {},
        fallback_city=str(result.get("resolved_city") or city or "unknown"),
        user_id=safe_user_id,
        persona_id=str(result.get("persona_id") or payload.persona_id or "professional"),
        units=str(result.get("units") or "metric"),
        memory_profile=result.get("profile") if isinstance(result.get("profile"), dict) else None,
        include_trace=bool(payload.include_trace),
        trace=result.get("trace") if isinstance(result.get("trace"), list) else [],
    )


@app.get("/chat", response_model=ChatResponse)
def chat_legacy(city: str = Query(min_length=1, max_length=100)) -> ChatResponse:
    return chat(ChatRequest(city=city, message=f"What is the weather in {city}?"))

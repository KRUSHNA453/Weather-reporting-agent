from typing import Any

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    city: str | None = Field(default=None, max_length=100)
    units: str | None = Field(default=None, max_length=20)
    response_style: str | None = Field(default=None, max_length=20)


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, max_length=500)
    city: str | None = Field(default=None, max_length=100)
    user_id: str | None = Field(default="guest", max_length=64)
    persona_id: str | None = Field(default=None, max_length=40)
    preferences: UserPreferences | None = None
    remember_memory: bool | None = None
    include_trace: bool | None = None


class ForecastDay(BaseModel):
    date: str
    temp_min_c: float
    temp_max_c: float
    temp_min: float | None = None
    temp_max: float | None = None
    description: str


class ChatResponse(BaseModel):
    city: str
    response: str
    user_id: str | None = None
    persona_id: str | None = None
    units: str = "metric"
    temperature_unit: str = "C"
    wind_speed_unit: str = "m/s"
    temperature_c: float | None = None
    temperature: float | None = None
    humidity_percent: int | None = None
    wind_speed_mps: float | None = None
    wind_speed: float | None = None
    forecast: list[ForecastDay] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    memory_profile: dict[str, Any] | None = None

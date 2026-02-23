from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, max_length=500)
    city: str | None = Field(default=None, max_length=100)


class ForecastDay(BaseModel):
    date: str
    temp_min_c: float
    temp_max_c: float
    description: str


class ChatResponse(BaseModel):
    city: str
    response: str
    temperature_c: float | None = None
    humidity_percent: int | None = None
    wind_speed_mps: float | None = None
    forecast: list[ForecastDay] = Field(default_factory=list)

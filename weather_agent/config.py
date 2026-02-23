import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPENWEATHER_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

AGENT_MEMORY_DB_PATH = os.getenv("AGENT_MEMORY_DB_PATH")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


DEFAULT_REMEMBER_MEMORY = _env_flag("DEFAULT_REMEMBER_MEMORY", False)
DEFAULT_INCLUDE_TRACE = _env_flag("DEFAULT_INCLUDE_TRACE", False)
# Debug gate: trace is never returned unless this is true.
TRACE_UI_ENABLED = _env_flag("TRACE_UI_ENABLED", False)

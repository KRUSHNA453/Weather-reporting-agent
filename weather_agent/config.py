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

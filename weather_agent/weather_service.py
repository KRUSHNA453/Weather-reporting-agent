import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests

from .config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_API_URL,
    OPENWEATHER_FORECAST_URL,
    OPENWEATHER_ONECALL_URL,
    TRANSIENT_STATUS_CODES,
)

WEATHER_LIVE_DATA_UNAVAILABLE = "Live weather data is temporarily unavailable"

WEATHER_QUERY_MARKERS = (
    "weather",
    "forecast",
    "temperature",
    "rain",
    "humidity",
    "wind",
    "storm",
    "alert",
    "climate",
)

RAIN_MARKERS = ("rain", "drizzle", "shower", "thunderstorm", "storm", "precipitation")
TEMPERATURE_MARKERS = ("temperature", "temp", "hot", "cold", "warm", "cool")
HUMIDITY_MARKERS = ("humidity", "humid")
WIND_MARKERS = ("wind", "breeze", "gust")
STORM_MARKERS = ("storm", "thunderstorm", "cyclone", "hurricane", "tornado")
ALERT_MARKERS = ("alert", "warning", "advisory", "severe")
FORECAST_MARKERS = (
    "forecast",
    "future",
    "upcoming",
    "hourly",
    "daily",
    "weekend",
    "this week",
    "next 3 days",
    "next few days",
)
CLIMATE_MARKERS = ("climate", "condition", "conditions", "overall")

HOURLY_MARKERS = ("hourly", "hour", "next few hours", "next 24 hours")
DAILY_MARKERS = ("daily", "week", "next days", "weekend", "tomorrow")

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MAX_HOURLY_POINTS = 12
MAX_DAILY_POINTS = 5

CITY_IN_TEXT_PATTERN = re.compile(
    r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z\s.'-]{1,80})",
    re.IGNORECASE,
)

TRAILING_NOISE_PATTERN = re.compile(
    r"\b(?:today|tonight|tomorrow|now|please|currently|right now)\b.*$",
    re.IGNORECASE,
)

NON_CITY_QUERY_WORDS = {
    "what",
    "how",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "show",
    "help",
    "me",
    "you",
    "is",
    "are",
    "the",
    "weather",
    "temperature",
    "humidity",
    "wind",
    "forecast",
    "today",
    "tomorrow",
    "hourly",
    "daily",
    "weekend",
    "storm",
    "alert",
    "chance",
    "probability",
    "there",
    "be",
}


def _extract_city_name(raw_city: str) -> str:
    city_name = raw_city.strip()
    if not city_name:
        return ""

    try:
        payload = json.loads(city_name)
        if isinstance(payload, dict):
            candidate = payload.get("city") or payload.get("location")
            if isinstance(candidate, str):
                return candidate.strip("`\"' \n\t")
    except json.JSONDecodeError:
        pass

    if "{" in city_name and "}" in city_name:
        try:
            start = city_name.find("{")
            end = city_name.rfind("}") + 1
            payload = json.loads(city_name[start:end])
            if isinstance(payload, dict):
                candidate = payload.get("city") or payload.get("location")
                if isinstance(candidate, str):
                    city_name = candidate
        except json.JSONDecodeError:
            pass

    match = re.search(r'"(?:city|location)"\s*:\s*"([^"]+)"', city_name)
    if match:
        city_name = match.group(1)

    return city_name.strip("`\"' \n\t")


def _sanitize_city_candidate(candidate: str) -> str:
    value = candidate.strip("`\"' \n\t")
    value = re.split(r"[?!;,]", value, maxsplit=1)[0].strip()
    value = TRAILING_NOISE_PATTERN.sub("", value).strip("`\"' \n\t")
    value = re.sub(r"\s{2,}", " ", value)
    return value


def infer_city_from_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    raw = text.strip()
    if not raw:
        return None

    match = CITY_IN_TEXT_PATTERN.search(raw)
    if match:
        candidate = _sanitize_city_candidate(match.group(1))
        if candidate:
            return candidate

    simple_candidate = _sanitize_city_candidate(raw)
    if re.fullmatch(r"[A-Za-z][A-Za-z\s.'-]{0,80}", simple_candidate or ""):
        words = [w.lower() for w in re.findall(r"[A-Za-z']+", simple_candidate)]
        if not words:
            return None
        if any(word in NON_CITY_QUERY_WORDS for word in words):
            return None
        if 1 <= len(words) <= 4:
            return simple_candidate
    return None


def _request_openweather(city_name: str, endpoint_url: str = OPENWEATHER_API_URL) -> requests.Response | None:
    params = {"q": city_name, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    last_response = None

    for _ in range(2):
        try:
            response = requests.get(endpoint_url, params=params, timeout=10)
        except requests.RequestException:
            continue

        last_response = response
        if response.status_code in TRANSIENT_STATUS_CODES:
            continue
        return response

    return last_response


def _fetch_openweather_payload(city_name: str, endpoint_url: str) -> dict[str, Any] | None:
    response = _request_openweather(city_name, endpoint_url=endpoint_url)
    if response is None or response.status_code != 200:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_city_name(city: str) -> str | None:
    if not isinstance(city, str):
        return None
    city_name = _extract_city_name(city)
    return city_name or None


def _build_weather_details(payload: dict[str, Any], fallback_city: str) -> dict[str, Any] | None:
    try:
        main_data = payload["main"]
        weather_data = payload["weather"][0]
    except (TypeError, KeyError, IndexError):
        return None

    if not isinstance(main_data, dict) or not isinstance(weather_data, dict):
        return None

    temperature = main_data.get("temp")
    humidity = main_data.get("humidity")
    wind_data = payload.get("wind")
    wind_speed = wind_data.get("speed") if isinstance(wind_data, dict) else None
    if not isinstance(temperature, (int, float)):
        return None
    if not isinstance(humidity, (int, float)):
        return None

    description = weather_data.get("description")
    if not isinstance(description, str):
        description = "No description"

    city_display = payload.get("name")
    if not isinstance(city_display, str) or not city_display.strip():
        city_display = fallback_city

    structured_wind = None
    if isinstance(wind_speed, (int, float)):
        structured_wind = round(float(wind_speed), 1)

    return {
        "city": city_display,
        "temperature_c": round(float(temperature), 1),
        "description": description.strip(),
        "humidity_percent": int(humidity),
        "wind_speed_mps": structured_wind,
    }


def _looks_like_weather_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in WEATHER_QUERY_MARKERS)


def _city_is_ambiguous(city_name: str) -> bool:
    lowered = city_name.lower()
    return " or " in lowered or "/" in lowered


def _next_weekday(base_date: date, target_index: int) -> date:
    delta = (target_index - base_date.weekday()) % 7
    if delta == 0:
        delta = 7
    return base_date + timedelta(days=delta)


def _parse_specific_date(text: str) -> date | None:
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if iso_match:
        try:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None

    dmy_match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", text)
    if dmy_match:
        day_value = int(dmy_match.group(1))
        month_value = int(dmy_match.group(2))
        year_value = int(dmy_match.group(3))
        try:
            return date(year_value, month_value, day_value)
        except ValueError:
            return None
    return None


def _extract_time_reference(text: str) -> dict[str, Any]:
    now_date = datetime.now(timezone.utc).date()
    lowered = text.lower()

    granularity = "daily"
    if any(token in lowered for token in HOURLY_MARKERS):
        granularity = "hourly"
    elif any(token in lowered for token in DAILY_MARKERS):
        granularity = "daily"

    explicit_date = _parse_specific_date(lowered)
    if explicit_date:
        return {
            "type": "specific_date",
            "start_date": explicit_date.isoformat(),
            "end_date": explicit_date.isoformat(),
            "granularity": granularity,
            "assumed_today": False,
        }

    if "future" in lowered or "upcoming" in lowered:
        start_target = now_date + timedelta(days=1)
        end_target = start_target + timedelta(days=2)
        return {
            "type": "future_window",
            "start_date": start_target.isoformat(),
            "end_date": end_target.isoformat(),
            "granularity": "daily",
            "assumed_today": False,
        }

    if "tomorrow" in lowered:
        target = now_date + timedelta(days=1)
        return {
            "type": "tomorrow",
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
            "granularity": granularity,
            "assumed_today": False,
        }

    if "weekend" in lowered:
        saturday = _next_weekday(now_date, WEEKDAY_INDEX["saturday"])
        sunday = saturday + timedelta(days=1)
        return {
            "type": "weekend",
            "start_date": saturday.isoformat(),
            "end_date": sunday.isoformat(),
            "granularity": "daily",
            "assumed_today": False,
        }

    for name, idx in WEEKDAY_INDEX.items():
        if name in lowered:
            target = _next_weekday(now_date, idx)
            return {
                "type": "weekday",
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
                "granularity": granularity,
                "assumed_today": False,
                "weekday": name,
            }

    return {
        "type": "today",
        "start_date": now_date.isoformat(),
        "end_date": now_date.isoformat(),
        "granularity": granularity,
        "assumed_today": True,
    }


def _request_openweather_params(endpoint_url: str, params: dict[str, Any]) -> requests.Response | None:
    last_response = None
    for _ in range(2):
        try:
            response = requests.get(endpoint_url, params=params, timeout=10)
        except requests.RequestException:
            continue
        last_response = response
        if response.status_code in TRANSIENT_STATUS_CODES:
            continue
        return response
    return last_response


def _wind_direction_label(degrees: float | int | None) -> str | None:
    if not isinstance(degrees, (int, float)):
        return None
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((float(degrees) % 360) / 45 + 0.5) % 8
    return directions[idx]


def _build_hourly_entries(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    entries = payload.get("list")
    if not isinstance(entries, list):
        return [], 0

    city_info = payload.get("city") if isinstance(payload.get("city"), dict) else {}
    timezone_shift = city_info.get("timezone", 0) if isinstance(city_info, dict) else 0
    if not isinstance(timezone_shift, (int, float)):
        timezone_shift = 0
    timezone_shift = int(timezone_shift)

    hourly: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dt_value = entry.get("dt")
        if not isinstance(dt_value, (int, float)):
            continue

        dt_local = datetime.fromtimestamp(int(dt_value) + timezone_shift, tz=timezone.utc)
        main_data = entry.get("main") if isinstance(entry.get("main"), dict) else {}
        wind_data = entry.get("wind") if isinstance(entry.get("wind"), dict) else {}
        weather_items = entry.get("weather") if isinstance(entry.get("weather"), list) else []
        weather_info = weather_items[0] if weather_items and isinstance(weather_items[0], dict) else {}
        description = str(weather_info.get("description") or "No description").strip()
        pop_value = entry.get("pop")
        pop_percent = None
        if isinstance(pop_value, (int, float)):
            pop_percent = int(round(float(pop_value) * 100))

        wind_deg = wind_data.get("deg")
        hourly.append(
            {
                "date": dt_local.date().isoformat(),
                "time": dt_local.strftime("%H:%M"),
                "local_time": dt_local.strftime("%Y-%m-%d %H:%M"),
                "temperature_c": round(float(main_data.get("temp")), 1)
                if isinstance(main_data.get("temp"), (int, float))
                else None,
                "humidity_percent": int(main_data.get("humidity"))
                if isinstance(main_data.get("humidity"), (int, float))
                else None,
                "wind_speed_mps": round(float(wind_data.get("speed")), 1)
                if isinstance(wind_data.get("speed"), (int, float))
                else None,
                "wind_deg": float(wind_deg) if isinstance(wind_deg, (int, float)) else None,
                "wind_direction": _wind_direction_label(wind_deg),
                "precip_probability_percent": pop_percent,
                "description": description,
                "storm_possible": any(marker in description.lower() for marker in STORM_MARKERS),
            }
        )
    return hourly, timezone_shift


def _build_daily_entries(hourly_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[Any]]] = {}
    for item in hourly_entries:
        date_key = item.get("date")
        if not isinstance(date_key, str):
            continue
        bucket = buckets.setdefault(
            date_key,
            {"temps": [], "humidity": [], "wind": [], "pop": [], "descriptions": [], "wind_dir": []},
        )
        if isinstance(item.get("temperature_c"), (int, float)):
            bucket["temps"].append(float(item["temperature_c"]))
        if isinstance(item.get("humidity_percent"), (int, float)):
            bucket["humidity"].append(int(item["humidity_percent"]))
        if isinstance(item.get("wind_speed_mps"), (int, float)):
            bucket["wind"].append(float(item["wind_speed_mps"]))
        if isinstance(item.get("precip_probability_percent"), (int, float)):
            bucket["pop"].append(int(item["precip_probability_percent"]))
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            bucket["descriptions"].append(description.strip())
        direction = item.get("wind_direction")
        if isinstance(direction, str) and direction:
            bucket["wind_dir"].append(direction)

    daily: list[dict[str, Any]] = []
    for date_key in sorted(buckets.keys()):
        bucket = buckets[date_key]
        temps = bucket["temps"]
        descriptions = bucket["descriptions"]
        daily.append(
            {
                "date": date_key,
                "temp_min_c": round(min(temps), 1) if temps else None,
                "temp_max_c": round(max(temps), 1) if temps else None,
                "humidity_percent": int(round(sum(bucket["humidity"]) / len(bucket["humidity"])))
                if bucket["humidity"]
                else None,
                "wind_speed_mps": round(max(bucket["wind"]), 1) if bucket["wind"] else None,
                "wind_direction": Counter(bucket["wind_dir"]).most_common(1)[0][0] if bucket["wind_dir"] else None,
                "precip_probability_percent": max(bucket["pop"]) if bucket["pop"] else None,
                "description": Counter(descriptions).most_common(1)[0][0] if descriptions else "No description",
            }
        )
    for item in daily:
        item["storm_possible"] = any(marker in str(item.get("description", "")).lower() for marker in STORM_MARKERS)
    return daily


def _filter_entries_by_date(
    items: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    date_key: str = "date",
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in items:
        date_value = item.get(date_key)
        if not isinstance(date_value, str):
            continue
        if start_date <= date_value <= end_date:
            filtered.append(item)
    return filtered


def _extract_alerts(lat: float | None, lon: float | None) -> list[dict[str, Any]]:
    if not OPENWEATHER_API_KEY:
        return []
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return []

    response = _request_openweather_params(
        OPENWEATHER_ONECALL_URL,
        {
            "lat": float(lat),
            "lon": float(lon),
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "exclude": "current,minutely,hourly,daily",
        },
    )
    if response is None or response.status_code != 200:
        return []
    try:
        payload = response.json()
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []

    raw_alerts = payload.get("alerts")
    if not isinstance(raw_alerts, list):
        return []

    alerts: list[dict[str, Any]] = []
    for alert in raw_alerts:
        if not isinstance(alert, dict):
            continue
        start_ts = alert.get("start")
        end_ts = alert.get("end")
        start_iso = (
            datetime.fromtimestamp(int(start_ts), tz=timezone.utc).isoformat()
            if isinstance(start_ts, (int, float))
            else None
        )
        end_iso = (
            datetime.fromtimestamp(int(end_ts), tz=timezone.utc).isoformat()
            if isinstance(end_ts, (int, float))
            else None
        )
        alerts.append(
            {
                "event": str(alert.get("event") or "Weather alert"),
                "start_utc": start_iso,
                "end_utc": end_iso,
                "description": str(alert.get("description") or "").strip(),
            }
        )
    return alerts


def _parse_tool_input(tool_input: str) -> tuple[str, str | None, str | None]:
    raw_input = str(tool_input or "").strip()
    query = raw_input
    location_hint = None
    date_hint = None
    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        location_value = payload.get("location") or payload.get("city")
        if isinstance(location_value, str):
            location_hint = location_value.strip()
        date_value = payload.get("date") or payload.get("time_reference")
        if isinstance(date_value, str):
            date_hint = date_value.strip()
        query_value = payload.get("query") or payload.get("message") or payload.get("question")
        if isinstance(query_value, str) and query_value.strip():
            query = query_value.strip()

    if location_hint and location_hint.lower() not in query.lower():
        query = f"{query} in {location_hint}".strip()
    if date_hint:
        query = f"{query} {date_hint}".strip()
    return query, location_hint, date_hint


def get_weather_forecast(tool_input: str) -> str:
    query, location_hint, _ = _parse_tool_input(tool_input)
    if not _looks_like_weather_query(query):
        query = f"weather {query}".strip()

    if not OPENWEATHER_API_KEY:
        return json.dumps(
            {
                "status": "service_unavailable",
                "message": WEATHER_LIVE_DATA_UNAVAILABLE,
            }
        )

    city_name = _resolve_city_name(location_hint) if location_hint else None
    if not city_name:
        city_name = infer_city_from_text(query)

    if not city_name:
        return json.dumps(
            {
                "status": "needs_location",
                "message": "Please specify the location (city) for the weather request.",
            }
        )

    city_name = _sanitize_city_candidate(city_name)
    if _city_is_ambiguous(city_name):
        return json.dumps(
            {
                "status": "ambiguous_location",
                "message": f"Please clarify the location: '{city_name}'.",
                "location": city_name,
            }
        )

    time_reference = _extract_time_reference(query)
    current_payload = _fetch_openweather_payload(city_name, OPENWEATHER_API_URL)
    forecast_payload = _fetch_openweather_payload(city_name, OPENWEATHER_FORECAST_URL)
    if current_payload is None or forecast_payload is None:
        return json.dumps(
            {
                "status": "service_unavailable",
                "message": WEATHER_LIVE_DATA_UNAVAILABLE,
                "location": city_name,
            }
        )

    current_details = _build_weather_details(current_payload, fallback_city=city_name)
    if current_details is None:
        return json.dumps(
            {
                "status": "service_unavailable",
                "message": WEATHER_LIVE_DATA_UNAVAILABLE,
                "location": city_name,
            }
        )

    hourly_entries, timezone_shift = _build_hourly_entries(forecast_payload)
    daily_entries = _build_daily_entries(hourly_entries)
    start_date = str(time_reference["start_date"])
    end_date = str(time_reference["end_date"])
    selected_hourly = _filter_entries_by_date(hourly_entries, start_date, end_date)
    selected_daily = _filter_entries_by_date(daily_entries, start_date, end_date)

    if not selected_hourly and hourly_entries:
        selected_hourly = hourly_entries[:MAX_HOURLY_POINTS]
    if not selected_daily and daily_entries:
        selected_daily = daily_entries[: min(MAX_DAILY_POINTS, len(daily_entries))]

    pop_values = [
        int(item["precip_probability_percent"])
        for item in selected_hourly
        if isinstance(item.get("precip_probability_percent"), (int, float))
    ]
    if not pop_values:
        pop_values = [
            int(item["precip_probability_percent"])
            for item in selected_daily
            if isinstance(item.get("precip_probability_percent"), (int, float))
        ]
    rain_probability = max(pop_values) if pop_values else None

    storm_periods = [str(item.get("local_time")) for item in selected_hourly if bool(item.get("storm_possible"))]
    if not storm_periods:
        storm_periods = [str(item.get("date")) for item in selected_daily if bool(item.get("storm_possible"))]

    coord = current_payload.get("coord") if isinstance(current_payload.get("coord"), dict) else {}
    lat_value = coord.get("lat") if isinstance(coord.get("lat"), (int, float)) else None
    lon_value = coord.get("lon") if isinstance(coord.get("lon"), (int, float)) else None
    alerts = _extract_alerts(lat_value, lon_value)

    response_payload = {
        "status": "ok",
        "source": "openweather",
        "location": str(current_details.get("city") or city_name),
        "query": query,
        "time_reference": {
            **time_reference,
            "timezone_shift_seconds": timezone_shift,
        },
        "current": {
            "temperature_c": current_details.get("temperature_c"),
            "humidity_percent": current_details.get("humidity_percent"),
            "wind_speed_mps": current_details.get("wind_speed_mps"),
            "wind_deg": float(current_payload.get("wind", {}).get("deg"))
            if isinstance(current_payload.get("wind"), dict)
            and isinstance(current_payload.get("wind", {}).get("deg"), (int, float))
            else None,
            "wind_direction": _wind_direction_label(current_payload.get("wind", {}).get("deg"))
            if isinstance(current_payload.get("wind"), dict)
            else None,
            "description": current_details.get("description"),
        },
        "rain_probability_percent": rain_probability,
        "hourly_forecast": selected_hourly[:MAX_HOURLY_POINTS],
        "daily_forecast": selected_daily[:MAX_DAILY_POINTS],
        "storm_possible": bool(storm_periods),
        "storm_periods": storm_periods[:MAX_HOURLY_POINTS],
        "severe_alerts": alerts,
    }
    return json.dumps(response_payload)


def decode_weather_tool_payload(raw_payload: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(str(raw_payload or ""))
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    text = str(raw_payload or "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _intent_flags(user_input: str) -> dict[str, bool]:
    lowered = user_input.lower()
    return {
        "rain": any(token in lowered for token in RAIN_MARKERS),
        "temperature": any(token in lowered for token in TEMPERATURE_MARKERS),
        "humidity": any(token in lowered for token in HUMIDITY_MARKERS),
        "wind": any(token in lowered for token in WIND_MARKERS),
        "storm": any(token in lowered for token in STORM_MARKERS),
        "alert": any(token in lowered for token in ALERT_MARKERS),
        "forecast": any(token in lowered for token in FORECAST_MARKERS),
        "climate": any(token in lowered for token in CLIMATE_MARKERS),
    }


def _time_scope_label(time_reference: dict[str, Any]) -> str:
    start_date = str(time_reference.get("start_date") or "")
    end_date = str(time_reference.get("end_date") or "")
    time_type = str(time_reference.get("type") or "today")
    if time_type == "today":
        return "today"
    if time_type == "tomorrow":
        return "tomorrow"
    if start_date and end_date and start_date != end_date:
        return f"{start_date} to {end_date}"
    if start_date:
        return start_date
    return "the requested time"


def _rain_statement(probability: int | float | None, location: str, time_scope: str) -> str:
    if not isinstance(probability, (int, float)):
        return f"Rain probability is unavailable for {location} {time_scope}."
    probability_int = int(round(float(probability)))
    if probability_int > 60:
        verdict = "Rain is likely."
    elif 30 <= probability_int <= 60:
        verdict = "There is a chance of rain."
    else:
        verdict = "Rain is unlikely."
    return f"{verdict} Rain probability in {location} for {time_scope}: {probability_int}%."


def build_weather_answer_from_tool(user_input: str, tool_payload: dict[str, Any]) -> str:
    status = str(tool_payload.get("status") or "").lower()
    if status == "needs_location":
        return "Please specify the location (city) for the weather request."
    if status == "ambiguous_location":
        return str(tool_payload.get("message") or "Please clarify the location.")
    if status != "ok":
        return WEATHER_LIVE_DATA_UNAVAILABLE

    location = str(tool_payload.get("location") or "the requested location")
    time_reference = tool_payload.get("time_reference") if isinstance(tool_payload.get("time_reference"), dict) else {}
    time_scope = _time_scope_label(time_reference)
    current = tool_payload.get("current") if isinstance(tool_payload.get("current"), dict) else {}
    daily = tool_payload.get("daily_forecast") if isinstance(tool_payload.get("daily_forecast"), list) else []
    hourly = tool_payload.get("hourly_forecast") if isinstance(tool_payload.get("hourly_forecast"), list) else []
    alerts = tool_payload.get("severe_alerts") if isinstance(tool_payload.get("severe_alerts"), list) else []
    flags = _intent_flags(user_input)

    if not any(flags.values()):
        flags["climate"] = True

    temp_now = current.get("temperature_c")
    humidity_now = current.get("humidity_percent")
    wind_now = current.get("wind_speed_mps")
    wind_dir = current.get("wind_direction")
    condition_now = str(current.get("description") or "condition unavailable")

    day_min = daily[0].get("temp_min_c") if daily and isinstance(daily[0], dict) else None
    day_max = daily[0].get("temp_max_c") if daily and isinstance(daily[0], dict) else None
    rain_probability = tool_payload.get("rain_probability_percent")
    storm_possible = bool(tool_payload.get("storm_possible"))

    first_answer = ""
    primary_metric = ""
    if flags["rain"]:
        first_answer = _rain_statement(rain_probability, location, time_scope)
        primary_metric = "rain"
    elif flags["storm"]:
        if storm_possible:
            first_answer = f"Storm conditions may occur in {location} for {time_scope}."
        else:
            first_answer = f"No storm conditions are indicated in {location} for {time_scope}."
        primary_metric = "storm"
    elif flags["alert"]:
        if alerts:
            first_alert = alerts[0] if isinstance(alerts[0], dict) else {}
            first_answer = f"Severe alert active in {location}: {first_alert.get('event', 'Weather alert')}."
        else:
            first_answer = f"No severe weather alerts are currently reported for {location}."
        primary_metric = "alert"
    elif flags["temperature"]:
        if isinstance(day_min, (int, float)) and isinstance(day_max, (int, float)):
            first_answer = f"Temperature in {location} for {time_scope}: {day_min} C to {day_max} C."
        elif isinstance(temp_now, (int, float)):
            first_answer = f"Current temperature in {location}: {temp_now} C."
        else:
            first_answer = f"Temperature data is unavailable for {location}."
        primary_metric = "temperature"
    elif flags["humidity"]:
        if isinstance(humidity_now, (int, float)):
            first_answer = f"Current humidity in {location}: {int(humidity_now)}%."
        else:
            first_answer = f"Humidity data is unavailable for {location}."
        primary_metric = "humidity"
    elif flags["wind"]:
        if isinstance(wind_now, (int, float)):
            direction = f" {wind_dir}" if isinstance(wind_dir, str) and wind_dir else ""
            first_answer = f"Current wind in {location}: {wind_now} m/s{direction}."
        else:
            first_answer = f"Wind data is unavailable for {location}."
        primary_metric = "wind"
    elif flags["forecast"]:
        granularity = str(time_reference.get("granularity") or "daily")
        if granularity == "hourly" and hourly:
            first_item = hourly[0] if isinstance(hourly[0], dict) else {}
            first_answer = (
                f"Hourly forecast for {location} {time_scope}: "
                f"{first_item.get('time', 'next hour')} {first_item.get('temperature_c', 'N/A')} C, "
                f"{first_item.get('description', 'No description')}."
            )
        elif daily:
            first_item = daily[0] if isinstance(daily[0], dict) else {}
            first_answer = (
                f"Forecast for {location} ({time_scope}): "
                f"{first_item.get('date', 'next day')} {first_item.get('temp_min_c', 'N/A')}-"
                f"{first_item.get('temp_max_c', 'N/A')} C, {first_item.get('description', 'No description')}."
            )
        else:
            first_answer = f"Forecast data is unavailable for {location}."
        primary_metric = "forecast"
    else:
        first_answer = f"Current conditions in {location}: {condition_now}."
        primary_metric = "climate"

    details: list[str] = []
    if flags["temperature"] and primary_metric != "temperature":
        if isinstance(day_min, (int, float)) and isinstance(day_max, (int, float)):
            details.append(f"Temperature range: {day_min} C to {day_max} C.")
        elif isinstance(temp_now, (int, float)):
            details.append(f"Temperature now: {temp_now} C.")
    if flags["humidity"] and primary_metric != "humidity" and isinstance(humidity_now, (int, float)):
        details.append(f"Humidity: {int(humidity_now)}%.")
    if flags["wind"] and primary_metric != "wind" and isinstance(wind_now, (int, float)):
        direction = f" {wind_dir}" if isinstance(wind_dir, str) and wind_dir else ""
        details.append(f"Wind: {wind_now} m/s{direction}.")
    if flags["rain"] and primary_metric != "rain" and isinstance(rain_probability, (int, float)):
        details.append(f"Rain probability: {int(round(float(rain_probability)))}%.")

    if flags["forecast"]:
        granularity = str(time_reference.get("granularity") or "daily")
        if granularity == "hourly" and hourly:
            points = [
                f"{item.get('time')}: {item.get('temperature_c')} C, {item.get('description')}, rain {item.get('precip_probability_percent')}%"
                for item in hourly[:3]
                if isinstance(item, dict)
            ]
            if points:
                details.append("Hourly: " + "; ".join(points) + ".")
        elif daily:
            points = [
                f"{item.get('date')}: {item.get('temp_min_c')}-{item.get('temp_max_c')} C, {item.get('description')}, rain {item.get('precip_probability_percent')}%"
                for item in daily[:3]
                if isinstance(item, dict)
            ]
            if points:
                details.append("Daily: " + "; ".join(points) + ".")

    if flags["storm"] and storm_possible:
        periods = tool_payload.get("storm_periods")
        if isinstance(periods, list) and periods:
            details.append("Storm windows: " + ", ".join(str(item) for item in periods[:3]) + ".")
    if flags["alert"]:
        if alerts:
            alert_summaries = []
            for item in alerts[:2]:
                if not isinstance(item, dict):
                    continue
                event = str(item.get("event") or "Weather alert")
                start_utc = str(item.get("start_utc") or "unknown start")
                alert_summaries.append(f"{event} ({start_utc})")
            if alert_summaries:
                details.append("Alerts: " + "; ".join(alert_summaries) + ".")
        else:
            details.append("Severe alerts: none currently reported.")

    if flags["climate"] and primary_metric != "climate" and not flags["forecast"]:
        details.append(f"Condition: {condition_now}.")

    if not details:
        return first_answer
    return first_answer + " " + " ".join(details)


def chat_fields_from_tool_payload(tool_payload: dict[str, Any]) -> dict[str, Any]:
    current = tool_payload.get("current") if isinstance(tool_payload.get("current"), dict) else {}
    daily = tool_payload.get("daily_forecast") if isinstance(tool_payload.get("daily_forecast"), list) else []
    forecast_days: list[dict[str, Any]] = []
    for item in daily[:MAX_DAILY_POINTS]:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("temp_min_c"), (int, float)) or not isinstance(item.get("temp_max_c"), (int, float)):
            continue
        forecast_days.append(
            {
                "date": str(item.get("date") or "unknown"),
                "temp_min_c": float(item["temp_min_c"]),
                "temp_max_c": float(item["temp_max_c"]),
                "description": str(item.get("description") or "No description"),
            }
        )
    return {
        "city": str(tool_payload.get("location") or "unknown"),
        "temperature_c": current.get("temperature_c") if isinstance(current.get("temperature_c"), (int, float)) else None,
        "humidity_percent": current.get("humidity_percent")
        if isinstance(current.get("humidity_percent"), (int, float))
        else None,
        "wind_speed_mps": current.get("wind_speed_mps")
        if isinstance(current.get("wind_speed_mps"), (int, float))
        else None,
        "forecast": forecast_days,
    }

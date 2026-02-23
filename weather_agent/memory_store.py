import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AGENT_MEMORY_DB_PATH, BASE_DIR

if isinstance(AGENT_MEMORY_DB_PATH, str) and AGENT_MEMORY_DB_PATH.strip():
    DB_PATH = Path(AGENT_MEMORY_DB_PATH.strip())
else:
    DB_PATH = BASE_DIR / "agent_memory.db"
DEFAULT_USER_ID = "guest"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_user_id(user_id: str | None) -> str:
    raw = str(user_id or "").strip()
    if not raw:
        return DEFAULT_USER_ID
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_", ".", ":"))
    cleaned = cleaned[:64].strip()
    return cleaned or DEFAULT_USER_ID


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_memory_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL,
                preferred_city TEXT,
                units TEXT NOT NULL,
                response_style TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                value TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0,
                source_turn TEXT,
                source_message TEXT,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                UNIQUE(user_id, memory_type, normalized_value)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_user_time
            ON conversation_memory (user_id, created_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_fact_user_type
            ON memory_facts (user_id, memory_type, last_used_at DESC)
            """
        )
        conn.commit()


def get_user_profile(user_id: str | None) -> dict[str, Any]:
    uid = normalize_user_id(user_id)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT user_id, persona_id, preferred_city, units, response_style, updated_at
            FROM user_profiles
            WHERE user_id = ?
            """,
            (uid,),
        ).fetchone()

    if row is None:
        return {
            "user_id": uid,
            "persona_id": "professional",
            "preferred_city": None,
            "units": "metric",
            "response_style": "brief",
            "updated_at": None,
        }

    return {
        "user_id": str(row["user_id"]),
        "persona_id": str(row["persona_id"]),
        "preferred_city": row["preferred_city"],
        "units": str(row["units"]),
        "response_style": str(row["response_style"]),
        "updated_at": str(row["updated_at"]),
    }


def upsert_user_profile(
    user_id: str | None,
    persona_id: str | None = None,
    preferred_city: str | None = None,
    units: str | None = None,
    response_style: str | None = None,
) -> dict[str, Any]:
    uid = normalize_user_id(user_id)
    existing = get_user_profile(uid)

    merged_persona = str(persona_id or existing["persona_id"] or "professional").strip().lower()
    merged_units = str(units or existing["units"] or "metric").strip().lower()
    merged_style = str(response_style or existing["response_style"] or "brief").strip().lower()
    merged_city = preferred_city if preferred_city is not None else existing["preferred_city"]
    timestamp = _utc_now()

    if merged_units not in {"metric", "imperial"}:
        merged_units = "metric"
    if merged_style not in {"brief", "balanced", "detailed"}:
        merged_style = "brief"

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, persona_id, preferred_city, units, response_style, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET
                persona_id=excluded.persona_id,
                preferred_city=excluded.preferred_city,
                units=excluded.units,
                response_style=excluded.response_style,
                updated_at=excluded.updated_at
            """,
            (uid, merged_persona, merged_city, merged_units, merged_style, timestamp),
        )
        conn.commit()

    return get_user_profile(uid)


def append_conversation(user_id: str | None, role: str, message: str) -> str | None:
    uid = normalize_user_id(user_id)
    role_value = str(role or "user").strip().lower()
    if role_value not in {"user", "assistant"}:
        role_value = "user"

    payload = str(message or "").strip()
    if not payload:
        return None

    created_at = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_memory (user_id, role, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (uid, role_value, payload[:2000], created_at),
        )
        conn.commit()
    return created_at


def get_recent_conversation(user_id: str | None, limit: int = 8) -> list[dict[str, Any]]:
    uid = normalize_user_id(user_id)
    bounded_limit = max(1, min(int(limit), 50))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, message, created_at
            FROM conversation_memory
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (uid, bounded_limit),
        ).fetchall()

    history = [
        {
            "role": str(row["role"]),
            "message": str(row["message"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]
    history.reverse()
    return history


def upsert_memory_fact(
    user_id: str | None,
    memory_type: str,
    value: str,
    importance: float = 1.0,
    source_turn: str | None = None,
    source_message: str | None = None,
) -> dict[str, Any] | None:
    uid = normalize_user_id(user_id)
    memory_kind = _normalize_text(memory_type)
    fact_value = str(value or "").strip()
    normalized_value = _normalize_text(fact_value)
    if not memory_kind or not normalized_value:
        return None

    bounded_importance = max(0.1, min(float(importance), 5.0))
    timestamp = _utc_now()
    source_turn_value = str(source_turn).strip() if isinstance(source_turn, str) else None
    source_message_value = str(source_message).strip()[:500] if isinstance(source_message, str) else None

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO memory_facts (
                user_id, memory_type, value, normalized_value, importance, source_turn, source_message, created_at, last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, memory_type, normalized_value)
            DO UPDATE SET
                value=excluded.value,
                importance=max(memory_facts.importance, excluded.importance),
                source_turn=excluded.source_turn,
                source_message=excluded.source_message,
                last_used_at=excluded.last_used_at
            """,
            (
                uid,
                memory_kind,
                fact_value[:300],
                normalized_value[:300],
                bounded_importance,
                source_turn_value,
                source_message_value,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()

    facts = get_memory_facts(uid, memory_types=[memory_kind], limit=200)
    for fact in facts:
        if _normalize_text(str(fact.get("value") or "")) == normalized_value:
            return fact
    return None


def get_memory_facts(
    user_id: str | None,
    memory_types: list[str] | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    uid = normalize_user_id(user_id)
    bounded_limit = max(1, min(int(limit), 500))
    with _connect() as conn:
        if isinstance(memory_types, list) and memory_types:
            cleaned = [_normalize_text(item) for item in memory_types if _normalize_text(item)]
            placeholders = ",".join("?" for _ in cleaned)
            rows = conn.execute(
                f"""
                SELECT id, memory_type, value, importance, source_turn, source_message, created_at, last_used_at
                FROM memory_facts
                WHERE user_id = ? AND memory_type IN ({placeholders})
                ORDER BY importance DESC, last_used_at DESC
                LIMIT ?
                """,
                (uid, *cleaned, bounded_limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, memory_type, value, importance, source_turn, source_message, created_at, last_used_at
                FROM memory_facts
                WHERE user_id = ?
                ORDER BY importance DESC, last_used_at DESC
                LIMIT ?
                """,
                (uid, bounded_limit),
            ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "memory_type": str(row["memory_type"]),
            "value": str(row["value"]),
            "importance": float(row["importance"]),
            "source_turn": row["source_turn"],
            "source_message": row["source_message"],
            "created_at": str(row["created_at"]),
            "last_used_at": str(row["last_used_at"]),
        }
        for row in rows
    ]


def _tokenize(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = []
    current = []
    for char in lowered:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return {token for token in tokens if len(token) >= 3}


def _memory_type_boost(memory_type: str) -> float:
    mapping = {
        "preferred_city": 2.0,
        "location_preference": 1.8,
        "activity_interest": 1.4,
        "schedule_pattern": 1.2,
        "weather_preference": 1.0,
    }
    return mapping.get(_normalize_text(memory_type), 1.0)


def retrieve_relevant_memories(user_id: str | None, query: str, limit: int = 6) -> list[dict[str, Any]]:
    uid = normalize_user_id(user_id)
    candidates = get_memory_facts(uid, limit=200)
    if not candidates:
        return []

    query_tokens = _tokenize(query)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        value_tokens = _tokenize(str(item.get("value") or ""))
        source_tokens = _tokenize(str(item.get("source_message") or ""))
        overlap_value = len(query_tokens.intersection(value_tokens))
        overlap_source = len(query_tokens.intersection(source_tokens))
        overlap_score = (overlap_value * 2.0) + overlap_source
        score = overlap_score + float(item.get("importance") or 0.0) + _memory_type_boost(str(item.get("memory_type")))
        if query_tokens and overlap_score <= 0:
            score *= 0.5
        ranked.append((score, item))

    ranked.sort(key=lambda pair: (pair[0], pair[1].get("last_used_at") or ""), reverse=True)
    selected = [item for _, item in ranked[: max(1, min(int(limit), 20))]]
    selected_ids = [int(item["id"]) for item in selected if isinstance(item.get("id"), int)]
    if selected_ids:
        timestamp = _utc_now()
        placeholders = ",".join("?" for _ in selected_ids)
        with _connect() as conn:
            conn.execute(
                f"""
                UPDATE memory_facts
                SET last_used_at = ?
                WHERE id IN ({placeholders})
                """,
                (timestamp, *selected_ids),
            )
            conn.commit()
        for item in selected:
            item["last_used_at"] = timestamp
    return selected


def clear_user_memory(user_id: str | None, clear_profile: bool = False) -> dict[str, int]:
    uid = normalize_user_id(user_id)
    with _connect() as conn:
        conversation_deleted = conn.execute(
            "DELETE FROM conversation_memory WHERE user_id = ?",
            (uid,),
        ).rowcount
        facts_deleted = conn.execute(
            "DELETE FROM memory_facts WHERE user_id = ?",
            (uid,),
        ).rowcount
        profile_deleted = 0
        if clear_profile:
            profile_deleted = conn.execute(
                "DELETE FROM user_profiles WHERE user_id = ?",
                (uid,),
            ).rowcount
        else:
            conn.execute(
                """
                UPDATE user_profiles
                SET preferred_city = NULL,
                    updated_at = ?
                WHERE user_id = ?
                """,
                (_utc_now(), uid),
            )
        conn.commit()

    return {
        "conversation_deleted": int(conversation_deleted or 0),
        "memory_facts_deleted": int(facts_deleted or 0),
        "profile_deleted": int(profile_deleted or 0),
    }


init_memory_db()

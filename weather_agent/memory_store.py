import sqlite3
from datetime import datetime, timezone
from typing import Any

from .config import BASE_DIR

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


def _connect() -> sqlite3.Connection:
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
            CREATE INDEX IF NOT EXISTS idx_conversation_user_time
            ON conversation_memory (user_id, created_at DESC)
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
            "response_style": "balanced",
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
    merged_style = str(response_style or existing["response_style"] or "balanced").strip().lower()
    merged_city = preferred_city if preferred_city is not None else existing["preferred_city"]
    timestamp = _utc_now()

    if merged_units not in {"metric", "imperial"}:
        merged_units = "metric"
    if merged_style not in {"brief", "balanced", "detailed"}:
        merged_style = "balanced"

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


def append_conversation(user_id: str | None, role: str, message: str) -> None:
    uid = normalize_user_id(user_id)
    role_value = str(role or "user").strip().lower()
    if role_value not in {"user", "assistant"}:
        role_value = "user"

    payload = str(message or "").strip()
    if not payload:
        return

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_memory (user_id, role, message, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (uid, role_value, payload[:2000], _utc_now()),
        )
        conn.commit()


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


init_memory_db()

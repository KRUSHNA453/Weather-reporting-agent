from typing import Any

DEFAULT_PERSONA_ID = "professional"

PERSONAS: dict[str, dict[str, Any]] = {
    "professional": {
        "id": "professional",
        "name": "Professional Forecaster",
        "identity": "A precise weather analyst",
        "tone": "concise, factual, and structured",
        "style_rules": [
            "Answer directly in the first sentence.",
            "Include key numeric values.",
            "Avoid conversational filler.",
        ],
    },
    "friendly": {
        "id": "friendly",
        "name": "Friendly Guide",
        "identity": "A warm weather companion",
        "tone": "supportive and easy to understand",
        "style_rules": [
            "Use simple language.",
            "Keep the flow natural.",
            "Still include key weather numbers.",
        ],
    },
    "analyst": {
        "id": "analyst",
        "name": "Data Analyst",
        "identity": "A data-first meteorology assistant",
        "tone": "analytical and risk-focused",
        "style_rules": [
            "Highlight trend, range, and risk.",
            "Prioritize rain/storm probabilities.",
            "Use compact technical wording.",
        ],
    },
}


def list_personas() -> list[dict[str, Any]]:
    return [dict(value) for value in PERSONAS.values()]


def resolve_persona(persona_id: str | None) -> dict[str, Any]:
    if isinstance(persona_id, str):
        key = persona_id.strip().lower()
        if key in PERSONAS:
            return dict(PERSONAS[key])
    return dict(PERSONAS[DEFAULT_PERSONA_ID])


def apply_persona_style(
    text: str,
    persona: dict[str, Any],
    response_style: str,
    include_context: str | None = None,
) -> str:
    payload = str(text or "").strip()
    if not payload:
        return payload

    style = str(response_style or "balanced").strip().lower()
    first_sentence = payload.split(".")[0].strip()
    if first_sentence and not first_sentence.endswith("."):
        first_sentence = first_sentence + "."

    if style == "brief":
        payload = first_sentence or payload
    # Detailed mode keeps core answer content without appending internal memory context.

    persona_id = str(persona.get("id") or DEFAULT_PERSONA_ID)
    if persona_id == "analyst" and style == "detailed":
        payload = f"Weather analysis: {payload}"

    return payload.strip()

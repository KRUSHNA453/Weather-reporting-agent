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
    "teacher": {
        "id": "teacher",
        "name": "Weather Teacher",
        "identity": "An explainer who keeps concepts simple",
        "tone": "clear and educational",
        "style_rules": [
            "Use plain wording.",
            "Add one short practical explanation.",
            "Stay concise.",
        ],
    },
    "safety": {
        "id": "safety",
        "name": "Safety Advisor",
        "identity": "A weather safety-focused advisor",
        "tone": "calm and precaution-oriented",
        "style_rules": [
            "Highlight rain, storms, and alert risk first.",
            "Give one short safety note when needed.",
            "Avoid long paragraphs.",
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
    elif style not in {"balanced", "detailed"}:
        payload = first_sentence or payload

    persona_id = str(persona.get("id") or DEFAULT_PERSONA_ID)
    lowered = payload.lower()
    if persona_id == "friendly":
        payload = f"Friendly update: {payload}"
    elif persona_id == "analyst":
        payload = f"Data view: {payload}"
    elif persona_id == "teacher":
        payload = f"Weather class note: {payload}"
    elif persona_id == "safety":
        if any(token in lowered for token in ("storm", "alert", "rain likely", "chance of rain")):
            payload = f"Safety briefing: {payload} Please carry rain protection and monitor local alerts."
        else:
            payload = f"Safety briefing: {payload}"

    return payload.strip()

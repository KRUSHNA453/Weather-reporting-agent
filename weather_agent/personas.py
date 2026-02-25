from typing import Any

DEFAULT_PERSONA_ID = "friendly"

PERSONAS: dict[str, dict[str, Any]] = {
    "friendly": {
        "id": "friendly",
        "name": "Friendly Guide",
        "identity": "A warm weather companion",
        "tone": "supportive and easy to understand",
        "vocabulary": "plain and approachable",
        "humor_style": "light",
        "risk_stance": "balanced",
        "style_rules": [
            "Use simple language.",
            "Keep the flow natural.",
            "Still include key weather numbers.",
        ],
    },
}


def list_personas() -> list[dict[str, Any]]:
    return [
        {
            "id": str(value.get("id") or ""),
            "name": str(value.get("name") or ""),
            "identity": str(value.get("identity") or ""),
            "tone": str(value.get("tone") or ""),
        }
        for value in PERSONAS.values()
    ]


def resolve_persona(persona_id: str | None) -> dict[str, Any]:
    if isinstance(persona_id, str):
        key = persona_id.strip().lower()
        if key in PERSONAS:
            return dict(PERSONAS[key])
    return dict(PERSONAS[DEFAULT_PERSONA_ID])


def persona_instruction_block(persona: dict[str, Any], response_style: str) -> str:
    style = str(response_style or "brief").strip().lower()
    style_rules = persona.get("style_rules") if isinstance(persona.get("style_rules"), list) else []
    rules_text = "; ".join(str(item) for item in style_rules if str(item).strip())
    lines = [
        "Persona policy:",
        f"- Identity: {str(persona.get('identity') or 'Professional weather assistant')}",
        f"- Tone: {str(persona.get('tone') or 'concise and factual')}",
        f"- Vocabulary: {str(persona.get('vocabulary') or 'plain language')}",
        f"- Humor style: {str(persona.get('humor_style') or 'none')}",
        f"- Risk stance: {str(persona.get('risk_stance') or 'balanced')}",
        f"- Response style: {style}",
    ]
    if rules_text:
        lines.append(f"- Response rules: {rules_text}")
    lines.append("- Do not expose internal reasoning trace to the user.")
    return "\n".join(lines)


def _clip_first_sentence(text: str) -> str:
    payload = str(text or "").strip()
    if not payload:
        return ""

    for index, char in enumerate(payload):
        if char not in ".!?":
            continue
        prev_char = payload[index - 1] if index > 0 else ""
        next_char = payload[index + 1] if index + 1 < len(payload) else ""
        if char == "." and prev_char.isdigit() and next_char.isdigit():
            continue
        return payload[: index + 1].strip() or payload
    return payload


def apply_persona_style(
    text: str,
    persona: dict[str, Any],
    response_style: str,
    include_context: str | None = None,
) -> str:
    payload = str(text or "").strip()
    if not payload:
        return payload

    style = str(response_style or "brief").strip().lower()
    if style == "brief":
        payload = _clip_first_sentence(payload)
    elif style not in {"balanced", "detailed"}:
        payload = _clip_first_sentence(payload)

    
    if payload.startswith("Current conditions in"):
        parts = payload.split(":", 1)
        location_part = parts[0].replace("Current conditions in ", "")
        description_part = parts[1].strip() if len(parts) > 1 else ""
        if description_part.endswith('.'):
            description_part = description_part[:-1]
        payload = f"It looks like the weather in {location_part} is currently showing {description_part}."

    if include_context:
        context_note = str(include_context).strip()
        if context_note and style == "detailed":
            payload = f"{payload} Context used: {context_note}"

    return payload.strip()

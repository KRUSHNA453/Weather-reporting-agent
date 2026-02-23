from typing import Any

DEFAULT_PERSONA_ID = "professional"

PERSONAS: dict[str, dict[str, Any]] = {
    "professional": {
        "id": "professional",
        "name": "Professional Forecaster",
        "identity": "A precise meteorological analyst",
        "tone": "concise, factual, and structured",
        "vocabulary": "meteorological terms when useful, otherwise plain language",
        "humor_style": "none",
        "risk_stance": "balanced",
        "style_rules": [
            "Answer directly in the first sentence.",
            "Include key numeric values.",
            "Keep formatting compact and practical.",
        ],
    },
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
    "analyst": {
        "id": "analyst",
        "name": "Data Analyst",
        "identity": "A highly analytical, slightly dry meteorological assistant",
        "tone": "data-driven and direct",
        "vocabulary": "technical but readable",
        "humor_style": "dry and subtle on extreme weather",
        "risk_stance": "balanced",
        "style_rules": [
            "Prioritize trend, range, and uncertainty.",
            "Quantify rain/storm probabilities.",
            "Call out confidence and limitations briefly.",
        ],
    },
    "teacher": {
        "id": "teacher",
        "name": "Weather Teacher",
        "identity": "An explainer who keeps concepts simple",
        "tone": "clear and educational",
        "vocabulary": "simple with short definitions",
        "humor_style": "minimal",
        "risk_stance": "balanced",
        "style_rules": [
            "Explain one key weather concept in one sentence.",
            "Avoid jargon unless explained.",
            "Stay concise.",
        ],
    },
    "safety": {
        "id": "safety",
        "name": "Safety Advisor",
        "identity": "A weather safety-focused advisor",
        "tone": "calm, cautionary, and practical",
        "vocabulary": "clear and action-oriented",
        "humor_style": "none",
        "risk_stance": "conservative",
        "style_rules": [
            "State risk first when rain, storms, heat, or wind are relevant.",
            "Give one practical safety action.",
            "Avoid long paragraphs.",
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
    first_sentence = text.split(".")[0].strip()
    if first_sentence and not first_sentence.endswith("."):
        first_sentence += "."
    return first_sentence or text


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

    persona_id = str(persona.get("id") or DEFAULT_PERSONA_ID)
    lowered = payload.lower()

    if persona_id == "friendly":
        if not lowered.startswith("friendly update:"):
            payload = f"Friendly update: {payload}"
    elif persona_id == "analyst":
        if not lowered.startswith("analysis:"):
            payload = f"Analysis: {payload}"
    elif persona_id == "teacher":
        if not lowered.startswith("quick explanation:"):
            payload = f"Quick explanation: {payload}"
    elif persona_id == "safety":
        risk_words = ("storm", "alert", "heavy rain", "rain likely", "heatwave", "strong wind")
        if any(token in lowered for token in risk_words):
            payload = f"Safety briefing: {payload} Action: plan shelter, hydration, and backup timing."
        else:
            payload = f"Safety briefing: {payload}"

    if include_context:
        context_note = str(include_context).strip()
        if context_note and style == "detailed":
            payload = f"{payload} Context used: {context_note}"

    return payload.strip()

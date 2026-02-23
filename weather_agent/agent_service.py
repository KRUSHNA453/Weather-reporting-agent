from typing import Any

from .config import HUGGINGFACE_REPO_ID, HUGGINGFACE_TOKEN
from .prompts import WEATHER_BOT_PERSONA
from .weather_service import decode_weather_tool_payload, get_weather_forecast

try:
    from langchain_classic.agents import AgentType, Tool, initialize_agent
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
except Exception:
    AgentType = None
    Tool = None
    initialize_agent = None
    ChatHuggingFace = None
    HuggingFaceEndpoint = None


def create_agent_executor() -> Any | None:
    if not HUGGINGFACE_TOKEN:
        return None
    if not all([AgentType, Tool, initialize_agent, ChatHuggingFace, HuggingFaceEndpoint]):
        return None

    endpoint_llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HUGGINGFACE_TOKEN,
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint_llm)
    weather_tool = Tool(
        name="get_weather_forecast",
        func=get_weather_forecast,
        description=(
            "Mandatory weather tool. Always call this tool once before answering any weather question. "
            "Input must be the full user request so location/time can be extracted. "
            "Output is JSON with current weather, hourly/daily forecast, rain probability, storms, and alerts."
        ),
    )

    return initialize_agent(
        tools=[weather_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_kwargs={"system_message": WEATHER_BOT_PERSONA},
    )


agent_executor = create_agent_executor()


def _clean_response(text: str) -> str:
    return str(text or "").replace("```", "").replace("**", "").strip()


def _compose_input_for_llm(
    user_input: str,
    persona: dict[str, Any] | None,
    response_style: str,
    memory_city: str | None,
) -> str:
    persona_payload = persona or {}
    identity = str(persona_payload.get("identity") or "Professional weather assistant")
    tone = str(persona_payload.get("tone") or "concise and factual")
    style_rules = persona_payload.get("style_rules") if isinstance(persona_payload.get("style_rules"), list) else []
    style_text = "; ".join(str(item) for item in style_rules if str(item).strip())
    memory_hint = f"User preferred city from memory: {memory_city}." if isinstance(memory_city, str) and memory_city.strip() else ""

    instruction_lines = [
        "Persona adaptation instructions:",
        f"- Identity: {identity}",
        f"- Tone: {tone}",
        f"- Response style: {response_style}",
    ]
    if style_text:
        instruction_lines.append(f"- Style rules: {style_text}")
    if memory_hint:
        instruction_lines.append(f"- Memory context: {memory_hint}")

    instruction_lines.append(f"User request: {user_input}")
    return "\n".join(instruction_lines)


def _extract_llm_steps_and_payload(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    steps = result.get("intermediate_steps")
    if not isinstance(steps, list):
        return [], None

    llm_steps: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    for item in steps:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        action, observation = item
        tool_name = str(getattr(action, "tool", "")).strip()
        tool_input = str(getattr(action, "tool_input", "")).strip()
        observation_text = str(observation or "").strip()
        parsed_payload = decode_weather_tool_payload(observation_text) if tool_name == "get_weather_forecast" else None
        if isinstance(parsed_payload, dict):
            final_payload = parsed_payload
            status = str(parsed_payload.get("status") or "").strip().lower()
        else:
            status = "unknown"
        llm_steps.append(
            {
                "tool": tool_name or "unknown",
                "tool_input": tool_input,
                "status": status,
                "observation_preview": observation_text[:220],
            }
        )
    return llm_steps, final_payload


def invoke_llm_weather_agent(
    user_input: str,
    persona: dict[str, Any] | None,
    response_style: str,
    memory_city: str | None = None,
) -> dict[str, Any] | None:
    if agent_executor is None:
        return None

    try:
        llm_input = _compose_input_for_llm(
            user_input=user_input,
            persona=persona,
            response_style=response_style,
            memory_city=memory_city,
        )
        result = agent_executor.invoke({"input": llm_input})
    except Exception as exc:
        return {
            "output": "",
            "tool_payload": None,
            "steps": [],
            "error": str(exc),
        }

    if isinstance(result, dict):
        output_text = _clean_response(str(result.get("output") or ""))
        llm_steps, tool_payload = _extract_llm_steps_and_payload(result)
        return {
            "output": output_text,
            "tool_payload": tool_payload,
            "steps": llm_steps,
        }

    return {
        "output": _clean_response(str(result)),
        "tool_payload": None,
        "steps": [],
    }

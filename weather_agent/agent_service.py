from typing import Any

from .config import HUGGINGFACE_REPO_ID, HUGGINGFACE_TOKEN
from .prompts import WEATHER_BOT_PERSONA
from .weather_service import get_weather_forecast

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

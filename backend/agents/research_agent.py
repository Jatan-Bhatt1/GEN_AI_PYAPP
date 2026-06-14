"""
Research Agent — LangGraph ReAct agent with web search, calculator, weather, and email tools.
Uses the modern langgraph.prebuilt.create_react_agent API (LangChain 1.x compatible).
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from loguru import logger

from backend.config import get_settings
from backend.tools.web_search import web_search_tool
from backend.tools.calculator import calculator_tool
from backend.tools.weather import get_weather_tool
from backend.tools.email_tool import send_email_tool

settings = get_settings()

_SYSTEM_PROMPT = SystemMessage(content=(
    "You are a senior research analyst with access to web search, weather data, "
    "a calculator, and email tools.\n"
    "For every research task:\n"
    "1. Search for the most current information using web_search\n"
    "2. Verify claims and cross-reference sources\n"
    "3. Present findings clearly with source URLs\n"
    "4. Use the calculator for any numerical analysis\n"
    "5. Only send emails if the user explicitly asks you to\n"
    "Always think step-by-step before choosing a tool."
))


def _get_llm():
    """Return LLM based on config."""
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


def create_research_agent():
    """
    Build and return a LangGraph ReAct research agent.

    The agent has access to:
    - web_search_tool: Real-time Tavily web search
    - calculator_tool: Safe math evaluation
    - get_weather_tool: Current weather for any city
    - send_email_tool: Send emails via SMTP

    Returns:
        Compiled LangGraph app. Invoke with:
        agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    """
    llm = _get_llm()

    tools = [
        web_search_tool,
        calculator_tool,
        get_weather_tool,
        send_email_tool,
    ]

    # create_react_agent returns a CompiledStateGraph (NOT AgentExecutor)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )

    logger.info("Research agent created with tools: web_search, calculator, weather, email")
    return agent


def invoke_research_agent(query: str) -> dict:
    """
    Helper to invoke the research agent and return clean output.

    Args:
        query: User's research question

    Returns:
        dict with 'output' (str) and 'messages' (list)
    """
    agent = create_research_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # The last message in the messages list is the final AI response
    messages = result.get("messages", [])
    final_output = messages[-1].content if messages else "No response generated."

    return {
        "output": final_output,
        "messages": [{"role": m.type, "content": m.content} for m in messages],
    }

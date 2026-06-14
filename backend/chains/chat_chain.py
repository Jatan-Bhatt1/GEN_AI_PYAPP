"""
Chat Chain with Memory — The core conversational chain using LangGraph persistence.

LangChain 1.x fully recommends LangGraph's built-in checkpointer for memory
instead of the deprecated RunnableWithMessageHistory.

Architecture (modern approach):
    - Use langgraph.prebuilt.create_react_agent OR a simple StateGraph
    - Use MemorySaver (in-process) or RedisSaver (persistent) as the checkpointer
    - The graph automatically persists state per thread_id (= session_id)

We use a lightweight approach:
    - Build a minimal StateGraph with one LLM node
    - Use InMemorySaver for now (swap to Redis-backed saver for production)
    - This eliminates the RunnableWithMessageHistory deprecation warning entirely

Invoke pattern (same as before):
    chain.invoke(
        {"messages": [HumanMessage(content="Hello!")]},
        config={"configurable": {"thread_id": "session_123"}}
    )
"""

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from loguru import logger

from backend.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────
# GRAPH STATE SCHEMA
# ─────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    """
    State for the chat graph.
    - messages: Auto-merged list of all messages (LangGraph handles the append logic)
    - semantic_context: Optional injected context from semantic memory search
    """
    messages: Annotated[list[BaseMessage], add_messages]
    semantic_context: str


# ─────────────────────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────────────────────

def _get_llm(streaming: bool = False):
    """Return the configured LLM."""
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            streaming=streaming,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=settings.google_api_key,
    )


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

_BASE_SYSTEM = (
    "You are an intelligent Enterprise AI Assistant that helps businesses with "
    "research, data analysis, writing, coding, planning, and general questions.\n\n"
    "Guidelines:\n"
    "- Be precise, professional, and thorough\n"
    "- When citing facts, mention your source if known\n"
    "- If you are uncertain, say so — never fabricate information\n"
    "- Match the user's tone (formal vs. casual)\n"
    "- For code, always use proper markdown code blocks with the language specified"
)


# ─────────────────────────────────────────────────────────────
# GRAPH NODE FUNCTION
# ─────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    """
    The single LLM node in the chat graph.

    Reads all messages from state, prepends a system message (with optional
    semantic context), calls the LLM, and returns the AI response.

    LangGraph automatically appends the new AI message to state['messages'].
    """
    llm = _get_llm(streaming=False)

    # Build the system message, optionally including semantic context
    semantic_ctx = state.get("semantic_context", "")
    system_content = _BASE_SYSTEM
    if semantic_ctx:
        system_content += f"\n\n{semantic_ctx}"

    # Full message list for LLM: [SystemMessage] + all existing messages
    messages_for_llm = [SystemMessage(content=system_content)] + state["messages"]

    response = llm.invoke(messages_for_llm)

    # Return only the new AI message — LangGraph merges it with existing messages
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────

def build_chat_graph():
    """
    Build and compile the minimal chat state graph.

    Graph topology:
        START → chat_node → END

    The MemorySaver checkpointer stores the full message state per thread_id.
    This replaces RunnableWithMessageHistory entirely.

    Returns:
        Compiled LangGraph app with checkpointing enabled.

    Invoke pattern:
        app = build_chat_graph()
        result = app.invoke(
            {
                "messages": [HumanMessage(content="Hello!")],
                "semantic_context": "",
            },
            config={"configurable": {"thread_id": "user_123"}}
        )
        ai_message = result["messages"][-1].content
    """
    graph = StateGraph(ChatState)

    # Add the single LLM node
    graph.add_node("chat", chat_node)

    # Wire: START → chat → END
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # MemorySaver keeps state in-process (sufficient for single-server dev)
    # For production multi-server setups, swap to a Redis or Postgres checkpointer
    checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Chat graph compiled with MemorySaver checkpointer")
    return compiled


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

# Global compiled chat graph — shared across all requests in the process
_chat_app = build_chat_graph()


def invoke_chat(session_id: str, message: str, semantic_context: str = "") -> str:
    """
    Invoke the chat chain synchronously for a given session.

    The graph checkpointer automatically:
    - Loads existing conversation history for this session (thread_id)
    - Saves the new human + AI messages after each invocation

    Args:
        session_id: Unique session identifier (used as thread_id).
        message: The user's input message.
        semantic_context: Optional context string from semantic memory search.

    Returns:
        The AI's response as a plain string.
    """
    result = _chat_app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "semantic_context": semantic_context,
        },
        config={"configurable": {"thread_id": session_id}},
    )
    # Last message in state is always the AI response
    return result["messages"][-1].content


async def astream_chat(session_id: str, message: str, semantic_context: str = ""):
    """
    Async generator that streams the chat response token-by-token.

    Yields string chunks as they arrive from the LLM.

    Args:
        session_id: Unique session identifier.
        message: The user's input message.
        semantic_context: Optional context from semantic memory.

    Yields:
        String chunks of the AI response.
    """
    async for event in _chat_app.astream_events(
        {
            "messages": [HumanMessage(content=message)],
            "semantic_context": semantic_context,
        },
        config={"configurable": {"thread_id": session_id}},
        version="v2",
    ):
        # Stream only the token-level events from the "chat" node
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("name") in ("ChatOpenAI", "ChatGoogleGenerativeAI", "ChatGroq")
        ):
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content


def get_session_state(session_id: str) -> list[dict]:
    """
    Get the full conversation history for a session from the graph checkpointer.

    Args:
        session_id: The session to inspect.

    Returns:
        List of {"role": "human"|"ai", "content": "..."} dicts.
    """
    try:
        state = _chat_app.get_state(
            config={"configurable": {"thread_id": session_id}}
        )
        messages = state.values.get("messages", [])
        result = []
        for msg in messages:
            role = "human" if msg.type == "human" else "ai"
            result.append({"role": role, "content": msg.content})
        return result
    except Exception as e:
        logger.warning(f"Could not get state for session '{session_id}': {e}")
        return []

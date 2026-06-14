"""
Summary Memory — Automatically summarizes long conversations using an LLM.

As a conversation grows, the LLM compresses older messages into a rolling summary.
This prevents token overflow while keeping meaningful context.

LangChain 1.x approach:
  - Store the running summary string in Redis (as a simple key-value)
  - Store the full raw messages in RedisChatMessageHistory
  - Use a dedicated summarization chain to compress messages on demand
  - Inject the summary into the chat prompt as a system message prefix

Design:
  - When message count exceeds SUMMARY_THRESHOLD, we summarize
  - The summary replaces old messages, keeping only the last KEEP_RECENT messages verbatim
  - Summary is stored under Redis key: "summary:<session_id>"
"""

import json
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import redis as redis_client
from loguru import logger

from backend.config import get_settings

settings = get_settings()

# How many messages to keep verbatim after summarizing
KEEP_RECENT = 6
# Summarize when total messages exceed this count
SUMMARY_THRESHOLD = 20


# ─────────────────────────────────────────────────────────────
# SUMMARIZATION LLM — Uses a fast/cheap model
# ─────────────────────────────────────────────────────────────

def _get_summary_llm():
    """Return a fast LLM for summarization (cheaper models preferred)."""
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",       # cheap + fast for summarization
            temperature=0,              # deterministic summaries
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",      # fast and cost-effective
        temperature=0,
        google_api_key=settings.google_api_key,
    )


# ─────────────────────────────────────────────────────────────
# SUMMARIZATION CHAIN
# ─────────────────────────────────────────────────────────────

_SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversation summarizer. Your job is to create a concise, factual summary "
     "of a conversation that preserves all important information, decisions, and context.\n\n"
     "Rules:\n"
     "- Keep it under 200 words\n"
     "- Preserve all specific facts, numbers, names, and conclusions\n"
     "- Write in third person: 'The user asked about X. The assistant explained Y.'\n"
     "- Do NOT add information not present in the conversation\n\n"
     "Previous summary (if any):\n{previous_summary}"
    ),
    ("human",
     "Summarize this conversation:\n\n{conversation}"
    ),
])


def build_summarization_chain():
    """Build and return the summarization chain."""
    llm = _get_summary_llm()
    return _SUMMARIZATION_PROMPT | llm | StrOutputParser()


_summarization_chain = build_summarization_chain()


# ─────────────────────────────────────────────────────────────
# REDIS HELPERS — Store/retrieve summary string
# ─────────────────────────────────────────────────────────────

def _get_redis_client():
    """Return a raw Redis client for simple key-value operations."""
    return redis_client.from_url(settings.redis_url, decode_responses=True)


def _get_summary_key(session_id: str) -> str:
    return f"summary:{session_id}"


def _load_summary(session_id: str) -> str:
    """Load existing summary from Redis. Returns empty string if none."""
    try:
        r = _get_redis_client()
        summary = r.get(_get_summary_key(session_id))
        return summary or ""
    except Exception as e:
        logger.warning(f"Could not load summary for session '{session_id}': {e}")
        return ""


def _save_summary(session_id: str, summary: str) -> None:
    """Persist summary to Redis with 7-day TTL."""
    try:
        r = _get_redis_client()
        r.setex(_get_summary_key(session_id), 86400 * 7, summary)
    except Exception as e:
        logger.warning(f"Could not save summary for session '{session_id}': {e}")


def _delete_summary(session_id: str) -> None:
    """Remove summary from Redis."""
    try:
        r = _get_redis_client()
        r.delete(_get_summary_key(session_id))
    except Exception as e:
        logger.warning(f"Could not delete summary for session '{session_id}': {e}")


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_summary_history(session_id: str) -> RedisChatMessageHistory:
    """
    Get the raw Redis-backed chat history for a session.
    Used internally by the summarizer.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url,
        key_prefix="summary_chat:",
        ttl=86400 * 7,   # 7 days
    )


def summarize_if_needed(session_id: str) -> str:
    """
    Check if the session's message count exceeds SUMMARY_THRESHOLD.
    If so, summarize the older messages and keep only the recent KEEP_RECENT verbatim.

    Call this AFTER saving a new message pair (human + ai).

    Args:
        session_id: The session to potentially summarize.

    Returns:
        The current summary string (may be empty if no summarization happened).

    How it works:
        1. Load all messages for the session
        2. If count <= SUMMARY_THRESHOLD, return existing summary unchanged
        3. Take messages[:-KEEP_RECENT] (the "old" ones)
        4. Format them as a conversation string
        5. Run the summarization chain → new summary
        6. Save new summary to Redis
        7. Keep only messages[-KEEP_RECENT:] in the history
    """
    try:
        history = get_summary_history(session_id)
        messages = history.messages
    except Exception as e:
        logger.debug(f"Redis memory unavailable for summarization: {e}")
        return ""

    if len(messages) <= SUMMARY_THRESHOLD:
        return _load_summary(session_id)

    logger.info(
        f"Session '{session_id}' has {len(messages)} messages — "
        f"summarizing messages 0..{len(messages) - KEEP_RECENT}"
    )

    # Split: old messages to summarize vs. recent to keep verbatim
    old_messages = messages[:-KEEP_RECENT]
    recent_messages = messages[-KEEP_RECENT:]

    # Format old messages as readable conversation text
    conversation_text = "\n".join(
        f"{'Human' if m.type == 'human' else 'AI'}: {m.content}"
        for m in old_messages
    )

    # Load any previously existing summary
    previous_summary = _load_summary(session_id)

    # Run the summarization chain
    try:
        new_summary = _summarization_chain.invoke({
            "previous_summary": previous_summary or "None",
            "conversation": conversation_text,
        })
        logger.info(f"Generated summary for session '{session_id}' ({len(new_summary)} chars)")
    except Exception as e:
        logger.error(f"Summarization failed for session '{session_id}': {e}")
        return previous_summary

    # Save the new summary
    _save_summary(session_id, new_summary)

    # Replace history: clear old messages, re-add only recent ones
    history.clear()
    for msg in recent_messages:
        history.add_message(msg)

    return new_summary


def get_current_summary(session_id: str) -> str:
    """
    Get the current running summary for a session.

    Args:
        session_id: Session to get summary for.

    Returns:
        Summary string, or "No summary generated yet." if none exists.
    """
    summary = _load_summary(session_id)
    return summary if summary else "No summary generated yet."


def clear_summary(session_id: str) -> None:
    """
    Delete the summary AND chat history for a session.

    Args:
        session_id: Session to clear.
    """
    _delete_summary(session_id)
    try:
        history = get_summary_history(session_id)
        history.clear()
        logger.info(f"Cleared summary memory for session: {session_id}")
    except Exception as e:
        logger.debug(f"Redis summary clear failed (Redis down?): {e}")

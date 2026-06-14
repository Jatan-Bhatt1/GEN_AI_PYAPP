"""
Buffer Memory — Short-term sliding-window conversation history backed by Redis.

This module provides the simplest form of memory: store the last N messages
per session in Redis. Messages persist across server restarts.

LangChain 1.x approach:
  - Use RedisChatMessageHistory as the persistent store
  - Use RunnableWithMessageHistory to auto-load/save per request
  - DO NOT use ConversationBufferWindowMemory (deprecated in 1.x)
"""

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from loguru import logger

from backend.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────
# REDIS HISTORY FACTORY
# Used by RunnableWithMessageHistory in the chat chain.
# ─────────────────────────────────────────────────────────────

def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    """
    Factory function: given a session_id, return its Redis-backed chat history.

    This is the function you pass to RunnableWithMessageHistory. It is called
    automatically before each chain invocation to load history, and after to save.

    The history is stored in Redis under key:
        "chat_history:<session_id>"

    Messages expire after 24 hours (TTL = 86400 seconds).

    Args:
        session_id: Unique string identifying a conversation session.
                    Examples: "user_123", "session_abc", "user_42_session_7"

    Returns:
        RedisChatMessageHistory instance with .messages list populated from Redis.

    Example usage in RunnableWithMessageHistory:
        chain_with_memory = RunnableWithMessageHistory(
            base_chain,
            get_redis_history,          # ← this factory
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        # Call with session config:
        chain_with_memory.invoke(
            {"input": "Hello!"},
            config={"configurable": {"session_id": "user_123"}}
        )
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url,
        key_prefix="chat_history:",   # Redis key = "chat_history:<session_id>"
        ttl=86400,                    # messages expire after 24 hours
    )


# ─────────────────────────────────────────────────────────────
# SESSION MANAGEMENT UTILITIES
# ─────────────────────────────────────────────────────────────

def get_session_messages(session_id: str) -> list[dict]:
    """
    Retrieve all messages for a session as a list of dicts.

    Useful for displaying chat history in the frontend.

    Args:
        session_id: Session to retrieve messages for.

    Returns:
        List of {"role": "human"|"ai", "content": "..."} dicts,
        in chronological order.
    """
    result = []
    try:
        history = get_redis_history(session_id)
        for msg in history.messages:
            role = "human" if msg.type == "human" else "ai"
            result.append({"role": role, "content": msg.content})
    except Exception as e:
        logger.debug(f"Redis memory unavailable for session {session_id}: {e}")
    return result


def clear_session(session_id: str) -> None:
    """
    Delete all messages for a session from Redis.

    Args:
        session_id: The session to clear.
    """
    try:
        history = get_redis_history(session_id)
        history.clear()
        logger.info(f"Cleared memory for session: {session_id}")
    except Exception as e:
        logger.debug(f"Redis memory clear failed (Redis down?): {e}")


def trim_session_to_last_n(session_id: str, n: int = 20) -> None:
    """
    Keep only the last N messages in a session, removing older ones.
    This acts as a manual sliding window when needed.

    Args:
        session_id: Session to trim.
        n: Number of messages to keep (default: 20 = 10 exchanges).
    """
    try:
        history = get_redis_history(session_id)
        messages = history.messages

        if len(messages) > n:
            # Clear and re-add only the last N messages
            history.clear()
            for msg in messages[-n:]:
                history.add_message(msg)
            logger.debug(f"Trimmed session '{session_id}' to last {n} messages")
    except Exception as e:
        logger.debug(f"Redis trim failed (Redis down?): {e}")

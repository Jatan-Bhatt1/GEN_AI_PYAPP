"""
Memory Package — Exports all memory modules for easy imports.

Usage:
    from backend.memory import get_redis_history, get_current_summary
    from backend.memory import store_conversation_turn, search_similar_conversations
"""

from backend.memory.buffer import (
    get_redis_history,
    get_session_messages,
    clear_session,
    trim_session_to_last_n,
)

from backend.memory.summary import (
    summarize_if_needed,
    get_current_summary,
    clear_summary,
    get_summary_history,
)

from backend.memory.semantic import (
    store_conversation_turn,
    store_multiple_turns,
    search_similar_conversations,
    format_semantic_context,
    get_session_memory_stats,
    delete_session_memory,
)

__all__ = [
    # Buffer memory
    "get_redis_history",
    "get_session_messages",
    "clear_session",
    "trim_session_to_last_n",
    # Summary memory
    "summarize_if_needed",
    "get_current_summary",
    "clear_summary",
    "get_summary_history",
    # Semantic memory
    "store_conversation_turn",
    "store_multiple_turns",
    "search_similar_conversations",
    "format_semantic_context",
    "get_session_memory_stats",
    "delete_session_memory",
]

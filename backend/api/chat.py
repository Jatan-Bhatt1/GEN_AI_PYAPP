"""
Chat API — Complete REST endpoints for the chat system with memory.

Endpoints:
  POST   /api/chat/           — Send a message, get streaming SSE response
  POST   /api/chat/sync       — Send a message, get full response at once (non-streaming)
  GET    /api/chat/history    — Get conversation history for a session
  GET    /api/chat/summary    — Get the running summary for a session
  GET    /api/chat/semantic   — Search semantic memory for similar past conversations
  DELETE /api/chat/session    — Clear all memory (buffer + summary + semantic) for a session
  GET    /api/chat/sessions   — List all active sessions (via Redis)

All memory is scoped to a session_id that the client provides.
A session_id can be any unique string: user ID, UUID, or composite key.
"""

import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from backend.chains.chat_chain import invoke_chat, astream_chat, get_session_state
from backend.memory.buffer import get_session_messages, clear_session
from backend.memory.summary import (
    summarize_if_needed,
    get_current_summary,
    clear_summary,
)
from backend.memory.semantic import (
    store_conversation_turn,
    search_similar_conversations,
    format_semantic_context,
    get_session_memory_stats,
    delete_session_memory,
)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ─────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message")
    session_id: str = Field(
        default="default_session",
        description="Unique session ID. Use a consistent ID per user/conversation.",
    )
    use_semantic_memory: bool = Field(
        default=False,
        description=(
            "If True, searches past conversations for relevant context and injects it. "
            "Slightly slower but improves responses for follow-up questions."
        ),
    )
    auto_summarize: bool = Field(
        default=True,
        description=(
            "If True, automatically summarizes the session when it grows long (>20 messages)."
        ),
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int


class ClearRequest(BaseModel):
    session_id: str = Field(..., description="Session whose memory to clear")
    clear_buffer: bool = Field(default=True, description="Clear short-term Redis buffer")
    clear_summary: bool = Field(default=True, description="Clear the LLM summary")
    clear_semantic: bool = Field(default=False, description="Clear ChromaDB semantic memory")


# ─────────────────────────────────────────────────────────────
# 1. STREAMING CHAT ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.post("/", summary="Chat with streaming SSE response")
async def chat_stream(request: ChatRequest):
    """
    Send a message and receive a **streaming** response (Server-Sent Events).

    The AI response is streamed token-by-token. Use this for real-time UI updates.

    **Memory behaviour:**
    - Conversation history is automatically loaded from Redis before each call
    - New messages are automatically saved to Redis after each call
    - If `use_semantic_memory=True`, similar past conversations are injected as context
    - If `auto_summarize=True`, the session is compressed when it exceeds 20 messages

    **Frontend usage (JavaScript):**
    ```javascript
    const response = await fetch('/api/chat/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: 'Hello!', session_id: 'user_123'})
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      console.log(decoder.decode(value));  // partial AI response
    }
    ```
    """
    logger.info(
        f"Chat stream request | session={request.session_id} | "
        f"message='{request.message[:60]}...'"
    )

    # Optionally enrich with semantic context from past conversations
    semantic_context = ""
    if request.use_semantic_memory:
        try:
            semantic_context = format_semantic_context(query=request.message, k=3)
        except Exception as e:
            logger.warning(f"Semantic memory lookup failed: {e}")

    async def generate():
        """Async generator that yields text chunks as they stream from the LLM."""
        full_response = []
        try:
            async for chunk in astream_chat(
                session_id=request.session_id,
                message=request.message,
                semantic_context=semantic_context,
            ):
                full_response.append(chunk)
                yield chunk

        except Exception as e:
            logger.error(f"Streaming error for session {request.session_id}: {e}")
            # Fallback: use sync invoke and yield the whole response at once
            try:
                response = await asyncio.to_thread(
                    invoke_chat,
                    session_id=request.session_id,
                    message=request.message,
                    semantic_context=semantic_context,
                )
                full_response.append(response)
                yield response
            except Exception as e2:
                yield f"\n\n[Error: {str(e2)}]"
                return

        # After completion: store in semantic memory + maybe summarize
        ai_response = "".join(full_response)
        _post_process_turn(
            session_id=request.session_id,
            human_message=request.message,
            ai_response=ai_response,
            auto_summarize=request.auto_summarize,
        )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Session-ID": request.session_id,
            "Cache-Control": "no-cache",
        },
    )


# ─────────────────────────────────────────────────────────────
# 2. SYNC CHAT ENDPOINT (non-streaming)
# ─────────────────────────────────────────────────────────────

@router.post("/sync", response_model=ChatResponse, summary="Chat with full JSON response")
async def chat_sync(request: ChatRequest):
    """
    Send a message and receive the **full response** at once (no streaming).

    Use this for:
    - API integrations where streaming isn't needed
    - Testing via Swagger UI (/docs)
    - Simple script-based clients

    Returns the full AI response plus session metadata.
    """
    logger.info(
        f"Chat sync request | session={request.session_id} | "
        f"message='{request.message[:60]}'"
    )

    # Optionally fetch semantic context
    semantic_context = ""
    if request.use_semantic_memory:
        try:
            semantic_context = format_semantic_context(query=request.message, k=3)
        except Exception as e:
            logger.warning(f"Semantic memory lookup failed: {e}")

    try:
        # Use the LangGraph-based invoke (handles its own memory via checkpointer)
        response_text = await asyncio.to_thread(
            invoke_chat,
            session_id=request.session_id,
            message=request.message,
            semantic_context=semantic_context,
        )
    except Exception as e:
        logger.error(f"Chat sync error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    # Post-process: semantic storage + summarization
    _post_process_turn(
        session_id=request.session_id,
        human_message=request.message,
        ai_response=response_text,
        auto_summarize=request.auto_summarize,
    )

    # Count messages from graph state
    graph_messages = get_session_state(request.session_id)
    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        message_count=len(graph_messages),
    )


# ─────────────────────────────────────────────────────────────
# 3. CONVERSATION HISTORY ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.get("/history", summary="Get conversation history for a session")
async def get_history(
    session_id: str = Query(..., description="Session ID to retrieve history for"),
    include_summary: bool = Query(True, description="Include the current summary in the response"),
    include_semantic_stats: bool = Query(False, description="Include semantic memory statistics"),
):
    """
    Retrieve all conversation messages for a session.

    Returns messages in chronological order plus optional metadata.

    Example response:
    ```json
    {
      "session_id": "user_123",
      "messages": [
        {"role": "human", "content": "What is Python?"},
        {"role": "ai", "content": "Python is a programming language..."}
      ],
      "message_count": 2,
      "summary": "The user asked about Python. The AI explained its features.",
      "semantic_stats": {"turn_count": 1}
    }
    ```
    """
    # Pull from graph state (graph checkpointer) — most accurate source
    graph_messages = get_session_state(session_id)
    # Also pull from Redis buffer as a backup / secondary view
    redis_messages = get_session_messages(session_id)

    # Prefer graph state; fall back to Redis if graph state is empty
    messages = graph_messages if graph_messages else redis_messages

    result: dict = {
        "session_id": session_id,
        "messages": messages,
        "message_count": len(messages),
    }

    if include_summary:
        result["summary"] = get_current_summary(session_id)

    if include_semantic_stats:
        result["semantic_stats"] = get_session_memory_stats(session_id)

    return result


# ─────────────────────────────────────────────────────────────
# 4. SUMMARY ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.get("/summary", summary="Get the running LLM summary for a session")
async def get_summary(
    session_id: str = Query(..., description="Session ID"),
    force_regenerate: bool = Query(
        False,
        description="Force a new summarization even if below threshold",
    ),
):
    """
    Get the current running conversation summary for a session.

    The summary is auto-generated when the conversation exceeds 20 messages.
    You can also force regeneration with `force_regenerate=true`.
    """
    if force_regenerate:
        summary = summarize_if_needed(session_id)
        return {
            "session_id": session_id,
            "summary": summary,
            "regenerated": True,
        }

    summary = get_current_summary(session_id)
    return {
        "session_id": session_id,
        "summary": summary,
        "regenerated": False,
    }


# ─────────────────────────────────────────────────────────────
# 5. SEMANTIC MEMORY SEARCH ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.get("/semantic", summary="Search semantic memory for similar past conversations")
async def search_semantic(
    query: str = Query(..., description="The search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    session_id: str | None = Query(
        None,
        description="If provided, only search within this session. Otherwise searches all sessions.",
    ),
    score_threshold: float = Query(
        0.5, ge=0.0, le=1.0, description="Minimum similarity score (0-1)"
    ),
):
    """
    Search the semantic vector memory for conversations similar to the query.

    Useful for:
    - Finding what the AI said about a topic in the past
    - Surfacing relevant past context before starting a new session
    - Debugging what's stored in semantic memory

    Results are ranked by semantic similarity score (1.0 = identical).
    """
    try:
        results = search_similar_conversations(
            query=query,
            k=k,
            session_id=session_id,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
        "session_filter": session_id,
    }


# ─────────────────────────────────────────────────────────────
# 6. CLEAR MEMORY ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.delete("/session", summary="Clear memory for a session")
async def clear_session_memory(request: ClearRequest):
    """
    Clear some or all memory for a session.

    **Memory types:**
    - **Buffer** (Redis): Short-term message history. Cleared by default.
    - **Summary** (Redis): LLM-generated summary. Cleared by default.
    - **Semantic** (ChromaDB): Long-term vector memory. NOT cleared by default
      (set `clear_semantic=true` explicitly).

    After clearing, the next message will start a fresh conversation.
    """
    cleared = []

    if request.clear_buffer:
        clear_session(request.session_id)
        cleared.append("buffer")

    if request.clear_summary:
        clear_summary(request.session_id)
        cleared.append("summary")

    if request.clear_semantic:
        deleted_count = delete_session_memory(request.session_id)
        cleared.append(f"semantic ({deleted_count} vectors deleted)")

    logger.info(f"Cleared memory for session '{request.session_id}': {cleared}")

    return {
        "session_id": request.session_id,
        "cleared": cleared,
        "message": (
            f"Memory cleared for session '{request.session_id}'. "
            f"Next message will start fresh."
        ),
    }


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _post_process_turn(
    session_id: str,
    human_message: str,
    ai_response: str,
    auto_summarize: bool = True,
) -> None:
    """
    Called after each successful chat turn to:
    1. Store the turn in semantic (vector) memory
    2. Optionally trigger summarization if the session is getting long

    Runs in a fire-and-forget pattern — errors are logged but don't affect the response.
    """
    # Store in semantic memory for cross-session retrieval
    try:
        messages = get_session_messages(session_id)
        turn_index = len(messages) // 2  # each turn = 2 messages (human + AI)
        store_conversation_turn(
            session_id=session_id,
            human_message=human_message,
            ai_response=ai_response,
            turn_index=turn_index,
        )
    except Exception as e:
        logger.warning(f"Semantic memory storage failed (non-fatal): {e}")

    # Auto-summarize if requested
    if auto_summarize:
        try:
            summarize_if_needed(session_id)
        except Exception as e:
            logger.warning(f"Auto-summarization failed (non-fatal): {e}")

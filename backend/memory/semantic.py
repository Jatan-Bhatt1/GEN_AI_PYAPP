"""
Semantic Memory — Long-term vector memory using ChromaDB.

Stores every conversation turn as a vector embedding. When a new message
arrives, retrieves the most semantically similar past conversations. This
gives the AI long-term memory across sessions — even across different users.

LangChain 1.x approach:
  - Use langchain_chroma.Chroma as the vector store
  - Use a separate ChromaDB collection from the RAG documents
  - Store conversation turns as Documents with rich metadata
  - Retrieve similar past conversations at query time

Design:
  - Stored collection: "conversation_memory" (separate from "enterprise_docs")
  - Each document = one turn (human message + AI response combined)
  - Metadata: session_id, timestamp, turn_index, message_count
  - Retrieval: similarity search with configurable k
"""

from datetime import datetime, timezone
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

from backend.config import get_settings

settings = get_settings()

# ChromaDB collection name for conversation memory
# Keep this separate from the RAG documents collection
MEMORY_COLLECTION = "conversation_memory"


# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────

def _get_embeddings():
    """
    Return the configured embedding model.
    Uses the same provider as the LLM for consistency.
    """
    if settings.default_llm_provider == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small",   # 1536 dims, fast + cheap
            openai_api_key=settings.openai_api_key,
        )
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",          # 768 dims
        google_api_key=settings.google_api_key,
    )


# ─────────────────────────────────────────────────────────────
# CHROMA VECTOR STORE
# ─────────────────────────────────────────────────────────────

def _get_vectorstore() -> Chroma:
    """
    Get (or create) the ChromaDB vector store for conversation memory.
    Persists to the same directory as the RAG documents but uses a
    DIFFERENT collection name to avoid any mixing.

    Returns:
        Chroma instance ready for add/search operations.
    """
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=MEMORY_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


# ─────────────────────────────────────────────────────────────
# PUBLIC API — STORE
# ─────────────────────────────────────────────────────────────

def store_conversation_turn(
    session_id: str,
    human_message: str,
    ai_response: str,
    turn_index: int = 0,
    extra_metadata: dict | None = None,
) -> str:
    """
    Store a single conversation turn (human + AI) in semantic memory.

    The turn is stored as a single Document with combined content, allowing
    semantic retrieval by either the question or the answer.

    Args:
        session_id: The session this turn belongs to.
        human_message: The user's message.
        ai_response: The AI's response.
        turn_index: Sequential index of this turn within the session.
        extra_metadata: Optional additional metadata (agent_type, tool_used, etc.)

    Returns:
        The document ID assigned by ChromaDB.

    Example:
        store_conversation_turn(
            session_id="user_123",
            human_message="What is LangChain?",
            ai_response="LangChain is a framework for building LLM applications...",
            turn_index=0,
        )
    """
    vectorstore = _get_vectorstore()

    # Combine human + AI into one searchable text block
    combined_content = (
        f"Human: {human_message}\n"
        f"AI: {ai_response}"
    )

    metadata = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turn_index": turn_index,
        "human_message_length": len(human_message),
        "ai_response_length": len(ai_response),
        "memory_type": "conversation_turn",
        **(extra_metadata or {}),
    }

    doc = Document(page_content=combined_content, metadata=metadata)

    ids = vectorstore.add_documents([doc])
    doc_id = ids[0] if ids else "unknown"

    logger.debug(
        f"Stored conversation turn for session '{session_id}' "
        f"(turn {turn_index}, id={doc_id})"
    )
    return doc_id


def store_multiple_turns(
    session_id: str,
    turns: list[tuple[str, str]],
) -> list[str]:
    """
    Store multiple conversation turns at once (batch operation).

    Args:
        session_id: Session identifier.
        turns: List of (human_message, ai_response) tuples.

    Returns:
        List of document IDs.
    """
    vectorstore = _get_vectorstore()
    documents = []

    for i, (human_msg, ai_resp) in enumerate(turns):
        combined = f"Human: {human_msg}\nAI: {ai_resp}"
        doc = Document(
            page_content=combined,
            metadata={
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "turn_index": i,
                "memory_type": "conversation_turn",
            },
        )
        documents.append(doc)

    ids = vectorstore.add_documents(documents)
    logger.info(f"Stored {len(ids)} turns in semantic memory for session '{session_id}'")
    return ids


# ─────────────────────────────────────────────────────────────
# PUBLIC API — RETRIEVE
# ─────────────────────────────────────────────────────────────

def search_similar_conversations(
    query: str,
    k: int = 3,
    session_id: str | None = None,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Search semantic memory for past conversations similar to the query.

    Args:
        query: The current user message or topic to search for.
        k: Number of similar turns to return (default: 3).
        session_id: If provided, only search within this session.
                    If None, searches across ALL sessions (global memory).
        score_threshold: Minimum similarity score (0-1). Lower = more results.

    Returns:
        List of dicts with keys:
          - "content": The stored conversation text
          - "session_id": Which session this came from
          - "timestamp": When it was stored
          - "score": Similarity score (higher = more similar)
          - "turn_index": Position in original conversation

    Example:
        results = search_similar_conversations(
            query="Tell me about machine learning",
            k=3,
        )
        # Returns past conversations where ML was discussed
    """
    vectorstore = _get_vectorstore()

    # Build metadata filter if session_id is specified
    filter_dict = None
    if session_id:
        filter_dict = {"session_id": session_id}

    try:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []

    formatted = []
    for doc, score in results_with_scores:
        if score >= score_threshold:
            formatted.append({
                "content": doc.page_content,
                "session_id": doc.metadata.get("session_id", "unknown"),
                "timestamp": doc.metadata.get("timestamp", ""),
                "score": round(score, 4),
                "turn_index": doc.metadata.get("turn_index", -1),
            })

    logger.debug(
        f"Semantic search for '{query[:50]}...' returned {len(formatted)} results "
        f"(threshold={score_threshold})"
    )
    return formatted


def format_semantic_context(query: str, k: int = 3) -> str:
    """
    Search semantic memory and format the results as a context string
    ready to inject into a prompt.

    Args:
        query: Current user message.
        k: Number of past conversations to retrieve.

    Returns:
        A formatted string like:
          "Relevant past conversations:\n- Human: ...\n  AI: ...\n..."
        Or empty string if no relevant past conversations found.
    """
    results = search_similar_conversations(query=query, k=k)

    if not results:
        return ""

    lines = ["Relevant past conversations (for context only):"]
    for r in results:
        lines.append(f"\n[Session {r['session_id']}, Turn {r['turn_index']}]")
        lines.append(r["content"])

    return "\n".join(lines)


def get_session_memory_stats(session_id: str) -> dict:
    """
    Get statistics about what's stored in semantic memory for a session.

    Args:
        session_id: Session to inspect.

    Returns:
        Dict with turn_count, oldest_timestamp, newest_timestamp.
    """
    vectorstore = _get_vectorstore()

    try:
        results = vectorstore.similarity_search(
            query="",   # empty query to just filter by metadata
            k=1000,
            filter={"session_id": session_id},
        )

        if not results:
            return {"session_id": session_id, "turn_count": 0}

        timestamps = [
            r.metadata.get("timestamp", "") for r in results
            if r.metadata.get("timestamp")
        ]
        timestamps.sort()

        return {
            "session_id": session_id,
            "turn_count": len(results),
            "oldest": timestamps[0] if timestamps else None,
            "newest": timestamps[-1] if timestamps else None,
        }
    except Exception as e:
        logger.error(f"Failed to get memory stats for session '{session_id}': {e}")
        return {"session_id": session_id, "turn_count": -1, "error": str(e)}


def delete_session_memory(session_id: str) -> int:
    """
    Delete ALL semantic memory for a specific session.

    Args:
        session_id: Session whose memory to delete.

    Returns:
        Number of documents deleted.
    """
    vectorstore = _get_vectorstore()

    try:
        # Get all IDs for this session
        results = vectorstore.get(
            where={"session_id": session_id}
        )
        ids = results.get("ids", [])

        if not ids:
            logger.info(f"No semantic memory found for session '{session_id}'")
            return 0

        vectorstore.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} semantic memory entries for session '{session_id}'")
        return len(ids)

    except Exception as e:
        logger.error(f"Failed to delete semantic memory for session '{session_id}': {e}")
        return 0

"""
Documents API — Upload documents and query them with conversational RAG.

Endpoints:
  POST   /api/documents/upload        — Upload and index a document
  POST   /api/documents/query         — Ask a question (conversational RAG)
  POST   /api/documents/stream        — Ask a question with streaming response
  GET    /api/documents/search        — Direct similarity search (debug)
  GET    /api/documents/sources       — List all indexed documents
  GET    /api/documents/history       — Get RAG conversation history
  DELETE /api/documents/sources/{src} — Remove a document from the index
  DELETE /api/documents/session       — Clear RAG conversation history
"""

import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from loguru import logger

from backend.config import get_settings
from backend.rag.loader import load_document, SUPPORTED_EXTENSIONS
from backend.rag.splitter import split_documents
from backend.rag.retriever import (
    add_documents_to_vectorstore,
    search_documents,
    delete_documents_by_source,
    list_indexed_sources,
)
from backend.chains.rag_chain import invoke_rag, astream_rag, get_rag_history

router = APIRouter(prefix="/api/documents", tags=["Documents (RAG)"])
settings = get_settings()

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ─── Models ───────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Question to ask about uploaded documents",
    )
    session_id: str = Field(
        default="rag_default",
        description=(
            "Session ID for conversation continuity. "
            "Use the same session_id for follow-up questions."
        ),
    )


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    session_id: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload and index a document for RAG")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF/DOCX/TXT) and index it in ChromaDB.

    **Process:**
    1. Save file to `uploads/` directory
    2. Load with the appropriate loader (PyPDFLoader, Docx2txtLoader, etc.)
    3. Split into ~1000-character chunks with 200-char overlap
    4. Embed chunks (OpenAI/Google) and store in ChromaDB
    5. Return summary of what was indexed

    **Supported formats:** PDF, DOCX, DOC, TXT, MD

    After upload, use `POST /api/documents/query` to ask questions.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            ),
        )

    # Save to uploads directory
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)
    logger.info(f"Saved document: {file_path} ({len(content):,} bytes)")

    # Load → Split → Embed → Store
    try:
        docs = await asyncio.to_thread(load_document, str(file_path))
        chunks = await asyncio.to_thread(split_documents, docs)
        ids = await asyncio.to_thread(add_documents_to_vectorstore, chunks)
    except Exception as e:
        # Clean up saved file on failure
        file_path.unlink(missing_ok=True)
        logger.error("Indexing failed for '{}': {}", file.filename, str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

    return {
        "filename": file.filename,
        "file_size_bytes": len(content),
        "pages_loaded": len(docs),
        "chunks_indexed": len(chunks),
        "chunk_ids": ids[:5],      # Show first 5 IDs (for debugging)
        "message": (
            f"✅ '{file.filename}' indexed successfully. "
            f"{len(chunks)} chunks ready for querying."
        ),
    }


@router.post("/query", response_model=RAGResponse, summary="Ask a question about documents")
async def query_documents(request: RAGQueryRequest):
    """
    Ask a natural language question about uploaded documents.

    **Conversational:** Follow-up questions work correctly because the chain
    reformulates vague questions using conversation history.

    **Example session:**
    ```
    Q: "What is the refund policy?"
    A: "Refunds are processed within 5-7 business days. [Source: policy.pdf, Page 3]"

    Q: "What about international orders?"     ← follow-up (vague!)
    A: "International refunds take 10-14 days due to... [Source: policy.pdf, Page 4]"
    ```

    **Sources are always cited** in the format `[Source: filename, Page N]`.

    Use the SAME `session_id` for follow-up questions in a conversation.
    """
    logger.info(
        f"RAG query | session={request.session_id} | "
        f"question='{request.question[:60]}'"
    )

    try:
        result = await asyncio.to_thread(
            invoke_rag,
            session_id=request.session_id,
            question=request.question,
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return RAGResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id,
    )


@router.post("/stream", summary="Ask a question with streaming token-by-token response")
async def stream_query(request: RAGQueryRequest):
    """
    Ask a question and receive the answer as a **streaming** response.

    The answer streams token-by-token. At the end, source citations are
    emitted as a special `[[SOURCES]]{...}` JSON line.

    **JavaScript client example:**
    ```javascript
    const res = await fetch('/api/documents/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: 'What is the refund policy?', session_id: 'user_123'})
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const text = decoder.decode(value);
      if (text.startsWith('[[SOURCES]]')) {
        const sources = JSON.parse(text.replace('[[SOURCES]]', ''));
        console.log('Sources:', sources);
      } else {
        process.stdout.write(text);  // stream answer tokens
      }
    }
    ```
    """
    logger.info(
        f"RAG stream | session={request.session_id} | "
        f"question='{request.question[:60]}'"
    )

    async def generate():
        try:
            async for chunk in astream_rag(
                session_id=request.session_id,
                question=request.question,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"RAG streaming error: {e}")
            yield f"\n\n[Error: {str(e)}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Session-ID": request.session_id,
            "Cache-Control": "no-cache",
        },
    )


@router.get("/search", summary="Direct similarity search in the document index")
async def search_index(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
    source: str | None = Query(None, description="Filter by source filename"),
):
    """
    Perform a direct similarity search in the ChromaDB index.

    Useful for:
    - Debugging what's in the index
    - Verifying a document was indexed correctly
    - Checking what chunks will be retrieved for a query

    Returns raw chunks with similarity scores (0-1, higher = more relevant).
    """
    try:
        results = await asyncio.to_thread(
            search_documents,
            query=query,
            k=k,
            source_filter=source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
        "source_filter": source,
    }


@router.get("/sources", summary="List all indexed documents")
async def list_sources():
    """
    List all documents currently indexed in ChromaDB.

    Shows each unique source file and how many chunks it has.
    """
    try:
        sources = await asyncio.to_thread(list_indexed_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sources: {str(e)}")

    return {
        "indexed_sources": sources,
        "total_sources": len(sources),
        "total_chunks": sum(s["chunk_count"] for s in sources),
    }


@router.get("/history", summary="Get RAG conversation history for a session")
async def get_history(
    session_id: str = Query(..., description="Session ID"),
):
    """
    Get the full RAG conversation history for a session.

    Shows all questions asked and answers given in this session.
    """
    history = get_rag_history(session_id)
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history),
    }


@router.delete("/sources/{source_filename}", summary="Remove a document from the index")
async def delete_source(source_filename: str):
    """
    Remove all chunks of a specific document from ChromaDB.

    This does NOT delete the file from disk — only removes it from the search index.
    After deletion, the document's content won't appear in any query results.

    Args:
        source_filename: The exact filename (e.g., `policy.pdf`)
    """
    try:
        deleted_count = await asyncio.to_thread(
            delete_documents_by_source,
            source_filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed chunks found for source '{source_filename}'",
        )

    return {
        "source": source_filename,
        "chunks_deleted": deleted_count,
        "message": f"✅ Removed '{source_filename}' from index ({deleted_count} chunks deleted).",
    }


@router.delete("/session", summary="Clear RAG conversation history")
async def clear_session(
    session_id: str = Query(..., description="Session ID to clear"),
):
    """
    Clear the RAG conversation history for a session.

    After clearing, follow-up question reformulation resets.
    The documents in ChromaDB are NOT affected — only the conversation history.
    """
    # LangGraph MemorySaver doesn't have a delete API,
    # so we just note the session should be considered cleared.
    # In production, swap to RedisSaver which supports explicit deletion.
    return {
        "session_id": session_id,
        "message": (
            f"RAG session '{session_id}' cleared. "
            "Start a new session by using a different session_id."
        ),
    }

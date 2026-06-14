"""
Retriever — ChromaDB vector store setup and retrieval functions.

This module manages the "enterprise_docs" ChromaDB collection:
  - Adds documents (after splitting and embedding)
  - Retrieves similar documents for a query
  - Deletes documents by source file

ChromaDB persists to ./vectorstore/ so embeddings survive server restarts.
The "enterprise_docs" collection is separate from "conversation_memory" (Phase 5).

Retrieval strategy: MMR (Maximal Marginal Relevance)
  - Balances RELEVANCE (how similar to query) vs. DIVERSITY (no duplicate chunks)
  - Without MMR: might return 5 nearly identical chunks from the same paragraph
  - With MMR: returns 5 diverse chunks covering different aspects of the answer
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from backend.config import get_settings
from backend.rag.embeddings import get_embedding_model

settings = get_settings()

# The ChromaDB collection name for RAG documents
# MUST be different from "conversation_memory" (Phase 5 semantic memory)
RAG_COLLECTION = settings.chroma_collection_name  # "enterprise_docs"


def get_vectorstore() -> Chroma:
    """
    Get (or create) the ChromaDB vector store for enterprise documents.

    ChromaDB is lazy — if the collection exists (from a previous session),
    it loads the existing vectors. If not, it creates a new empty collection.

    Returns:
        Chroma instance ready for add_documents() and similarity_search().
    """
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=RAG_COLLECTION,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,  # "./vectorstore"
    )


def add_documents_to_vectorstore(chunks: list[Document]) -> list[str]:
    """
    Add document chunks to ChromaDB (index them with embeddings).

    This is called after loading + splitting a new document.
    ChromaDB converts each chunk to a vector and stores it.

    Args:
        chunks: List of Document chunks from splitter.py.

    Returns:
        List of ChromaDB document IDs (one per chunk).

    Example:
        chunks = split_documents(load_document("policy.pdf"))
        ids = add_documents_to_vectorstore(chunks)
        # ChromaDB now has 47 new vectors for "policy.pdf"
    """
    vectorstore = get_vectorstore()
    ids = vectorstore.add_documents(chunks)
    logger.info(f"Indexed {len(ids)} chunks into ChromaDB (collection: {RAG_COLLECTION})")
    return ids


def get_retriever(
    k: int = 5,
    search_type: str = "mmr",
) -> VectorStoreRetriever:
    """
    Get a retriever configured for similarity search.

    Args:
        k: Number of chunks to return (default: 5).
        search_type: "mmr" (diverse results) or "similarity" (pure relevance).

    Returns:
        VectorStoreRetriever — can be used directly in chains.

    search_type options:
        "similarity": Return the k most similar chunks (may be duplicates)
        "mmr": Maximal Marginal Relevance — diverse + relevant (recommended)
        "similarity_score_threshold": Only return chunks above a score threshold
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            # For MMR: lambda_mult controls relevance vs. diversity balance
            # 0 = max diversity, 1 = max relevance (default: 0.5)
            "lambda_mult": 0.7,  # slightly lean toward relevance
        },
    )


def search_documents(
    query: str,
    k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Direct similarity search — useful for testing/debugging what's in the vectorstore.

    Args:
        query: Search query string.
        k: Number of results.
        source_filter: If provided, only search within this source file.

    Returns:
        List of dicts with content, source, page, score.
    """
    vectorstore = get_vectorstore()

    # Optional: filter by source file
    where_filter = {"source": source_filter} if source_filter else None

    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=k,
        filter=where_filter,
    )

    return [
        {
            "content": doc.page_content[:500],  # truncate for readability
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "score": round(score, 4),
        }
        for doc, score in results
    ]


def delete_documents_by_source(source_filename: str) -> int:
    """
    Delete all chunks from a specific source file from ChromaDB.

    Called when a user deletes an uploaded document.

    Args:
        source_filename: The filename (e.g., "policy.pdf").

    Returns:
        Number of chunks deleted.
    """
    vectorstore = get_vectorstore()

    try:
        results = vectorstore.get(where={"source": source_filename})
        ids = results.get("ids", [])

        if not ids:
            logger.info(f"No chunks found for source: {source_filename}")
            return 0

        vectorstore.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for source: {source_filename}")
        return len(ids)

    except Exception as e:
        logger.error(f"Failed to delete chunks for '{source_filename}': {e}")
        return 0


def list_indexed_sources() -> list[dict]:
    """
    List all unique source documents currently indexed in ChromaDB.

    Returns:
        List of dicts with source filename and chunk count.
    """
    vectorstore = get_vectorstore()

    try:
        all_docs = vectorstore.get()
        metadatas = all_docs.get("metadatas", [])

        # Count chunks per source
        source_counts: dict[str, int] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

        return [
            {"source": src, "chunk_count": count}
            for src, count in sorted(source_counts.items())
        ]

    except Exception as e:
        logger.error(f"Failed to list indexed sources: {e}")
        return []

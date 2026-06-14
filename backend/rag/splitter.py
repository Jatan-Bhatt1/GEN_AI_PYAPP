"""
Document Splitter — Split large documents into overlapping chunks.

Why split?
  - LLMs have context window limits (can't process 100-page PDFs at once)
  - Embedding a huge document loses granularity
  - Small, focused chunks give better retrieval accuracy

Why overlap?
  - If a sentence spans the boundary of two chunks,
    overlap ensures both chunks contain it
  - Prevents losing context at chunk edges

LangChain 1.x approach:
  - RecursiveCharacterTextSplitter: splits by \\n\\n, \\n, space, char (in order)
    This preserves paragraph boundaries when possible.
  - DO NOT use CharacterTextSplitter alone — it ignores paragraph boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger


# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000       # Each chunk is ~1000 characters
DEFAULT_CHUNK_OVERLAP = 200     # Adjacent chunks share 200 characters


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter which tries to split at:
      1. Double newlines (\\n\\n) — paragraph boundaries (preferred)
      2. Single newlines (\\n) — line boundaries
      3. Spaces — word boundaries
      4. Characters — last resort

    This hierarchy preserves natural language structure.

    Args:
        documents: List of Document objects (from loader.py).
        chunk_size: Maximum characters per chunk (default: 1000).
        chunk_overlap: Characters shared between adjacent chunks (default: 200).

    Returns:
        List of smaller Document chunks. Each chunk inherits the
        metadata from its parent document (source, page, etc.) plus
        a "chunk_index" field.

    Example:
        # 10-page PDF → 47 chunks
        docs = load_document("policy.pdf")    # 10 Documents
        chunks = split_documents(docs)        # 47 Documents (smaller)
        chunks[0].metadata
        # {"source": "policy.pdf", "page": 0, "chunk_index": 0}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # separators tried in order (paragraph → sentence → word → char)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to each chunk's metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict | None = None,
) -> list[Document]:
    """
    Split a plain text string into Document chunks.

    Useful when you have raw text (not a file) that you want to add to the RAG system.

    Args:
        text: The raw text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.
        metadata: Optional metadata to attach to all chunks.

    Returns:
        List of Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[metadata or {}],
    )
    return chunks

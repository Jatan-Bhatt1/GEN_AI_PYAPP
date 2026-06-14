"""
Document Loader — Load files from disk into LangChain Document objects.

Supports: PDF, DOCX, TXT, MD
Each Document has:
  - page_content: the text
  - metadata: {"source": "filename.pdf", "page": 3, "file_type": "pdf"}

LangChain 1.x approach:
  - PyPDFLoader for PDFs (splits by page automatically)
  - Docx2txtLoader for Word documents
  - TextLoader for plain text
  - UnstructuredFileLoader as universal fallback
"""

import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document
from loguru import logger


# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".txt": "text",
    ".md": "text",
}


def load_document(file_path: str) -> list[Document]:
    """
    Load a document from disk and return a list of LangChain Document objects.

    Each Document has:
      - page_content: the actual text content
      - metadata: source file info (source, page, file_type)

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of Documents. PDF returns one doc per page.
        DOCX/TXT return a single doc with all text.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file doesn't exist.

    Example:
        docs = load_document("uploads/policy.pdf")
        # docs[0].page_content = "Refund Policy\n\nAll refunds..."
        # docs[0].metadata = {"source": "policy.pdf", "page": 0}
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    file_type = SUPPORTED_EXTENSIONS[ext]
    logger.info(f"Loading {file_type} file: {file_path.name}")

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            # PyPDFLoader splits by page, adds metadata["page"] automatically
            docs = loader.load()

        elif ext in {".docx", ".doc"}:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()

        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()

        else:
            # Universal fallback (requires unstructured library)
            loader = UnstructuredFileLoader(str(file_path))
            docs = loader.load()

    except Exception as e:
        logger.error(f"Failed to load '{file_path.name}': {e}")
        raise

    # Enrich metadata with our standard fields
    for doc in docs:
        doc.metadata.setdefault("source", file_path.name)
        doc.metadata["file_type"] = file_type
        doc.metadata["file_path"] = str(file_path)

    logger.info(f"Loaded '{file_path.name}': {len(docs)} page(s) / sections")
    return docs


def load_documents_from_directory(directory: str) -> list[Document]:
    """
    Load ALL supported documents from a directory.

    Args:
        directory: Path to directory containing documents.

    Returns:
        Combined list of Documents from all files.
    """
    directory = Path(directory)
    all_docs = []

    for file_path in directory.iterdir():
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping '{file_path.name}': {e}")

    logger.info(f"Loaded {len(all_docs)} documents from '{directory}'")
    return all_docs

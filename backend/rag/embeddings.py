"""
Embeddings — Convert text into vector representations for ChromaDB.

An embedding is a list of numbers (a vector) that captures the MEANING of text.
Similar texts → similar vectors → close together in vector space.

This is what enables semantic search:
  "refund policy" and "money back guarantee" have similar vectors
  → both returned when searching for "refund" topic

LangChain 1.x approach:
  - OpenAIEmbeddings: text-embedding-3-small (1536 dims, fast+cheap)
  - GoogleGenerativeAIEmbeddings: embedding-001 (768 dims)
  Both implement the same Embeddings interface, so they're interchangeable.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def get_embedding_model() -> Embeddings:
    """
    Return the configured embedding model.

    Embeddings are used:
    1. At INDEXING time: convert document chunks → vectors → store in ChromaDB
    2. At QUERY time: convert user query → vector → find similar vectors in ChromaDB

    IMPORTANT: Use the SAME model for both indexing and querying.
    Mixing models (e.g., OpenAI at index, Google at query) gives WRONG results
    because the vector spaces are different.

    Returns:
        Embeddings instance ready to use with Chroma.

    Model details:
        OpenAI text-embedding-3-small:
          - 1536 dimensions
          - ~$0.02 per 1M tokens (very cheap)
          - Excellent quality for English + multilingual
          - Best choice for production

        Google embedding-001:
          - 768 dimensions
          - Free tier available
          - Good quality, good for multilingual
    """
    if settings.default_llm_provider == "openai":
        logger.debug("Using OpenAI text-embedding-3-small")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )

    elif settings.default_llm_provider == "groq":
        logger.debug("Using HuggingFace local embeddings for Groq (all-MiniLM-L6-v2)")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    class PatchedGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Workaround for langchain-google-genai bug with the new google-genai SDK
            # where batching returns a single embedding instead of one per text.
            return [self.embed_query(text) for text in texts]

    logger.debug("Using Google GenerativeAI embeddings (Patched)")
    return PatchedGoogleEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=settings.google_api_key,
    )

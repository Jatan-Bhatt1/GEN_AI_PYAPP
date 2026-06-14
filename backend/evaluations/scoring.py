"""
Answer Quality Scorer — LLM-as-Judge quality evaluation.

Rates answers on three dimensions (each 1-5 scale):
  1. Relevance: Does the answer actually address the question?
  2. Completeness: Is the answer thorough enough?
  3. Accuracy: Is the information correct based on the context?

This produces an overall quality_score = average of the three dimensions.

When to use:
  - After RAG queries to check if answers are good
  - As a quality gate before showing responses to users
  - For continuous monitoring of answer quality over time
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def _get_scorer_llm():
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert evaluator rating the quality of AI answers.\n\n"
     "Rate the ANSWER against the QUESTION and CONTEXT on these dimensions:\n\n"
     "1. **Relevance** (1-5): Does the answer directly address what was asked?\n"
     "   5 = perfectly on-topic | 3 = partially addresses | 1 = completely off-topic\n\n"
     "2. **Completeness** (1-5): Is the answer thorough and does it cover all aspects?\n"
     "   5 = comprehensive | 3 = covers main points | 1 = very incomplete\n\n"
     "3. **Accuracy** (1-5): Is the information correct based on the context?\n"
     "   5 = fully accurate | 3 = mostly accurate | 1 = major inaccuracies\n\n"
     "Return ONLY a JSON object:\n"
     "{{\n"
     "  \"relevance\": 1-5,\n"
     "  \"completeness\": 1-5,\n"
     "  \"accuracy\": 1-5,\n"
     "  \"quality_score\": average_of_three_dimensions,\n"
     "  \"strengths\": [\"what the answer did well\"],\n"
     "  \"weaknesses\": [\"what the answer could improve\"],\n"
     "  \"suggestion\": \"one-sentence improvement suggestion\"\n"
     "}}"
    ),
    ("human",
     "QUESTION: {question}\n\n"
     "CONTEXT (source documents):\n{context}\n\n"
     "ANSWER TO EVALUATE:\n{answer}\n\n"
     "Rate this answer. Return JSON only."
    ),
])


def score_answer(question: str, answer: str, context: str = "") -> dict:
    """
    Score an answer on relevance, completeness, and accuracy.

    Args:
        question: The original user question.
        answer: The AI-generated answer to evaluate.
        context: Source context (for RAG answers). Empty string for chat answers.

    Returns:
        Dict with:
          - relevance (int): 1-5
          - completeness (int): 1-5
          - accuracy (int): 1-5
          - quality_score (float): Average of the three dimensions (1.0-5.0)
          - strengths (list): What the answer did well
          - weaknesses (list): What could be improved
          - suggestion (str): One-sentence improvement suggestion

    Example:
        result = score_answer(
            question="What is the refund policy?",
            answer="Refunds take 5-7 business days.",
            context="Standard refunds are processed within 5-7 business days."
        )
        # result = {"relevance": 5, "completeness": 3, "accuracy": 5,
        #           "quality_score": 4.33, "weaknesses": ["Could mention fees"]}
    """
    llm = _get_scorer_llm()
    chain = _SCORING_PROMPT | llm

    try:
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:3000] if context else "No context provided (direct chat).",
        })

        raw = response.content.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        # Compute quality_score if not provided or incorrect
        dimensions = [
            result.get("relevance", 3),
            result.get("completeness", 3),
            result.get("accuracy", 3),
        ]
        result["quality_score"] = round(sum(dimensions) / len(dimensions), 2)
        result.setdefault("strengths", [])
        result.setdefault("weaknesses", [])
        result.setdefault("suggestion", "No suggestion.")

        logger.info(
            f"Quality score: {result['quality_score']}/5.0 | "
            f"relevance={result['relevance']} | "
            f"completeness={result['completeness']} | "
            f"accuracy={result['accuracy']}"
        )
        return result

    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return {
            "relevance": 0,
            "completeness": 0,
            "accuracy": 0,
            "quality_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "suggestion": f"Scoring error: {str(e)}",
            "error": str(e),
        }

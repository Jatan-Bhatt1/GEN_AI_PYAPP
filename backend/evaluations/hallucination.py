"""
Hallucination Checker — LLM-as-Judge pattern.

"LLM-as-Judge" means using a SECOND LLM call to evaluate the output of a FIRST LLM call.

This is the gold standard for RAG hallucination detection:
  1. User asks: "What is the refund policy?"
  2. RAG retrieves 5 chunks from ChromaDB
  3. LLM generates an answer using those chunks
  4. Hallucination checker: "Is THIS answer supported by THESE chunks?"
     → Returns: {grounded: True/False, confidence: 0.0-1.0, issues: [...]}

Why not just check the answer ourselves?
  Because detecting hallucinations is a hard NLP problem.
  LLMs are surprisingly good at judging their own kind.

The judge prompt is carefully designed to:
  - Ask for STRUCTURED output (JSON) — not just "yes/no"
  - Provide specific criteria for what counts as grounded
  - Ask for quotes from the context that support/contradict the answer
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from backend.config import get_settings

settings = get_settings()


def _get_judge_llm():
    """
    Return a low-temperature LLM for judging.
    Temperature=0 for maximum consistency in evaluations.
    Using a DIFFERENT model than the main LLM is ideal (reduces self-evaluation bias).
    """
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=settings.google_api_key,
    )


_HALLUCINATION_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert fact-checker evaluating whether an AI answer is grounded "
     "in the provided source context.\n\n"
     "Your task: Determine if every factual claim in the ANSWER is directly supported "
     "by the CONTEXT. An answer is NOT grounded if it:\n"
     "- Makes claims not present in the context\n"
     "- Contradicts information in the context\n"
     "- Adds specific details (numbers, dates, names) not in the context\n"
     "- Makes logical inferences that go beyond what the context states\n\n"
     "An answer IS grounded if:\n"
     "- All factual claims can be directly traced to specific passages in the context\n"
     "- It accurately summarizes or paraphrases context content\n"
     "- It correctly states when information is not available in the context\n\n"
     "Return ONLY a JSON object:\n"
     "{{\n"
     "  \"grounded\": true | false,\n"
     "  \"confidence\": 0.0-1.0,\n"
     "  \"issues\": [\"list of specific hallucinations if grounded=false, else []\"],\n"
     "  \"supporting_quotes\": [\"quotes from context that support the answer\"],\n"
     "  \"verdict\": \"one sentence summary\"\n"
     "}}"
    ),
    ("human",
     "CONTEXT (source documents used to generate the answer):\n"
     "{context}\n\n"
     "ANSWER TO EVALUATE:\n"
     "{answer}\n\n"
     "Is this answer grounded in the context? Return JSON only."
    ),
])


def check_hallucination(answer: str, context: str) -> dict:
    """
    Check if an answer is grounded in the provided context.

    This is the LLM-as-Judge hallucination check:
    - Send the answer + the source context to a judge LLM
    - Judge determines if every claim in the answer is backed by context
    - Returns structured verdict with confidence score

    Args:
        answer: The AI-generated answer to evaluate.
        context: The source documents/chunks that were used to generate the answer.
                 (Typically the concatenated text of retrieved ChromaDB chunks)

    Returns:
        Dict with:
          - grounded (bool): True if answer is factually grounded in context
          - confidence (float): 0.0-1.0 confidence in the judgment
          - issues (list): Specific hallucination issues found (empty if grounded)
          - supporting_quotes (list): Context quotes that support the answer
          - verdict (str): One-sentence summary
          - error (str): Error message if evaluation failed

    Example:
        result = check_hallucination(
            answer="Refunds take 5-7 business days.",
            context="Section 1: Standard refunds are processed within 5-7 business days."
        )
        # result = {"grounded": True, "confidence": 0.98, "issues": [], ...}
    """
    llm = _get_judge_llm()
    chain = _HALLUCINATION_JUDGE_PROMPT | llm

    try:
        response = chain.invoke({"answer": answer, "context": context[:4000]})
        raw = response.content.strip()

        # Parse JSON (handle markdown code fences)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        # Ensure all expected fields exist
        result.setdefault("grounded", False)
        result.setdefault("confidence", 0.0)
        result.setdefault("issues", [])
        result.setdefault("supporting_quotes", [])
        result.setdefault("verdict", "Unable to determine")

        logger.info(
            f"Hallucination check: grounded={result['grounded']} | "
            f"confidence={result['confidence']}"
        )
        return result

    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        return {
            "grounded": None,   # None = evaluation failed (not same as False)
            "confidence": 0.0,
            "issues": [],
            "supporting_quotes": [],
            "verdict": f"Evaluation error: {str(e)}",
            "error": str(e),
        }


def batch_check_hallucinations(
    qa_pairs: list[dict],
) -> list[dict]:
    """
    Check hallucinations for a batch of question-answer-context triplets.

    Args:
        qa_pairs: List of dicts, each with keys:
                  - question (str)
                  - answer (str)
                  - context (str)

    Returns:
        List of hallucination check results, one per qa_pair.
    """
    results = []
    for pair in qa_pairs:
        result = check_hallucination(
            answer=pair.get("answer", ""),
            context=pair.get("context", ""),
        )
        result["question"] = pair.get("question", "")
        results.append(result)

    grounded_count = sum(1 for r in results if r.get("grounded") is True)
    logger.info(
        f"Batch hallucination check: {grounded_count}/{len(results)} grounded"
    )
    return results

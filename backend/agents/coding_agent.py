"""
Coding Agent — Specialized LLM chain for code generation, explanation, and debugging.
Uses a rich system prompt tuned for software engineering tasks.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from loguru import logger

from backend.config import get_settings
from backend.prompts.templates import coding_prompt

settings = get_settings()

_CODING_SYSTEM = """You are a world-class software engineer and coding assistant.

When given a coding task, you MUST:
1. **Understand** the requirements fully — ask for clarification if ambiguous
2. **Plan** your approach before writing code (brief 2-3 line plan)
3. **Write** clean, well-commented, production-quality code
4. **Explain** key design decisions and any trade-offs
5. **Test** — include example inputs/outputs or unit test snippets
6. **Warn** about edge cases, security issues, or performance concerns

Code formatting rules:
- Always wrap code in ```language code blocks
- Include imports at the top
- Follow PEP 8 for Python, standard conventions for other languages
- Add docstrings/JSDoc comments to all functions

You support all major languages: Python, JavaScript, TypeScript, Java, Go, Rust, SQL, Bash, etc."""


def _get_llm():
    if settings.default_llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,           # slightly creative but mostly deterministic
            openai_api_key=settings.openai_api_key,
        )
    elif settings.default_llm_provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=settings.groq_api_key,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        google_api_key=settings.google_api_key,
    )


def create_coding_chain():
    """
    Build a coding assistant chain.

    Unlike the other agents which use ReAct loops, the coding agent is a
    simple chain (prompt → LLM → parser) because code generation doesn't
    require tool use — it's pure LLM reasoning.

    Returns:
        A LangChain runnable chain. Invoke with:
        chain.invoke({"input": "Write a Python function to...", "chat_history": []})
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", _CODING_SYSTEM),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()

    logger.info("Coding chain created")
    return chain


def create_code_review_chain():
    """
    Build a code review chain that analyzes and suggests improvements.

    Invoke with:
    chain.invoke({"code": "...", "language": "python", "context": "optional context"})
    """
    llm = _get_llm()

    review_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior code reviewer. Analyze the provided code and give structured feedback:\n"
         "1. **Correctness** — Does the code do what it's supposed to?\n"
         "2. **Security** — Any vulnerabilities (SQL injection, XSS, etc.)?\n"
         "3. **Performance** — Any bottlenecks or inefficiencies?\n"
         "4. **Maintainability** — Is it readable and well-structured?\n"
         "5. **Best Practices** — Does it follow language/framework conventions?\n"
         "6. **Suggested Improvements** — Rewrite any problematic sections.\n"
         "Be specific and constructive, not generic."
        ),
        ("human",
         "Language: {language}\n\nContext: {context}\n\nCode to review:\n```{language}\n{code}\n```"
        ),
    ])

    chain = review_prompt | llm | StrOutputParser()
    return chain


# Pre-built instances
coding_chain = create_coding_chain()
code_review_chain = create_code_review_chain()

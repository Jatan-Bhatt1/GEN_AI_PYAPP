"""
Web Search Tool — Tavily-powered real-time web search.
Returns structured results with URLs, titles, and content snippets.
"""

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from backend.config import get_settings

settings = get_settings()


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web")
    max_results: int = Field(default=5, description="Number of results to return (1-10)")


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web for real-time information using Tavily.
    Use this tool when you need current news, recent events, or live data
    that your training data may not contain.

    Args:
        query: Search query string
        max_results: How many results to return

    Returns:
        Formatted string with search results including URLs and content snippets
    """
    try:
        searcher = TavilySearchResults(
            max_results=max_results,
            tavily_api_key=settings.tavily_api_key,
        )
        results = searcher.invoke(query)

        if not results:
            return f"No results found for query: '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            url = r.get("url", "N/A")
            content = r.get("content", "").strip()
            title = r.get("title", "Untitled")
            formatted.append(
                f"Result {i}:\n"
                f"  Title: {title}\n"
                f"  URL: {url}\n"
                f"  Content: {content[:500]}..."
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {str(e)}. Query was: '{query}'"

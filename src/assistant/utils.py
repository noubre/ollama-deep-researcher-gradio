import os
import requests
from typing import Dict, Any, List, Optional
from tavily import TavilyClient
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from functools import wraps

def traceable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper

@traceable
def tavily_search(query, api_key=None, max_results=5, search_depth="basic", include_raw_content=False):
    """
    Search the web using Tavily API.
    
    Args:
        query (str): The search query.
        api_key (str, optional): Tavily API key. Defaults to None.
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        search_depth (str, optional): Depth of search. Defaults to "basic".
        include_raw_content (bool, optional): Whether to include raw content. Defaults to False.
        
    Returns:
        dict: The search results.
    """
    import requests
    import json
    import re
    
    # Check if query is None or empty
    if not query:
        return {"results": []}
    
    # Convert to string if not already
    if not isinstance(query, str):
        query = str(query)
    
    # Original query for error reporting
    original_query = query
    
    # Advanced query cleaning and simplification
    try:
        # Remove any instructions or explanations
        # Look for patterns like "search for...", "query:", etc.
        patterns = [
            r"^.*?(search\s+(?:for|about)|query\s*:|find\s+information\s+about)(.+)$",
            r"^to\s+gather\s+.*?about\s+(.*?),.*$",
            r"^.*?(?:search|query)\s+(?:for|using)\s*[\"'](.+?)[\"'].*$"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                simplified = match.group(2) if len(match.groups()) > 1 else match.group(1)
                if simplified and len(simplified.strip()) > 5:
                    query = simplified.strip()
                    break
        
        # Strip quotes and excess whitespace
        query = query.strip('"\'').strip()
        
        # Remove meta-instructions
        query = re.sub(r"this\s+approach\s+ensures.*$", "", query, flags=re.IGNORECASE).strip()
        query = re.sub(r"this\s+query\s+will\s+help.*$", "", query, flags=re.IGNORECASE).strip()
        
        # If still too long (more than 50 words), take just the first 10 words
        words = query.split()
        if len(words) > 50:
            query = " ".join(words[:10])
        
        # Remove any remaining quotes
        query = query.replace('"', '').replace("'", "")
        
        # Clean up extra spaces
        query = re.sub(r'\s+', ' ', query).strip()
        
    except Exception as e:
        print(f"Error in query cleaning: {e}")
        # If cleaning fails, use a simple fallback approach
        # Just take the first 10 words
        words = query.split()
        query = " ".join(words[:10])
    
    print(f"Original query: {original_query}")
    print(f"Cleaned query for Tavily search: {query}")
    
    # If API key is not provided, try to get it from environment
    if not api_key:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
    
    # Check if API key exists
    if not api_key:
        raise ValueError("Tavily API key is required. Please provide it or set it in your environment variables.")
    
    try:
        # Define the API endpoint
        url = "https://api.tavily.com/search"
        
        # Define the parameters
        params = {
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": False,
            "include_raw_content": include_raw_content,
            "max_results": max_results,
        }
        
        # Make the request
        response = requests.post(url, json=params)
        response.raise_for_status()
        
        # Return the response
        return response.json()
    except Exception as e:
        print(f"Error in Tavily search: {e}")
        print(f"Query that caused the error: '{query}'")
        # Return empty results
        return {"results": []}

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int, api_key: Optional[str] = None) -> Dict[str, Any]:
    if api_key is None:
        api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("Perplexity API key is required but not provided")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Search the web and provide factual information with sources."},
            {"role": "user", "content": query}
        ]
    }
    response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations", ["https://perplexity.ai"])
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    return {"results": results}

@traceable
def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, Any]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if fetch_full_page:
            for result in results:
                try:
                    response = requests.get(result["href"], timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        # Extract text content from the page
                        paragraphs = soup.find_all("p")
                        full_content = "\n".join([p.get_text() for p in paragraphs])
                        result["raw_content"] = full_content
                    else:
                        result["raw_content"] = f"Failed to fetch content: HTTP {response.status_code}"
                except Exception as e:
                    result["raw_content"] = f"Error fetching content: {str(e)}"
            
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append({
                "title": r.get("title", f"Result {i}"),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
                "raw_content": r.get("raw_content", r.get("body", ""))
            })
        
        return {"results": formatted_results}
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return {"results": []}

def deduplicate_and_format_sources(search_results: Dict[str, Any], max_tokens_per_source: int = 1000, include_raw_content: bool = True) -> str:
    """Format search results as a string, deduplicating content and limiting tokens per source."""
    formatted_results = []
    urls_seen = set()
    
    for result in search_results.get("results", []):
        url = result.get("url", "")
        if url in urls_seen:
            continue
        urls_seen.add(url)
        
        title = result.get("title", "Untitled")
        content = result.get("raw_content", "") if include_raw_content and result.get("raw_content") else result.get("content", "")
        
        # Simple token truncation (approximation)
        if content and len(content.split()) > max_tokens_per_source:
            truncated_content = " ".join(content.split()[:max_tokens_per_source]) + "..."
        else:
            truncated_content = content
        
        formatted_source = f"## {title}\n\nURL: {url}\n\n{truncated_content}\n\n---\n\n"
        formatted_results.append(formatted_source)
    
    return "".join(formatted_results)

def format_sources(search_results: Dict[str, Any]) -> str:
    """Format search results as a compact string for state tracking."""
    formatted_results = []
    for result in search_results.get("results", []):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        formatted_results.append(f"{title} - {url}")
    
    return ", ".join(formatted_results)

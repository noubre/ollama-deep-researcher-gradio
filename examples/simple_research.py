#!/usr/bin/env python
"""
Simple command-line example of using the Ollama Deep Researcher.
This script allows testing the research functionality without using the Gradio interface.
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv()

from src.assistant.graph import graph

def main():
    parser = argparse.ArgumentParser(description="Run the Ollama Deep Researcher from the command line")
    parser.add_argument("topic", help="The research topic to investigate")
    parser.add_argument("--search-api", choices=["duckduckgo", "tavily", "perplexity"], 
                        default=os.getenv("SEARCH_API", "duckduckgo"), help="The search API to use")
    parser.add_argument("--tavily-api-key", default=os.getenv("TAVILY_API_KEY", ""),
                       help="Tavily API Key (required if using Tavily)")
    parser.add_argument("--perplexity-api-key", default=os.getenv("PERPLEXITY_API_KEY", ""),
                       help="Perplexity API Key (required if using Perplexity)")
    parser.add_argument("--max-loops", type=int, default=int(os.getenv("MAX_WEB_RESEARCH_LOOPS", "3")), 
                       help="Maximum number of research loops")
    parser.add_argument("--fetch-full-page", action="store_true", 
                       default=os.getenv("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t"),
                       help="Fetch full page content for DuckDuckGo")
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "llama3.2"), help="Ollama model to use")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/"), 
                        help="Base URL for Ollama API")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.search_api == "tavily" and not args.tavily_api_key:
        parser.error("--tavily-api-key is required when using the tavily search API")
    if args.search_api == "perplexity" and not args.perplexity_api_key:
        parser.error("--perplexity-api-key is required when using the perplexity search API")
    
    config = {
        "configurable": {
            "max_web_research_loops": args.max_loops,
            "local_llm": args.ollama_model,
            "search_api": args.search_api,
            "fetch_full_page": args.fetch_full_page,
            "ollama_base_url": args.ollama_base_url,
            "tavily_api_key": args.tavily_api_key,
            "perplexity_api_key": args.perplexity_api_key
        }
    }
    
    input_state = {"research_topic": args.topic}
    
    try:
        print(f"Starting research on: {args.topic}")
        print(f"Using search API: {args.search_api}")
        print(f"Maximum research loops: {args.max_loops}")
        print("Working...\n")
        
        output = graph.invoke(input_state, config)
        
        print("\n===== RESEARCH SUMMARY =====\n")
        print(output["running_summary"])
        print("\n============================\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

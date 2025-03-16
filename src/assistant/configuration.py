import os
from dotenv import load_dotenv
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from enum import Enum

# Load environment variables from .env file
load_dotenv()

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", "duckduckgo"))
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    tavily_api_key: Optional[str] = os.environ.get("TAVILY_API_KEY", None)
    perplexity_api_key: Optional[str] = os.environ.get("PERPLEXITY_API_KEY", None)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: configurable.get(f.name, os.environ.get(f.name.upper()))
            for f in fields(cls)
            if f.init
        }
        # Convert search_api to Enum if it's a string
        if "search_api" in values and isinstance(values["search_api"], str):
            values["search_api"] = SearchAPI(values["search_api"])
        return cls(**{k: v for k, v in values.items() if v is not None})

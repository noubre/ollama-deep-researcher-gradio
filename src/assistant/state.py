import operator
from dataclasses import dataclass, field
from typing import List, Optional
from typing_extensions import TypedDict, Annotated

class SummaryState(TypedDict):
    """Represents the state including a research topic and the running summary."""
    research_topic: str
    running_summary: Optional[str]
    web_research_results: List[str]
    research_loop_count: int
    sources_gathered: List[str]
    search_query: Optional[str]

@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None) # Final report

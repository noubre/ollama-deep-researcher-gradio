from typing import TypedDict, List, Dict, Any, Optional, Sequence
import json
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from .configuration import Configuration, SearchAPI
from .utils import tavily_search, perplexity_search, duckduckgo_search, deduplicate_and_format_sources, format_sources
from .state import SummaryState
from .prompts import query_writer_instructions, summarizer_instructions, reflection_instructions

def create_research_prompt(state: SummaryState, config: RunnableConfig) -> str:
    """Create a prompt for generating a search query based on the research topic and state."""
    if not state.get("running_summary"):
        user_message = f"""I need detailed information about: {state.get("research_topic")}
        
        Generate a specific web search query that will help gather comprehensive information about this topic. 
        The query should be focused on finding factual, detailed, and credible information.
        
        Output only the search query with no additional text, explanations, or formatting."""
    else:
        previous_summary = state.get("running_summary", "No summary available yet.")
        
        # Format the sources_gathered for display
        sources_list = []
        for source_group in state.get("sources_gathered", []):
            for source in source_group:
                source_str = f"- {source.get('title', 'Untitled')} ({source.get('url', 'No URL')})"
                sources_list.append(source_str)
        
        previous_sources = "\n".join(sources_list) if sources_list else "No sources gathered yet."
        
        user_message = f"""I'm researching: {state.get("research_topic")}
        
        Current summary: 
        {previous_summary}
        
        Sources already gathered:
        {previous_sources}
        
        Based on the current summary and sources, generate a specific search query to gather additional information 
        that would enhance this research. Focus on filling knowledge gaps or exploring new aspects.
        
        Output only the search query with no additional text, explanations, or formatting."""
    return user_message

def create_search_query(state: SummaryState, config: RunnableConfig):
    """Generate a search query based on the research topic and current state."""
    configurable = Configuration.from_runnable_config(config)
    
    llm = ChatOllama(
        model=configurable.local_llm,
        base_url=configurable.ollama_base_url,
    )
    
    prompt_text = create_research_prompt(state, config)
    # Create a HumanMessage from the prompt text
    prompt_message = [HumanMessage(content=prompt_text)]
    response = llm.invoke(prompt_message)
    
    # Extract the content from the AIMessage
    if hasattr(response, "content"):
        raw_query = response.content
    else:
        raw_query = str(response)
    
    # Clean up the query - handle think tags, JSON formatting, and other issues
    clean_query = raw_query.strip()
    
    # Remove thinking process if enclosed in <think> tags
    think_start = clean_query.find("<think>")
    think_end = clean_query.find("</think>")
    
    if think_start != -1 and think_end != -1 and think_end > think_start:
        # Extract text after the thinking section
        after_thinking = clean_query[think_end + 8:].strip()
        if after_thinking:
            clean_query = after_thinking
        else:
            # If nothing after thinking, try to extract a concise version from within thinking
            thinking_content = clean_query[think_start + 7:think_end].strip()
            # Try to use the last paragraph or sentence as it might be more conclusive
            paragraphs = thinking_content.split('\n\n')
            if paragraphs:
                clean_query = paragraphs[-1].strip()
    
    # Check if it's JSON formatted and extract the actual query
    if clean_query.startswith('{') and clean_query.endswith('}'):
        try:
            query_obj = json.loads(clean_query)
            # Extract the query from common JSON formats
            if 'q' in query_obj:
                clean_query = query_obj['q']
            elif 'query' in query_obj:
                clean_query = query_obj['query']
            elif 'searchQuery' in query_obj:
                clean_query = query_obj['searchQuery']
        except json.JSONDecodeError as e:
            print(f"JSON decode error in create_search_query: {e}")
            print(f"Proceeding with original query string: {clean_query}")
            # If it's not valid JSON, just use the cleaned string
            pass
    
    # If the query is too long, truncate it
    if len(clean_query) > 500:
        clean_query = clean_query[:497] + "..."
    
    print(f"Generated search query: {clean_query}")
    
    return {"search_query": clean_query}

def increment_research_count(state: SummaryState) -> SummaryState:
    """Increment the research loop count."""
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    return state

def web_research(state: SummaryState, config: RunnableConfig):
    """Conduct web research based on the search query."""
    configurable = Configuration.from_runnable_config(config)

    search_query = state["search_query"]
    print(f"Web research query: {search_query}")
    
    # Increment the research loop count
    current_loop_count = state.get("research_loop_count", 0)
    next_loop_count = current_loop_count + 1
    
    search_api = configurable.search_api

    print(f"Research loop count: {current_loop_count}/{configurable.max_web_research_loops}")
    print(f"Using search API: {search_api}")

    search_results = {}
    
    try:
        # Show which search API we're using
        print(f"Searching with {search_api} API...")
        
        if search_api == SearchAPI.TAVILY:
            print(f"Executing Tavily search for: {search_query[:50]}...")
            search_results = tavily_search(
                query=search_query,
                api_key=configurable.tavily_api_key,
                search_depth="advanced",
                include_raw_content=True,
                max_results=3
            )
        elif search_api == SearchAPI.PERPLEXITY:
            print(f"Executing Perplexity search for: {search_query[:50]}...")
            search_results = perplexity_search(
                query=search_query,
                perplexity_search_loop_count=current_loop_count,
                api_key=configurable.perplexity_api_key,
            )
        elif search_api == SearchAPI.DUCKDUCKGO:
            print(f"Executing DuckDuckGo search for: {search_query[:50]}...")
            search_results = duckduckgo_search(
                query=search_query,
                fetch_full_page=configurable.fetch_full_page
            )
        else:
            # Default to DuckDuckGo if not specified
            print(f"Using default DuckDuckGo search for: {search_query[:50]}...")
            search_results = duckduckgo_search(
                query=search_query,
                fetch_full_page=configurable.fetch_full_page
            )
            
        # Get formatted sources
        print("Processing search results...")
        formatted_sources = deduplicate_and_format_sources(search_results)

        # Get clean sources list for record-keeping
        clean_sources = format_sources(search_results)
        
        # Show how many sources were found
        source_count = len(clean_sources) if clean_sources else 0
        print(f"Found {source_count} relevant sources")
        
        # Update research results
        web_research_results = state.get("web_research_results", [])
        web_research_results.append(formatted_sources)
        
        # Update sources
        sources_gathered = state.get("sources_gathered", [])
        sources_gathered.append(clean_sources)
        
        print(f"Completed web research cycle {next_loop_count}")
        
        # Return updated state
        return {
            "web_research_results": web_research_results,
            "sources_gathered": sources_gathered,
            "research_loop_count": next_loop_count
        }
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        # Even on error, increment the loop count to avoid infinite loops
        return {
            "research_loop_count": next_loop_count
        }

def format_sources(search_results):
    """Format the sources from search results."""
    sources = []
    for result in search_results.get("results", []):
        sources.append({
            "title": result.get("title", ""),
            "url": result.get("url", "")
        })
    return sources

def deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True):
    """Deduplicate and format search results."""
    results = search_results.get("results", [])
    formatted_results = []
    
    for i, result in enumerate(results):
        title = result.get("title", f"Result {i+1}")
        url = result.get("url", "")
        
        # Choose the appropriate content field
        if include_raw_content and "raw_content" in result and result["raw_content"]:
            content = result["raw_content"]
        else:
            content = result.get("content", "")
        
        # Limit content length for token efficiency
        if content and len(content) > max_tokens_per_source:
            content = content[:max_tokens_per_source] + "..."
        
        formatted_result = f"Source {i+1}: {title}\nURL: {url}\n\nContent:\n{content}\n\n"
        formatted_results.append(formatted_result)
    
    return "\n".join(formatted_results)

def create_summary_prompt(state: SummaryState):
    """Create a prompt for summarizing the web research results."""
    research_topic = state["research_topic"]
    web_research_results = "\n\n".join(state["web_research_results"])
    
    template = """You are a research assistant that creates comprehensive, well-structured summaries.

    RESEARCH TOPIC:
    {research_topic}

    SOURCES:
    {web_research_results}

    Guidelines:
    1. Create a detailed, informative summary that synthesizes the information from all sources.
    2. Organize the information logically with clear headings and subheadings.
    3. Include specific facts, data, and key concepts.
    4. Maintain a neutral, informative tone.
    5. Use markdown formatting for structure: headings (# for main, ## for sub), bullet points, etc.
    6. Don't reference the source numbers or URLs in your summary.
    7. The summary should be comprehensive yet concise.

    SUMMARY:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(research_topic=research_topic, web_research_results=web_research_results)

def summarize_research(state: SummaryState, config: RunnableConfig):
    """Generate a summary of the research based on web search results."""
    configurable = Configuration.from_runnable_config(config)
    
    llm = ChatOllama(
        model=configurable.local_llm,
        base_url=configurable.ollama_base_url,
    )
    
    prompt = create_summary_prompt(state)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Extract the content from the AIMessage
    if hasattr(response, "content"):
        summary_content = response.content
    else:
        summary_content = str(response)
    
    return {"running_summary": summary_content}

def should_continue_research(state: SummaryState, config: RunnableConfig) -> str:
    """Determine if more research is needed based on loop count."""
    configurable = Configuration.from_runnable_config(config)
    
    # Get current loop count
    current_loop_count = state.get("research_loop_count", 0)
    
    # Get max loop count from config
    max_loops = configurable.max_web_research_loops
    
    print(f"Research loop count: {current_loop_count}/{max_loops}")
    
    # Check if we've reached the maximum number of loops
    if current_loop_count >= max_loops:
        print("Reached maximum number of research loops. Stopping.")
        return "exit"
    
    # Increment loop count for next iteration
    return "continue_research"

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    print("Starting to summarize research findings...")
    
    # Existing summary
    existing_summary = state.get("running_summary")

    # Most recent web research
    most_recent_web_research = state["web_research_results"][-1]

    # Build the human message
    if existing_summary:
        print("Updating existing summary with new information...")
        human_message_content = (
            f"<User Input> \n {state['research_topic']} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        print("Creating initial summary...")
        human_message_content = (
            f"<User Input> \n {state['research_topic']} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    print("Generating summary with LLM...")
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result.content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    print("Summary generation complete")
    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    print("Reflecting on current summary to identify knowledge gaps...")
    
    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    
    try:
        print("Generating follow-up query...")
        result = llm_json_mode.invoke(
            [SystemMessage(content=reflection_instructions.format(research_topic=state["research_topic"])),
            HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state['running_summary']}")]
        )
        
        if hasattr(result, "content"):
            content = result.content
        else:
            content = str(result)
            
        print(f"Reflection result: {content}")
        
        try:
            import json
            follow_up_query = json.loads(content)
            
            # Get the follow-up query
            query = follow_up_query.get('follow_up_query')
            
            # If query exists and is valid, use it
            if query and isinstance(query, str) and len(query.strip()) > 0:
                print(f"Generated follow-up query: {query}")
                return {"search_query": query}
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content: {content}")
    except Exception as e:
        print(f"Error in reflection: {e}")
    
    # Fallback to a placeholder query
    fallback_query = f"Tell me more about {state['research_topic']}"
    print(f"Using fallback query: {fallback_query}")
    return {"search_query": fallback_query}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    print("Finalizing research summary...")
    
    # Format all accumulated sources into a single bulleted list
    all_sources = []
    for source_group in state.get("sources_gathered", []):
        for source in source_group:
            source_entry = f"- {source.get('title', 'Untitled')} ({source.get('url', 'No URL')})"
            all_sources.append(source_entry)
    
    sources_formatted = "\n".join(all_sources) if all_sources else "No sources gathered."
    
    print("Research process complete!")
    return {"running_summary": f"## Summary\n\n{state['running_summary']}\n\n### Sources:\n{sources_formatted}"}

def route_research(state: SummaryState, config: RunnableConfig) -> str:
    """Route the research based on the research loop count"""
    
    configurable = Configuration.from_runnable_config(config)
    current_count = state["research_loop_count"]
    max_loops = configurable.max_web_research_loops
    
    print(f"Research loop count: {current_count}/{max_loops}")
    
    if current_count >= max_loops:
        print("Reached maximum number of research loops. Proceeding to finalize summary.")
        return "finalize_summary"
    else:
        print(f"Continuing research (loop {current_count} of {max_loops}).")
        return "web_research"

# Create and configure the graph
def build_graph():
    """Build the research workflow graph."""
    workflow = StateGraph(SummaryState)
    
    # Add nodes to the graph
    workflow.add_node("generate_query", create_search_query)
    workflow.add_node("web_research", web_research)
    workflow.add_node("summarize", summarize_sources)  # Use summarize_sources function
    workflow.add_node("reflect", reflect_on_summary)   # Add reflection node
    workflow.add_node("finalize", finalize_summary)    # Add finalize node
    
    # Set the entry point
    workflow.set_entry_point("generate_query")
    
    # Add edges between nodes
    workflow.add_edge("generate_query", "web_research")
    workflow.add_edge("web_research", "summarize")
    
    # Decision node after summarize
    workflow.add_conditional_edges(
        "summarize",
        route_research,  # Use route_research function
        {
            "web_research": "reflect",
            "finalize_summary": "finalize"
        },
    )
    
    # Connect reflect back to generate_query to continue the loop
    workflow.add_edge("reflect", "generate_query")
    
    # Set the end node
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

# Initialize the graph
graph = build_graph()

# Default state initialization
def initialize_state(research_topic: str) -> SummaryState:
    """Initialize the state for a new research session."""
    return {
        "research_topic": research_topic,
        "running_summary": None,
        "web_research_results": [],
        "research_loop_count": 0,
        "sources_gathered": [],
        "search_query": None,
    }

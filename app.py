import gradio as gr
import os
import time
from dotenv import load_dotenv
from src.assistant.graph import graph

# Load environment variables from .env file
load_dotenv()

def run_research(research_topic, search_api, tavily_api_key, perplexity_api_key, max_loops, fetch_full_page, ollama_model, ollama_base_url, progress=gr.Progress()):
    """Run the research workflow with the provided parameters."""
    if not research_topic:
        return "Please provide a research topic.", None
    if search_api == "tavily" and not tavily_api_key:
        return "Please provide Tavily API Key", None
    elif search_api == "perplexity" and not perplexity_api_key:
        return "Please provide Perplexity API Key", None

    # Provide initial status
    yield "🔍 **Starting research process...**", gr.update(visible=True)
    
    print(f"Starting research on: {research_topic}")
    print(f"Using search API: {search_api}")
    
    # Make sure we're passing string values for the search_api
    config = {
        "configurable": {
            "max_web_research_loops": max_loops,
            "local_llm": ollama_model,
            "search_api": search_api,  # This is already a string from the dropdown
            "fetch_full_page": fetch_full_page,
            "ollama_base_url": ollama_base_url,
            "tavily_api_key": tavily_api_key,
            "perplexity_api_key": perplexity_api_key
        }
    }
    input_state = {
        "research_topic": research_topic,
        "running_summary": None,
        "web_research_results": [],
        "research_loop_count": 0,
        "sources_gathered": [],
        "search_query": None
    }

    try:
        # Update progress as we go
        progress(0, "Initializing research...")
        yield f"🔍 **Researching:** {research_topic}\n\n🔄 **Initializing...**", gr.update(visible=True)
        time.sleep(0.5)  # Small delay to show progress

        # Generate query
        progress(0.1, "Generating search query...")
        yield f"🔍 **Researching:** {research_topic}\n\n🔄 **Generating search query...**", gr.update(visible=True)
        
        # Start the graph execution
        output = graph.invoke(input_state, config)
        
        # Research complete
        progress(1.0, "Research complete!")
        yield output["running_summary"], gr.update(visible=False)
    except Exception as e:
        import traceback
        error_msg = f"❌ **An error occurred:**\n\n```\n{str(e)}\n```\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg, gr.update(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("# Ollama Deep Researcher")
    gr.Markdown("Enter a research topic and configure the options below to generate a summary.")
    with gr.Row():
        with gr.Column():
            research_topic = gr.Textbox(label="Research Topic", placeholder="Enter a topic to research...")
            search_api = gr.Radio(
                ["duckduckgo", "tavily", "perplexity"], 
                label="Search API", 
                value=os.getenv("SEARCH_API", "duckduckgo")
            )
            tavily_api_key = gr.Textbox(
                label="Tavily API Key (required if Tavily selected)", 
                type="password",
                value=os.getenv("TAVILY_API_KEY", "")
            )
            perplexity_api_key = gr.Textbox(
                label="Perplexity API Key (required if Perplexity selected)", 
                type="password",
                value=os.getenv("PERPLEXITY_API_KEY", "")
            )
            max_loops = gr.Slider(
                1, 10, 
                value=int(os.getenv("MAX_WEB_RESEARCH_LOOPS", "3")), 
                step=1, 
                label="Max Research Loops"
            )
            fetch_full_page = gr.Checkbox(
                label="Fetch Full Page (DuckDuckGo only)", 
                value=os.getenv("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
            )
            ollama_model = gr.Textbox(
                label="Ollama Model", 
                value=os.getenv("OLLAMA_MODEL", "llama3.2")
            )
            ollama_base_url = gr.Textbox(
                label="Ollama Base URL", 
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/")
            )
            submit_btn = gr.Button("Start Research", variant="primary")
        with gr.Column():
            with gr.Row():
                status_indicator = gr.Markdown(visible=False)
            output = gr.Markdown(label="Research Summary")
    
    submit_btn.click(
        fn=run_research,
        inputs=[research_topic, search_api, tavily_api_key, perplexity_api_key, max_loops, fetch_full_page, ollama_model, ollama_base_url],
        outputs=[output, status_indicator]
    )

if __name__ == "__main__":
    demo.launch()

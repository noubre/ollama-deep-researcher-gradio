import gradio as gr
import os
import time
from dotenv import load_dotenv
from src.assistant.graph import graph

# Load environment variables from .env file
load_dotenv()

def run_research(research_topic, search_api, tavily_api_key, perplexity_api_key, max_loops, fetch_full_page, ollama_model, ollama_base_url):
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
        # Provide user with informative status message
        yield f"🔍 **Researching:** {research_topic}\n\n⏳ Preparing to search...", gr.update(visible=True)
        
        # Start the graph execution
        output = graph.invoke(input_state, config)
        
        # Research complete - hide status and show final results
        yield output["running_summary"], gr.update(visible=False)
    except Exception as e:
        import traceback
        error_msg = f"❌ **Error:**\n\n```\n{str(e)}\n```"
        print(f"An error occurred: {str(e)}\n\n{traceback.format_exc()}")
        yield error_msg, gr.update(visible=False)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🔍 Ollama Deep Researcher")
    gr.Markdown("Enter a research topic and configure the options below to generate a detailed summary.")
    
    with gr.Row():
        with gr.Column(scale=1):
            research_topic = gr.Textbox(
                label="Research Topic", 
                placeholder="Enter a topic to research...",
                info="What would you like to learn about?"
            )
            search_api = gr.Radio(
                ["duckduckgo", "tavily", "perplexity"], 
                label="Search API", 
                value=os.getenv("SEARCH_API", "duckduckgo"),
                info="Which search API to use for web research"
            )
            tavily_api_key = gr.Textbox(
                label="Tavily API Key", 
                type="password",
                value=os.getenv("TAVILY_API_KEY", ""),
                info="Required for Tavily search"
            )
            perplexity_api_key = gr.Textbox(
                label="Perplexity API Key", 
                type="password",
                value=os.getenv("PERPLEXITY_API_KEY", ""),
                info="Required for Perplexity search"
            )
            max_loops = gr.Slider(
                1, 10, 
                value=int(os.getenv("MAX_WEB_RESEARCH_LOOPS", "3")), 
                step=1, 
                label="Max Research Loops",
                info="Number of search iterations to perform"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
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
            
            submit_btn = gr.Button("Start Research", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status_indicator = gr.Markdown(visible=False, elem_id="status_box")
            output = gr.Markdown(label="Research Summary", elem_id="output_box")
    
    # Add some custom styling for the status indicator
    demo.load(js="""
    function setupStyles() {
        const statusBox = document.getElementById('status_box');
        if (statusBox) {
            statusBox.style.padding = '15px';
            statusBox.style.marginBottom = '15px';
            statusBox.style.backgroundColor = '#f0f7ff';
            statusBox.style.border = '1px solid #cce5ff';
            statusBox.style.borderRadius = '5px';
            statusBox.style.fontFamily = 'monospace';
        }
        
        const outputBox = document.getElementById('output_box');
        if (outputBox) {
            outputBox.style.maxHeight = '600px';
            outputBox.style.overflowY = 'auto';
        }
    }
    
    // Run when the page loads
    setupStyles();
    // Also run after each update
    document.addEventListener('DOMContentLoaded', setupStyles);
    return [];
    """)
    
    submit_btn.click(
        fn=run_research,
        inputs=[research_topic, search_api, tavily_api_key, perplexity_api_key, max_loops, fetch_full_page, ollama_model, ollama_base_url],
        outputs=[output, status_indicator]
    )

if __name__ == "__main__":
    demo.launch()

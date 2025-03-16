# Ollama Deep Researcher

A research assistant that uses Ollama local LLMs and web search to perform deep research on topics. This implementation uses Gradio for a user-friendly web interface and removes Docker dependencies for simpler setup.

## 🚀 Quickstart

### Prerequisites
- Install [Ollama](https://ollama.com/download) and pull a local LLM (e.g., `ollama pull llama3.2`).
- Ensure Python 3.9 or higher is installed.

### Running with Gradio
1. Clone the repository:
   ```bash
   git clone https://github.com/langchain-ai/ollama-deep-researcher.git
   cd ollama-deep-researcher
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Run the Gradio app:
   ```bash
   python app.py
   ```

5. Open the provided URL (e.g., http://127.0.0.1:7860) in your browser to access the interface.

## 📝 Usage
- Enter a research topic.
- Select a search API (DuckDuckGo, Tavily, or Perplexity).
- Provide API keys if using Tavily or Perplexity.
- Configure options like max research loops and Ollama settings.
- Click "Start Research" to generate a markdown summary.

> **Note**: Ensure Ollama is running locally (default: http://localhost:11434) with the specified model.

## 🔍 Search APIs

### DuckDuckGo
- Doesn't require an API key
- Option to fetch full webpage content for more detailed results

### Tavily
- Requires an API key (get one at [tavily.com](https://tavily.com))
- Provides high-quality search results with content extraction

### Perplexity
- Requires an API key (get one at [perplexity.ai](https://perplexity.ai))
- Uses their Sonar model for intelligent search results

## 📋 Features
- Automatically generates search queries based on your research topic
- Performs iterative research to gather comprehensive information
- Creates well-structured, detailed summaries in markdown format
- Fully configurable parameters for customized research experience

## 🧠 How It Works
1. The system generates an initial search query from your research topic
2. It searches the web using your selected search API
3. Results are processed and fed to the LLM
4. The LLM generates a new, more focused search query
5. This process repeats for the specified number of loops
6. Finally, the LLM creates a comprehensive markdown summary

## 🛠️ Architecture
This project is built on:
- LangGraph for workflow orchestration
- LangChain for LLM integration
- Ollama for local LLM inference
- Gradio for the web interface

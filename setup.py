from setuptools import setup, find_packages

setup(
    name="ollama-deep-researcher",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "langgraph>=0.2.55",
        "langchain-core>=0.1.18",
        "langchain-community>=0.3.9",
        "tavily-python>=0.5.0",
        "langchain-ollama>=0.2.1",
        "duckduckgo-search>=7.3.0",
        "beautifulsoup4>=4.13.3",
        "gradio>=4.0.0",
        "requests>=2.31.0",
        "typing-extensions>=4.8.0",
        "python-dotenv>=1.0.0",
    ],
    author="LangChain AI",
    author_email="info@langchain.dev",
    description="A research assistant that uses Ollama local LLMs and web search to perform deep research on topics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/langchain-ai/ollama-deep-researcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

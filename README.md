# Cognito Agent: AI Research Platform

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/streamlit-1.30.0-orange.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/crewai-0.28.8-blueviolet.svg" alt="CrewAI">
  <img src="https://img.shields.io/badge/pinecone-3.2.2-yellow.svg" alt="Pinecone">
</p>

Cognito Agent is a Streamlit-based web application that provides a powerful and interactive interface for conducting AI-powered research. The platform leverages a multi-agent system built with `crewai` to automate research tasks, and it uses Pinecone for efficient vector-based storage and retrieval of research data.

## ✨ Features

- **🤖 AI Research Agent:** Conduct research on any topic using a crew of AI agents (Researcher and Writer).
- **🌐 Web-Powered:** The agents can search the web, scrape websites, and generate comprehensive reports.
- **🧠 Vector-Based Memory:** Research jobs and content are stored in a Pinecone vector database for semantic search and retrieval.
- **🖥️ Interactive UI:** A user-friendly web interface built with Streamlit for interacting with the AI agents and viewing results.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An account with [Pinecone](https://www.pinecone.io/)
- An account with [SerpAPI](https://serpapi.com/) for web searches
- An account with [Groq](https://groq.com/) for LLM access

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cognito-agent.git
    cd cognito-agent
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root of the project and add the following:
    ```
    PINECONE_API_KEY="your-pinecone-api-key"
    PINECONE_ENVIRONMENT="your-pinecone-environment"
    SERPAPI_API_KEY="your-serpapi-api-key"
    GROQ_API_KEY="your-groq-api-key"
    ```


### Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run src/app.py
    ```

2.  **Open your browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Enter a research topic** in the text input field.

4.  **Click "Start Research"** to begin the research process. The AI agents will start working, and the results will be displayed on the screen.

## 🛠️ Configuration

- **Agents and Tasks:** The behavior of the AI agents and the tasks they perform can be configured in `config/agents.yaml` and `config/tasks.yaml`.
- **Pinecone Indexes:** The configuration for the Pinecone indexes can be modified in `config/pinecone_indexes.yaml`.

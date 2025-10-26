import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.pinecone_ops.operations import PineconeOperations
from src.agents.crew_manager import run_research
from src.tools.serpapi_search_tool import SerpAPISearchTool
from src.tools.scraper_tool import WebScraperTool
from src.tools.rag_tool import PineconeRAGTool

# Page config
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# --- Authentication --- 
# Simple password-based authentication for demonstration
# For this to work, you need to create a file `.streamlit/secrets.toml`
# and add the following line:
# PASSWORD = "your_password"

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# --- Main Application --- 

# Initialize session state
if 'job_status' not in st.session_state:
    st.session_state['job_status'] = {}
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'current_job_id' not in st.session_state:
    st.session_state['current_job_id'] = None

# Sidebar
with st.sidebar:
    st.header("Configuration")
    serpapi_api_key = st.text_input("SerpAPI API Key", value=os.getenv("SERPAPI_API_KEY"), type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", value=os.getenv("PINECONE_API_KEY"), type="password")
    groq_api_key = st.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY"), type="password")

# Main interface
st.title("Interactive AI Research Agent")

# Create status placeholder before columns
status_placeholder = st.empty()

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Cognito Agent Prototype")
    topic = st.text_input("Research Topic:")
    if st.button("Start Research"):
        if topic:
            if not all([serpapi_api_key, pinecone_api_key, groq_api_key]):
                st.error("Please provide all API keys in the sidebar.")
            else:
                st.info(f"Starting research on: {topic}")
                # Set GROQ_API_KEY for crew_manager to use
                os.environ["GROQ_API_KEY"] = groq_api_key
                with st.spinner('Agents are working...'):
                    try:
                        # 1. Initialize Pinecone Operations (embeddings are local now)
                        pinecone_ops = PineconeOperations(
                            api_key=pinecone_api_key,
                            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
                        )

                        # 2. Create a job in Pinecone
                        job_info = pinecone_ops.create_research_job(topic)
                        st.session_state.current_job_id = job_info['job_id']
                        status_placeholder.info(f"Job {job_info['job_id']} created. Agents are starting...")

                        # 3. Initialize Tools
                        search_tool = SerpAPISearchTool(api_key=serpapi_api_key, pinecone_ops=pinecone_ops)
                        scraper_tool = WebScraperTool()
                        rag_tool = PineconeRAGTool(pinecone_ops=pinecone_ops)

                        # 4. Run the research crew
                        result = run_research(topic, search_tool, scraper_tool, rag_tool)
                        
                        # 5. Update job status and show result
                        pinecone_ops.update_research_job(job_id=job_info['job_id'], status='complete', report=result)
                        st.success("Research complete!")
                        st.markdown(result)
                        status_placeholder.success(f"Job {job_info['job_id']} complete.")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        if 'current_job_id' in st.session_state and st.session_state.current_job_id:
                            pinecone_ops.update_research_job(job_id=st.session_state.current_job_id, status='error', error_message=str(e))
                            status_placeholder.error(f"Job {st.session_state.current_job_id} failed.")

        else:
            st.error("Please enter a research topic.")


with col2:
    st.header("Status")
    if not st.session_state.get('current_job_id'):
        st.info("Awaiting job...")
    
    st.header("History")
    # This part would also need to be implemented to fetch history from Pinecone
    if st.session_state.get('history', []):
        for job in st.session_state.get('history', []):
            st.write(f"• {job['topic']} - {job['status']}")
    else:
        st.write("No job history yet.")


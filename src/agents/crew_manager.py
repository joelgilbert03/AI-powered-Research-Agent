from crewai import Agent, Task, Crew
import yaml
import os
from langchain_groq import ChatGroq

def load_agents_config():
    # Corrected path to be relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), '../../config/agents.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_tasks_config():
    # Corrected path to be relative to this file's location
    config_path = os.path.join(os.path.dirname(__file__), '../../config/tasks.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_crew(topic, web_search_tool, web_scraper_tool, pinecone_retriever_tool):
    # Create a single agent that does both research and writing to stay within rate limits
    llm = ChatGroq(
        model="groq/llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=2048,  # Enough for complete response
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=2,
        request_timeout=120,
    )
    
    # Single agent that does research AND writing
    research_writer = Agent(
        role=f"{topic} Researcher & Writer",
        goal=f"Research {topic} and write a comprehensive report",
        backstory="You're an expert researcher who can find information and write clear reports.",
        verbose=True,
        allow_delegation=False,
        max_iter=1,
        memory=False,
        llm=llm,
        tools=[web_search_tool, web_scraper_tool, pinecone_retriever_tool]
    )
    
    # Single combined task
    research_and_write_task = Task(
        description=f"""Research {topic} and create a report:
        1. Search for 3-5 credible sources about {topic}
        2. Extract key information from each source
        3. Write a brief markdown report with:
           - Introduction
           - Key findings (bullet points)
           - Source citations with URLs
        Keep it concise but informative.""",
        expected_output="A markdown report with research findings and source citations",
        agent=research_writer,
        output_file=f"report_{topic.replace(' ', '_')}.md"
    )
    
    return Crew(
        agents=[research_writer],
        tasks=[research_and_write_task],
        verbose=True
    )

# Usage
def run_research(topic, web_search_tool, web_scraper_tool, pinecone_retriever_tool):
    import time
    
    crew = create_crew(topic, web_search_tool, web_scraper_tool, pinecone_retriever_tool)
    
    print("\n🔄 Starting single-agent research workflow...")
    print("ℹ️  Using 1 agent (instead of 2) to stay within rate limits\n")
    
    # Brief cooldown to ensure clean rate limit state
    initial_cooldown = 15
    print(f"⏱️  Waiting {initial_cooldown}s...\n")
    time.sleep(initial_cooldown)
    
    max_retries = 2
    retry_delay = 60  # 60 seconds between retries
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\n⏳ Attempt {attempt + 1}/{max_retries} - Cooling down for {retry_delay}s...")
                time.sleep(retry_delay)
            
            result = crew.kickoff(inputs={'topic': topic})
            print("\n✅ Research completed successfully!")
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limit errors
            if 'rate_limit' in error_str or 'ratelimit' in error_str:
                if attempt < max_retries - 1:
                    print(f"\n⚠️  Rate limit hit. Waiting {retry_delay}s for reset...")
                    time.sleep(retry_delay)
                else:
                    print(f"\n❌ Rate limit exceeded after {max_retries} attempts.")
                    print("💡 Groq Free Tier: 6000 tokens/minute limit")
                    print("   → Wait 90 seconds, then try again")
                    print("   → Or upgrade: https://console.groq.com/settings/billing")
                    raise e
            
            # Check for empty response errors
            elif 'none or empty' in error_str or 'invalid response' in error_str:
                print(f"\n⚠️  Empty response from LLM (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay}s and retrying...")
                    time.sleep(retry_delay)
                else:
                    print("\n❌ Failed to get valid response after retries")
                    raise e
            else:
                raise e

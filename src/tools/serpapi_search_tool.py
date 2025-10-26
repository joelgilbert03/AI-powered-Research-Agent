from crewai.tools import BaseTool
from serpapi import GoogleSearch
from typing import List, Dict, Any, Optional
from pydantic import Field

from src.pinecone_ops.operations import PineconeOperations

class SerpAPISearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search the web for information on a given topic using SerpAPI"
    api_key: str = Field(..., description="SerpAPI API key")
    pinecone_ops: Optional[PineconeOperations] = Field(default=None, description="Pinecone operations instance")

    def _run(self, query: str, job_id: str = None, max_results: int = 5) -> List[Dict]:
        """Execute web search and store results in Pinecone"""
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": max_results
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        
        formatted_results = self._format_results(results)

        # Store results in Pinecone if job_id provided
        if job_id and self.pinecone_ops and formatted_results:
            for result in formatted_results:
                # The 'content' is not available in SerpAPI results, so we use the snippet.
                self.pinecone_ops.content_manager.store_source(
                    job_id=job_id,
                    url=result.get('url'),
                    title=result.get('title'),
                    content=result.get('snippet', ''),
                    credibility_score=result.get('relevance_score', 0.5)
                )
        
        return formatted_results

    def _format_results(self, raw_results: Dict) -> List[Dict]:
        """Format API results for agent consumption"""
        formatted = []
        if "organic_results" in raw_results:
            for result in raw_results.get("organic_results", []):
                formatted.append({
                    'url': result.get('link', ''),
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    # SerpAPI does not provide full content, so snippet is used as a fallback.
                    'content': result.get('snippet', ''), 
                    'relevance_score': 1 / result.get('position', 1) # Using position as a proxy for relevance
                })
        return formatted

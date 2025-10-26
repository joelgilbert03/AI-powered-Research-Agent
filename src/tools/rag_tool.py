from crewai.tools import BaseTool
from typing import List, Dict
from pydantic import Field
from src.pinecone_ops.operations import PineconeOperations

class PineconeRAGTool(BaseTool):
    name: str = "Pinecone RAG Tool"
    description: str = "Retrieve relevant context from Pinecone vector database"
    pinecone_ops: PineconeOperations = Field(..., description="Pinecone operations instance")
    
    def _run(self, query: str, job_id: str = None, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for query"""
        try:
            # Search in job-specific content or across all content
            contexts = self.pinecone_ops.content_manager.search_content(
                query=query,
                job_id=job_id,
                top_k=top_k
            )
            
            return contexts
            
        except Exception as e:
            return [{"error": f"RAG retrieval failed: {str(e)}"}]
    
    def get_context_string(self, query: str, job_id: str = None, top_k: int = 5) -> str:
        """Get formatted context string for LLM"""
        contexts = self._run(query, job_id, top_k)
        
        if not contexts or 'error' in contexts[0]:
            return "No relevant context found."
        
        context_str = "Relevant context:\n\n"
        for i, ctx in enumerate(contexts, 1):
            context_str += f"Source {i}: {ctx['title']}\n"
            context_str += f"URL: {ctx['url']}\n"
            context_str += f"Content: {ctx['content_preview']}\n"
            context_str += f"Relevance: {ctx['relevance_score']:.3f}\n\n"
        
        return context_str
    
    def get_similar_research(self, topic: str, limit: int = 3) -> List[Dict]:
        """Find similar research jobs for context"""
        try:
            similar_jobs = self.pinecone_ops.research_manager.search_jobs_by_topic(
                search_topic=topic,
                limit=limit
            )
            
            return similar_jobs
            
        except Exception as e:
            return [{"error": f"Similar research search failed: {str(e)}"}]

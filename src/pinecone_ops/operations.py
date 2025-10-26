
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional

class PineconeDataManager:
    """Unified Pinecone manager for all data storage"""
    
    def __init__(self, api_key: str, environment: str, google_api_key: str = None):
        self.pc = Pinecone(api_key=api_key)
        self.environment = environment
        
        # Initialize sentence-transformer model for embeddings (local, no API needed)
        print("[INFO] Loading embedding model (sentence-transformers)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[INFO] Embedding model loaded successfully")
        
        # Define indexes for different data types
        self.indexes = {
            'research_jobs': 'cognito-jobs',      # Job metadata and status
            'research_content': 'cognito-content',  # Research content and sources
        }
        
        # Initialize indexes if they don't exist
        self._initialize_indexes()
    
    def _initialize_indexes(self):
        """Create Pinecone indexes if they don't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        for index_name, pinecone_name in self.indexes.items():
            if pinecone_name not in existing_indexes:
                self.pc.create_index(
                    name=pinecone_name,
                    dimension=384,  # all-MiniLM-L6-v2 embeddings
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
        
        # Connect to indexes
        self.job_index = self.pc.Index(self.indexes['research_jobs'])
        self.content_index = self.pc.Index(self.indexes['research_content'])
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence-transformers"""
        # Generate embedding using local model (no API calls)
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _generate_job_id(self, topic: str, job_type: str = 'research') -> str:
        """Generate unique job ID"""
        timestamp = datetime.now().isoformat()
        content = f"{job_type}_{topic}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

# Research Job Management
class ResearchJobManager(PineconeDataManager):
    """Manage research jobs in Pinecone"""
    
    def create_job(self, topic: str, user_id: str = 'anonymous') -> Dict[str, str]:
        """Create new research job"""
        job_id = self._generate_job_id(topic, 'research')
        namespace = f"job_{job_id}"
        
        # Create job metadata
        job_metadata = {
            'job_id': job_id,
            'topic': topic,
            'user_id': user_id,
            'job_type': 'research',
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'namespace': namespace,
            'source_count': 0,
            'processing_time_seconds': 0,
            'report': '',
            'error_message': ''
        }
        
        # Generate embedding from topic and metadata
        embedding_text = f"Research job: {topic} Status: pending Type: research"
        embedding = self._generate_embedding(embedding_text)
        
        # Store in Pinecone
        self.job_index.upsert(
            vectors=[{
                'id': job_id,
                'values': embedding,
                'metadata': job_metadata
            }],
            namespace='jobs'
        )
        
        return {'job_id': job_id, 'namespace': namespace}
    
    def update_job_status(self, job_id: str, status: str, report: str = None, 
                         error_message: str = None, source_count: int = None) -> bool:
        """Update job status and results"""
        try:
            # Get current job data
            job_data = self.get_job(job_id)
            if not job_data:
                return False
            
            # Update metadata
            updated_metadata = job_data['metadata'].copy()
            updated_metadata.update({
                'status': status,
                'updated_at': datetime.now().isoformat()
            })
            
            if report:
                updated_metadata['report'] = report
            if error_message:
                updated_metadata['error_message'] = error_message
            if source_count is not None:
                updated_metadata['source_count'] = source_count
            
            # Generate new embedding with updated status
            embedding_text = f"Research job: {updated_metadata['topic']} Status: {status} Type: research"
            if report:
                embedding_text += f" Report: {report[:500]}"  # Truncate for embedding
            
            embedding = self._generate_embedding(embedding_text)
            
            # Update in Pinecone
            self.job_index.upsert(
                vectors=[{
                    'id': job_id,
                    'values': embedding,
                    'metadata': updated_metadata
                }],
                namespace='jobs'
            )
            
            return True
            
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job by ID"""
        try:
            response = self.job_index.fetch(ids=[job_id], namespace='jobs')
            # New Pinecone API returns object with .vectors attribute
            if hasattr(response, 'vectors') and job_id in response.vectors:
                return response.vectors[job_id]
            return None
        except Exception as e:
            print(f"Error fetching job {job_id}: {e}")
            return None
    
    def get_job_history(self, user_id: str = None, limit: int = 10) -> List[Dict]:
        """Get job history using similarity search"""
        try:
            # Search query for job history
            if user_id:
                query_text = f"Research jobs for user {user_id} job history"
            else:
                query_text = "Research job history all jobs"
            
            query_embedding = self._generate_embedding(query_text)
            
            response = self.job_index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                namespace='jobs',
                filter={'job_type': {'$eq': 'research'}} if not user_id else {
                    'job_type': {'$eq': 'research'},
                    'user_id': {'$eq': user_id}
                }
            )
            
            jobs = []
            for match in response['matches']:
                job_data = match['metadata']
                job_data['score'] = match['score']
                jobs.append(job_data)
            
            # Sort by creation date
            jobs.sort(key=lambda x: x['created_at'], reverse=True)
            return jobs
            
        except Exception as e:
            print(f"Error fetching job history: {e}")
            return []
    
    def search_jobs_by_topic(self, search_topic: str, limit: int = 10) -> List[Dict]:
        """Search jobs by topic similarity"""
        try:
            query_embedding = self._generate_embedding(f"Research topic: {search_topic}")
            
            response = self.job_index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                namespace='jobs'
            )
            
            jobs = []
            for match in response['matches']:
                job_data = match['metadata']
                job_data['similarity_score'] = match['score']
                jobs.append(job_data)
            
            return jobs
            
        except Exception as e:
            print(f"Error searching jobs: {e}")
            return []

# Source and Content Management
class ContentManager(PineconeDataManager):
    """Manage research content and sources in Pinecone"""
    
    def store_source(self, job_id: str, url: str, title: str, content: str, 
                    credibility_score: float = 0.5) -> str:
        """Store research source in Pinecone"""
        source_id = f"{job_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        
        source_metadata = {
            'source_id': source_id,
            'job_id': job_id,
            'url': url,
            'title': title,
            'content_preview': content[:1000],  # First 1000 chars for metadata
            'credibility_score': credibility_score,
            'scraped_at': datetime.now().isoformat(),
            'content_length': len(content),
            'source_type': 'web_article'
        }
        
        # Generate embedding from content
        embedding_text = f"Title: {title} Content: {content}"
        embedding = self._generate_embedding(embedding_text)
        
        # Store in content index
        self.content_index.upsert(
            vectors=[{
                'id': source_id,
                'values': embedding,
                'metadata': source_metadata
            }],
            namespace=f"job_{job_id}"
        )
        
        return source_id
    
    def get_job_sources(self, job_id: str, limit: int = 10) -> List[Dict]:
        """Get all sources for a job"""
        try:
            # Query for sources in job namespace
            query_embedding = self._generate_embedding("research sources content")
            
            response = self.content_index.query(
                vector=query_embedding,
                top_k=limit,
                include_metadata=True,
                namespace=f"job_{job_id}"
            )
            
            sources = []
            for match in response['matches']:
                source_data = match['metadata']
                sources.append(source_data)
            
            return sources
            
        except Exception as e:
            print(f"Error fetching sources for job {job_id}: {e}")
            return []
    
    def search_content(self, query: str, job_id: str = None, top_k: int = 5) -> List[Dict]:
        """Search content using semantic similarity"""
        try:
            query_embedding = self._generate_embedding(query)
            
            namespace = f"job_{job_id}" if job_id else None
            
            response = self.content_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            results = []
            for match in response['matches']:
                content_data = match['metadata']
                content_data['relevance_score'] = match['score']
                results.append(content_data)
            
            return results
            
        except Exception as e:
            print(f"Error searching content: {e}")
            return []

# Database Operations Helper
class PineconeOperations:
    """Helper class for common Pinecone operations"""
    
    def __init__(self, api_key: str, environment: str, google_api_key: str = None):
        self.research_manager = ResearchJobManager(api_key, environment, google_api_key)
        self.content_manager = ContentManager(api_key, environment, google_api_key)
    
    def create_research_job(self, topic: str, user_id: str = 'anonymous') -> Dict[str, str]:
        """Create new research job"""
        return self.research_manager.create_job(topic, user_id)
    
    def update_research_job(self, job_id: str, status: str, **kwargs) -> bool:
        """Update research job status"""
        return self.research_manager.update_job_status(job_id, status, **kwargs)
    
    def get_job_history(self, user_id: str = None, limit: int = 10) -> List[Dict]:
        """Get combined job history"""
        research_jobs = self.research_manager.get_job_history(user_id, limit)
        
        # Combine and sort by creation date
        all_jobs = research_jobs
        all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return all_jobs[:limit]
    
    def cleanup_old_data(self, days_old: int = 7):
        """Clean up old job data"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # This would require custom implementation based on your cleanup strategy
        # Pinecone doesn't have built-in TTL, so you'd need to track and delete manually
        pass

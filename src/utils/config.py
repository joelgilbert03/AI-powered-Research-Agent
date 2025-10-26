import yaml
import pinecone
from typing import Dict, Any, List

class PineconeConfigManager:
    def __init__(self, config_path='config/pinecone_indexes.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.initialized_indexes = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load Pinecone configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default Pinecone configuration"""
        return {
            'indexes': {
                'jobs': {
                    'name': 'cognito-jobs',
                    'dimension': 768,
                    'metric': 'cosine',
                    'replicas': 1,
                    'shards': 1
                }
            },
            'namespaces': {
                'job_prefix': 'job_',
                'user_prefix': 'user_'
            },
            'cleanup': {
                'auto_cleanup': True,
                'retention_days': 30
            }
        }
    
    def initialize_pinecone(self, api_key: str, environment: str):
        """Initialize Pinecone with configuration"""
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create indexes if they don't exist

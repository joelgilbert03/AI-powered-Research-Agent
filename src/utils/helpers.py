import logging
from enum import Enum
from typing import Optional, Dict, Any
from src.pinecone_ops.operations import PineconeOperations

class ErrorType(Enum):
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    PARSING_ERROR = "parsing_error"
    PINECONE_ERROR = "pinecone_error"
    AGENT_ERROR = "agent_error"
    TIMEOUT_ERROR = "timeout_error"
    EMBEDDING_ERROR = "embedding_error"

class ResearchError(Exception):
    def __init__(self, error_type: ErrorType, message: str, details: Optional[Dict] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class PineconeErrorHandler:
    def __init__(self, pinecone_ops: PineconeOperations):
        self.pinecone_ops = pinecone_ops
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, job_id: str, error: Exception) -> Dict[str, Any]:
        """Handle and log errors, update job status in Pinecone"""
        if isinstance(error, ResearchError):
            error_info = {
                "type": error.error_type.value,
                "message": error.message,
                "details": error.details
            }
        else:
            error_info = {
                "type": "unknown_error",
                "message": str(error),
                "details": {"exception_type": type(error).__name__}
            }
        
        # Log error
        self.logger.error(f"Job {job_id} failed: {error_info}")
        
        # Update job status in Pinecone
        try:
            if job_id.startswith('sql_'):
                self.pinecone_ops.sql_manager.update_sql_job(
                    job_id=job_id,
                    status="error",
                    error_message=error_info["message"]
                )
            else:
                self.pinecone_ops.research_manager.update_job_status(
                    job_id=job_id,
                    status="error",
                    error_message=error_info["message"]
                )
        except Exception as update_error:
            self.logger.error(f"Failed to update job status: {update_error}")
        
        return error_info
    
    def get_user_friendly_message(self, error_type: ErrorType) -> str:
        """Get user-friendly error messages"""
        messages = {
            ErrorType.NETWORK_ERROR: "Unable to connect to external services. Please try again later.",
            ErrorType.API_ERROR: "External API is currently unavailable. Please try again later.",
            ErrorType.PARSING_ERROR: "Unable to process the retrieved content. Please try a different topic.",
            ErrorType.PINECONE_ERROR: "Vector database error. Please contact support.",
            ErrorType.AGENT_ERROR: "AI agent encountered an error. Please try again with a more specific topic.",
            ErrorType.TIMEOUT_ERROR: "Request timed out. Please try again or contact support.",
            ErrorType.EMBEDDING_ERROR: "Text embedding generation failed. Please try again."
        }
        return messages.get(error_type, "An unexpected error occurred. Please try again.")

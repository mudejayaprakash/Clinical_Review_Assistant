"""
Security manager for PHI protection and rate limiting
"""
import time
import logging
from functools import wraps
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityManager:
    """Manages security features including rate limiting, validation, and logging"""
    
    def __init__(self):
        self.rate_limits = {}
        logger.info("SecurityManager initialized")
    
    def rate_limit(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Rate limiting decorator to prevent abuse
        
        Args:
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                session_id = kwargs.get('session_id', 'default')
                now = time.time()
                
                # Initialize rate limit tracking for new sessions
                if session_id not in self.rate_limits:
                    self.rate_limits[session_id] = []
                
                # Remove old timestamps outside the window
                self.rate_limits[session_id] = [
                    t for t in self.rate_limits[session_id]
                    if now - t < window_seconds
                ]
                
                # Check if rate limit exceeded
                if len(self.rate_limits[session_id]) >= max_requests:
                    logger.warning(f"Rate limit exceeded for session: {session_id}")
                    raise Exception(f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds.")
                
                # Add current timestamp
                self.rate_limits[session_id].append(now)
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def validate_pdf(self, pdf_bytes: bytes) -> tuple[bool, str]:
        """
        Validate PDF file for security
        
        Args:
            pdf_bytes: PDF file as bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if bytes exist
        if not pdf_bytes or len(pdf_bytes) == 0:
            return False, "Empty file"
        
        # Check PDF header
        if not pdf_bytes.startswith(b'%PDF'):
            return False, "Not a valid PDF file"
        
        # Check file size (50MB limit)
        max_size = 50 * 1024 * 1024
        if len(pdf_bytes) > max_size:
            return False, f"File too large. Maximum size is 50MB"
        
        return True, ""
    
    def validate_criterion(self, criterion: str) -> tuple[bool, str]:
        """
        Validate user-provided criterion
        
        Args:
            criterion: Criterion text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not criterion or len(criterion.strip()) == 0:
            return False, "Criterion cannot be empty"
        
        if len(criterion) > 500:
            return False, "Criterion too long (max 500 characters)"
        
        return True, ""
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove PHI from metadata before logging
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Sanitized metadata without PHI
        """
        phi_keys = [
            'medical_records', 
            'summary', 
            'patient_name',
            'patient_id',
            'dob',
            'ssn',
            'mrn',
            'content',
            'record_chunks',
            'evaluation_results'
        ]
        
        safe_metadata = {
            k: v for k, v in metadata.items()
            if k.lower() not in phi_keys and not any(phi in k.lower() for phi in phi_keys)
        }
        
        return safe_metadata
    
    def log_action(self, action: str, user_id: str, metadata: dict = None):
        """
        Log user action with PHI protection
        
        Args:
            action: Action description
            user_id: User ID
            metadata: Additional metadata (will be sanitized)
        """
        if metadata is None:
            metadata = {}
        
        safe_metadata = self.sanitize_metadata(metadata)
        logger.info(f"Action: {action} | User: {user_id} | Metadata: {safe_metadata}")
    
    def validate_file_list(self, files: list) -> tuple[bool, str]:
        """
        Validate list of uploaded files
        
        Args:
            files: List of uploaded files
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not files or len(files) == 0:
            return False, "No files uploaded"
        
        if len(files) > 10:
            return False, "Too many files (maximum 10)"
        
        return True, ""


# Global security manager instance
security = SecurityManager()

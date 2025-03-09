import logging
from typing import Optional

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Logging level (defaults to INFO if None)
        
    Returns:
        Configured logger instance
    """
    if level is None:
        level = logging.INFO
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if it doesn't already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
import sys
import os, time
import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any
from loguru import logger as log
from datetime import timezone, timedelta

os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
time.tzset()

class Logging:
    """
    A utility class for handling logging operations using Loguru.
    Provides methods for configuring log outputs, setting log levels,
    and managing log rotation with UTC+7 timezone support.
    """
    def __init__(self, 
                 log_file: Optional[Union[str, Path]] = None,
                 log_level: str = "INFO",
                 rotation: str = "1 day",
                 retention: str = "1 week",
                 compression: str = "zip"):
        """
        Initialize logging configuration.
        
        Args:
            log_file: Path to log file. If None, logs only to stderr
            log_level: Minimum log level to record
            rotation: When to rotate the log file
            retention: How long to keep log files
            compression: Compression format for rotated logs
        """
        # Store logger instance
        self.log = log
        
        # Remove default logger
        self.log.remove()
        
        self.format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss} UTC+7</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Configure stderr logging with time converter
        self.log.add(
            sys.stderr,
            format=self.format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        
        # Configure file logging if path is provided
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.log.add(
                str(log_file),
                format=self.format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression=compression,
                enqueue=True,
                backtrace=True,
                diagnose=True,
            )
    
    def set_level(self, level: str) -> None:
        """Change the logging level."""
        self.log.level(level)
    
    def add_context(self, **kwargs: Any) -> None:
        """Add contextual information to log messages."""
        for key, value in kwargs.items():
            self.log.bind(**{key: value})
    
    def create_logger(self, name: str) -> log:
        """Create a named logger instance."""
        return self.log.bind(name=name)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log.debug(message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log.warning(message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log.error(message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log.critical(message, **kwargs)


# Testcase
# logger = Logging(
# )
# logger.debug("Hello world")
# logger.info("Info message")
# logger.warning("Warning message")
# logger.error("Error message")

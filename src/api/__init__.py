"""
Support Ticket Classification API Package

This package provides REST API endpoints for classifying support tickets
into departments (queue) and priority levels using Random Forest models.

Exports:
    - TicketClassifier: Main prediction class
    - app: FastAPI application instance
"""

from .predict import TicketClassifier
from .schemas import TicketRequest, TicketResponse, HealthResponse
from .preprocess import clean_text, get_top_keywords, generate_recommendation

__version__ = "2.0.0"
__author__ = "Aluka Precious Oluchukwu"

__all__ = [
    "TicketClassifier",
    "TicketRequest", 
    "TicketResponse",
    "HealthResponse",
    "clean_text",
    "get_top_keywords", 
    "generate_recommendation"
]
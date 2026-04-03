"""
Pydantic schemas for request and response validation.
These define the shape of data entering and leaving the API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

class TicketRequest(BaseModel):
    """
    Request schema for ticket classification.
    
    Example:
    {
        "text": "My payment was charged twice"
    }
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The support ticket text to classify",
        example="My payment was charged twice. Please refund the duplicate charge."
    )
    
    class Config:
       json_schema_extra = {
            "example": {
                "text": "My payment was charged twice. Please refund the duplicate charge."
            }
        }

class QueuePrediction(BaseModel):
    """
    Prediction result for queue (department).
    """
    department: str = Field(..., description="Predicted department name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")

class PriorityPrediction(BaseModel):
    """
    Prediction result for priority level.
    """
    level: str = Field(..., description="Predicted priority level: high, medium, or low")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")

class TicketResponse(BaseModel):
    """
    Response schema containing predictions, probabilities, and metadata.
    """
    # Unique identifier for this prediction
    ticket_reference: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique reference ID for this prediction"
    )
    
    # Timestamp of prediction
    assessment_date: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO format timestamp of when prediction was made"
    )
    
    # Main predictions
    queue: QueuePrediction = Field(..., description="Queue/department prediction")
    priority: PriorityPrediction = Field(..., description="Priority level prediction")
    
    # Explainability
    top_keywords: List[str] = Field(
        ...,
        description="Top 5 words that influenced the prediction",
        example=["payment", "charged", "refund", "duplicate", "transaction"]
    )
    
    # Business recommendation
    recommendation: str = Field(
        ...,
        description="Plain English action recommendation based on predictions",
        example="This ticket requires immediate attention. Route to Billing team within 2 hours."
    )
    
    # Full probability distributions
    all_queue_probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution across all departments"
    )
    all_priority_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all priority levels"
    )
    
    class Config:
       json_schema_extra = {
            "example": {
                "ticket_reference": "a3f2b9e1",
                "assessment_date": "2025-04-03T14:30:00",
                "queue": {
                    "department": "Billing and Payments",
                    "confidence": 0.92
                },
                "priority": {
                    "level": "high",
                    "confidence": 0.78
                },
                "top_keywords": ["payment", "charged", "refund", "duplicate", "transaction"],
                "recommendation": "This ticket requires immediate attention. Route to Billing team within 2 hours.",
                "all_queue_probabilities": {
                    "Billing and Payments": 0.92,
                    "Customer Service": 0.03,
                    "Technical Support": 0.02,
                    "Product Support": 0.01,
                    "IT Support": 0.01,
                    "Returns and Exchanges": 0.01,
                    "Service Outages and Maintenance": 0.00,
                    "Sales and Pre-Sales": 0.00,
                    "Human Resources": 0.00,
                    "General Inquiry": 0.00
                },
                "all_priority_probabilities": {
                    "high": 0.78,
                    "medium": 0.15,
                    "low": 0.07
                }
            }
        }

class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str = Field(..., description="API status: healthy or unhealthy")
    models_loaded: Dict[str, bool] = Field(..., description="Status of each model component")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "queue_model": True,
                    "priority_model": True,
                    "vectorizer": True
                }
            }
        }
"""
FastAPI application for Support Ticket Classification.
Provides REST API endpoints for predicting queue and priority.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List
import joblib
import os
from pathlib import Path

from .schemas import TicketRequest, TicketResponse, HealthResponse
from .predict import TicketClassifier

# Path to models directory
MODEL_PATH = Path(__file__).parent.parent.parent / "models"

# Global classifier instance
classifier = None


def load_models():
    """
    Load trained models and vectorizer from disk.
    
    Returns:
        TicketClassifier instance or None if loading fails
    """
    try:
        model_queue_path = MODEL_PATH / "model_queue.pkl"
        model_priority_path = MODEL_PATH / "model_priority.pkl"
        vectorizer_path = MODEL_PATH / "vectorizer.pkl"
        
        # Check if files exist
        if not model_queue_path.exists():
            raise FileNotFoundError(f"Queue model not found at {model_queue_path}")
        if not model_priority_path.exists():
            raise FileNotFoundError(f"Priority model not found at {model_priority_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
        
        # Load models
        print(f"Loading queue model from {model_queue_path}")
        model_queue = joblib.load(model_queue_path)
        
        print(f"Loading priority model from {model_priority_path}")
        model_priority = joblib.load(model_priority_path)
        
        print(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        
        # Create classifier
        classifier_instance = TicketClassifier(model_queue, model_priority, vectorizer)
        
        print("All models loaded successfully")
        return classifier_instance
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    Replaces deprecated on_event.
    """
    # Startup
    global classifier
    print("Starting up...")
    classifier = load_models()
    if classifier is None:
        print("Warning: Models failed to load. API will return 503 errors.")
    yield
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Support Ticket Classifier API",
    description="Classifies support tickets into department (queue) and priority level. Uses Random Forest models trained on 28,587 tickets.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Aluka Precious Oluchukwu",
    },
    license_info={
        "name": "MIT",
    }
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Support Ticket Classifier API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "/predict": "POST - Predict department and priority",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Verifies that all models are loaded and ready.
    """
    if classifier is None or not classifier.is_loaded:
        return HealthResponse(
            status="unhealthy",
            models_loaded={
                "queue_model": False,
                "priority_model": False,
                "vectorizer": False
            }
        )
    
    return HealthResponse(
        status="healthy",
        models_loaded=classifier.health_check()["models_loaded"]
    )


@app.post("/predict", response_model=TicketResponse, tags=["Prediction"])
async def predict_ticket(request: TicketRequest):
    """
    Predict department and priority for a support ticket.
    
    Args:
        request: TicketRequest containing the ticket text
        
    Returns:
        TicketResponse with predictions, probabilities, and recommendations
        
    Raises:
        HTTPException 503: If models are not loaded
        HTTPException 400: If input validation fails
        HTTPException 500: If prediction fails
    """
    # Check if models are loaded
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        # Make prediction
        result = classifier.predict(request.text)
        
        # Return response
        return TicketResponse(
            ticket_reference=result["ticket_reference"],
            assessment_date=result["assessment_date"],
            queue=result["queue"],
            priority=result["priority"],
            top_keywords=result["top_keywords"],
            recommendation=result["recommendation"],
            all_queue_probabilities=result["all_queue_probabilities"],
            all_priority_probabilities=result["all_priority_probabilities"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(tickets: List[TicketRequest]):
    """
    Predict for multiple tickets in one request.
    
    Args:
        tickets: List of TicketRequest objects
        
    Returns:
        List of predictions
    """
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    
    texts = [ticket.text for ticket in tickets]
    results = classifier.predict_batch(texts)
    
    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
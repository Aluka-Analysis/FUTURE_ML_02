"""
Prediction logic for ticket classification.
Handles model inference, keyword extraction, and recommendation generation.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Union
import numpy as np

from .preprocess import clean_text, get_top_keywords, generate_recommendation


class TicketClassifier:
    """
    Main classifier class that orchestrates prediction pipeline.
    
    Responsibilities:
    1. Load and hold models and vectorizer
    2. Process incoming text through preprocessing
    3. Generate predictions for queue and priority
    4. Extract explainability features (top keywords)
    5. Generate business recommendations
    """
    
    def __init__(self, model_queue, model_priority, vectorizer):
        """
        Initialize the classifier with trained models.
        
        Args:
            model_queue: Trained Random Forest model for queue/department
            model_priority: Trained Random Forest model for priority
            vectorizer: Fitted TF-IDF vectorizer
        """
        self.model_queue = model_queue
        self.model_priority = model_priority
        self.vectorizer = vectorizer
        self.is_loaded = all([model_queue, model_priority, vectorizer])
    
    def _generate_ticket_reference(self) -> str:
        """
        Generate a unique ticket reference ID.
        Format: TKT-XXXXXXXX (8 character uppercase alphanumeric)
        
        Returns:
            Ticket reference string
        """
        return f"TKT-{str(uuid.uuid4())[:8].upper()}"
    
    def _validate_text(self, text: str) -> None:
        """
        Validate input text length and content.
        
        Args:
            text: Ticket text to validate
            
        Raises:
            ValueError: If text is empty, too short, or invalid
        """
        if not text or len(text.strip()) == 0:
            raise ValueError(
                "Ticket text cannot be empty. Please provide the customer's message."
            )
        
        if len(text.strip()) < 10:
            raise ValueError(
                f"Ticket text too short ({len(text.strip())} characters). "
                "Please provide at least 10 characters for meaningful classification."
            )
        
        if len(text) > 10000:
            raise ValueError(
                f"Ticket text exceeds maximum length ({len(text)} > 10000 characters). "
                "Please truncate the message."
            )
    
    def predict(self, text: str) -> Dict:
        """
        Main prediction method.
        
        Args:
            text: Raw ticket text
            
        Returns:
            Dictionary containing:
            - ticket_reference: Unique ID (TKT-XXXXXXXX format)
            - assessment_date: ISO timestamp
            - queue: Department prediction with confidence
            - priority: Priority prediction with confidence
            - top_keywords: Words that influenced prediction
            - recommendation: Business action
            - all_queue_probabilities: Full distribution across departments
            - all_priority_probabilities: Full distribution across priorities
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Models not loaded. Please check model files and restart the service."
            )
        
        # Validate input
        self._validate_text(text)
        
        # Clean and preprocess text (once)
        cleaned_text = clean_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Get queue predictions
        queue_probs = self.model_queue.predict_proba(text_vectorized)[0]
        queue_idx = np.argmax(queue_probs)
        queue = self.model_queue.classes_[queue_idx]
        queue_confidence = float(queue_probs[queue_idx])
        
        # Get priority predictions
        priority_probs = self.model_priority.predict_proba(text_vectorized)[0]
        priority_idx = np.argmax(priority_probs)
        priority = self.model_priority.classes_[priority_idx]
        priority_confidence = float(priority_probs[priority_idx])
        
        # Get all probabilities as dictionaries
        all_queue_probs = dict(zip(self.model_queue.classes_, queue_probs.tolist()))
        all_priority_probs = dict(zip(self.model_priority.classes_, priority_probs.tolist()))
        
        # Sort descending for better readability
        all_queue_probs = dict(sorted(all_queue_probs.items(), key=lambda x: x[1], reverse=True))
        all_priority_probs = dict(sorted(all_priority_probs.items(), key=lambda x: x[1], reverse=True))
        
        # Extract top keywords (pass cleaned_text to avoid double processing)
        top_keywords = get_top_keywords(
            cleaned_text, 
            self.vectorizer, 
            self.model_queue, 
            top_n=5
        )
        
        # Generate business recommendation
        recommendation = generate_recommendation(queue, priority, queue_confidence)
        
        # Return complete response
        return {
            "ticket_reference": self._generate_ticket_reference(),
            "assessment_date": datetime.now().isoformat(),
            "queue": {
                "department": queue,
                "confidence": round(queue_confidence, 4)
            },
            "priority": {
                "level": priority,
                "confidence": round(priority_confidence, 4)
            },
            "top_keywords": top_keywords,
            "recommendation": recommendation,
            "all_queue_probabilities": {k: round(v, 4) for k, v in all_queue_probs.items()},
            "all_priority_probabilities": {k: round(v, 4) for k, v in all_priority_probs.items()}
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict for multiple tickets with individual error handling.
        
        Args:
            texts: List of ticket texts
            
        Returns:
            List of prediction dictionaries. Failed predictions include error information.
        """
        results = []
        for i, text in enumerate(texts):
            try:
                results.append(self.predict(text))
            except ValueError as e:
                results.append({
                    "ticket_reference": f"TKT-ERROR-{i+1}",
                    "error": str(e),
                    "error_type": "validation_error"
                })
            except RuntimeError as e:
                results.append({
                    "ticket_reference": f"TKT-ERROR-{i+1}",
                    "error": str(e),
                    "error_type": "model_error"
                })
            except Exception as e:
                results.append({
                    "ticket_reference": f"TKT-ERROR-{i+1}",
                    "error": f"Unexpected error: {str(e)}",
                    "error_type": "unexpected_error"
                })
        return results
    
    def health_check(self) -> Dict:
        """
        Check if all models are loaded and ready.
        
        Returns:
            Status dictionary with individual component status
        """
        return {
            "status": "healthy" if self.is_loaded else "unhealthy",
            "models_loaded": {
                "queue_model": self.model_queue is not None,
                "priority_model": self.model_priority is not None,
                "vectorizer": self.vectorizer is not None
            }
        }
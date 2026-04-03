"""
Text preprocessing utilities for ticket classification.
Matches the preprocessing done during model training.
"""

import re
import nltk
from nltk.corpus import stopwords
from typing import List

# Download stopwords if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Combined English and German stopwords
ENGLISH_STOPS = set(stopwords.words('english'))
GERMAN_STOPS = set(stopwords.words('german'))
STOPWORDS = ENGLISH_STOPS.union(GERMAN_STOPS)

# Additional custom stopwords identified during EDA (no duplicates)
CUSTOM_STOPS = {
    'please', 'thank', 'thanks', 'dear', 'regards', 'hello', 'hi',
    'would', 'could', 'kindly', 'assistance', 'help',
    'support', 'team', 'customer', 'user', 'service'
}
STOPWORDS = STOPWORDS.union(CUSTOM_STOPS)


def clean_text(text: str) -> str:
    """
    Clean and preprocess ticket text.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Remove standalone numbers
    4. Remove newline characters
    5. Remove extra whitespace
    6. Strip leading/trailing spaces
    
    Args:
        text: Raw ticket text (subject + body)
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Remove newline characters
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    # Remove extra whitespace (multiple spaces to single)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Cleaned text
        
    Returns:
        Text with stopwords removed
    """
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return ' '.join(filtered_words)


def get_top_keywords(text: str, vectorizer, model, top_n: int = 5) -> List[str]:
    """
    Extract top keywords that influenced the prediction.
    
    This uses the model's feature importances and the TF-IDF vectorizer
    to identify which words in the input text were most important
    for the prediction.
    
    Args:
        text: Raw ticket text
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained Random Forest model
        top_n: Number of top keywords to return
        
    Returns:
        List of top keywords
    """
    # Clean text AND remove stopwords (matches training preprocessing)
    cleaned = clean_text(text)
    cleaned = remove_stopwords(cleaned)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get non-zero features in this text
    non_zero_indices = vectorized.nonzero()[1]
    
    if len(non_zero_indices) == 0:
        return []
    
    # Get feature importances for these features
    importances = model.feature_importances_
    
    # Create list of (word, importance) pairs
    word_importance = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        importance = importances[idx]
        word_importance.append((word, importance))
    
    # Sort by importance and take top_n
    word_importance.sort(key=lambda x: x[1], reverse=True)
    
    return [word for word, _ in word_importance[:top_n]]


def generate_recommendation(queue: str, priority: str, confidence: float) -> str:
    """
    Generate business-friendly recommendation based on predictions.
    """
    # Lowered threshold from 0.7 to 0.5
    if confidence < 0.5:
        return f"Low confidence prediction ({confidence:.0%}). Flag for human review before routing to {queue}."
    
    # Auto-route departments (high precision)
    auto_route_depts = [
        'Billing and Payments',
        'Service Outages and Maintenance',
        'Technical Support',
        'IT Support',
        'Product Support'
    ]
    
    if queue in auto_route_depts:
        if priority == 'high':
            return f"High priority {queue} ticket. Route immediately to {queue} team. Response within 2 hours recommended."
        else:
            return f"Route to {queue} team for standard processing."
    
    else:
        if priority == 'high':
            return f"High priority ticket. Route to {queue} team with urgent flag."
        else:
            return f"Route to {queue} team for standard handling."
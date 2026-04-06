# FUTURE INTERNS ML TASK 02 — Support Ticket Classification System

Production-grade machine learning system for automated customer support ticket classification and priority routing.

**Author:** Aluka Precious Oluchukwu  
**Machine Learning Engineer**

**Live Demo**  
https://aluka-analysis.github.io/FUTURE_ML_02

**API Documentation**  
https://Aluka-P-support-ticket-api.hf.space/docs

**GitHub Repository**  
https://github.com/Aluka-Analysis/FUTURE_ML_02

---

## Project Overview

This system automatically classifies customer support tickets into departments and predicts priority levels.

A customer submits a complaint through the portal. A support agent receives:

• Department classification (Technical Support, Billing, HR, etc.)  
• Priority level (High / Medium / Low)  
• Confidence score  
• Key words that drove the decision  
• Business recommendation for routing  

The system uses Random Forest models with TF-IDF vectorization and explainable keyword extraction.

---

## Live Demo

Try the deployed system:

https://aluka-analysis.github.io/FUTURE_ML_02

The demo includes:
- **Customer Portal** — Submit tickets for instant classification
- **Agent Dashboard** — Real-time ticket stream with filters and auto-refresh

---

## System Architecture
```
Customer Message
│
▼
Frontend Interface (HTML/CSS/JS)
│
▼
FastAPI Inference Service
│
▼
Text Preprocessing
(Lowercase / Punctuation Removal / Stopwords)
│
▼
TF-IDF Vectorization
(10,000 features, 1-2 n-grams)
│
▼
Random Forest Classifiers
(Queue + Priority)
│
▼
Department + Priority + Confidence + Keywords
│
▼
Displayed instantly on Agent Dashboard

text
```
---

## The Problem This Solves

Position 345. Position 420.

Real queue positions from real banks. Customers waiting hours for support. Agents manually sorting tickets. Urgent issues delayed.

This system reduces manual sorting from 100% to 20% — routing tickets in under one second.

---

## Phases Completed

| Phase | Description |
|-------|-------------|
| Phase 1 | Project Setup and Data Loading |
| Phase 2 | Exploratory Data Analysis |
| Phase 3 | Text Preprocessing and Feature Engineering |
| Phase 4 | Model Training and Evaluation |
| Phase 5 | Explainability and Keyword Extraction |
| Phase 6 | FastAPI Inference Service |
| Phase 7 | Frontend Development (Customer + Agent Views) |
| Phase 8 | Docker Containerisation and Hugging Face Deployment |
| Phase 9 | GitHub Pages Deployment |
| Phase 10 | Model Card and Documentation |

---

## Model Performance

### Queue Classification (10 departments)

| Metric | Validation | Test |
|--------|------------|------|
| Weighted F1 | 0.5730 | 0.5692 |
| Macro F1 | — | 0.5330 |

**Best performing departments:**

| Department | Precision | Recall | F1 |
|------------|-----------|--------|-----|
| Billing and Payments | 0.92 | 0.67 | 0.78 |
| Service Outages | 0.86 | 0.56 | 0.68 |
| Technical Support | 0.48 | 0.90 | 0.62 |

### Priority Classification (3 levels)

| Metric | Validation | Test |
|--------|------------|------|
| Weighted F1 | 0.6419 | 0.6353 |
| Macro F1 | — | 0.6117 |

**Per-class performance:**

| Priority | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| High | 0.66 | 0.72 | 0.69 |
| Medium | 0.60 | 0.73 | 0.66 |
| Low | 0.82 | 0.34 | 0.48 |

### Critical Error Analysis

| Error | Rate | Business Impact |
|-------|------|-----------------|
| High → Low | 2.0% | Costly — urgent ticket delayed |
| Low → High | 22.5% | Less costly — non-urgent gets fast response |

---

## Key Visualisations

| Chart | Purpose |
|-------|---------|
| Queue Distribution | Shows class imbalance (20.6:1 ratio) |
| Priority Distribution | Shows urgency balance |
| Language Distribution | Shows multilingual mix (57% EN, 43% DE) |
| Confusion Matrix (Queue) | Shows department confusion patterns |
| Confusion Matrix (Priority) | Shows priority misclassification |
| Feature Importance | Shows top predictive words |

All visualisations are available in the `images/charts/` directory.

---

## Fairness and Responsible AI

The system handles class imbalance through:

- **Macro F1-score** — treats all departments equally
- **Class weights** — higher importance for minority classes
- **Stratified splitting** — preserves class distribution in train/test

Urgent keyword override ensures security and outage tickets receive High priority regardless of model prediction.

Low confidence predictions (<50%) are flagged for human review — the system is honest about uncertainty.

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Programming Language | Python 3.10 |
| Machine Learning | Scikit-learn (Random Forest) |
| Vectorization | TF-IDF (10,000 features, 1-2 n-grams) |
| Text Preprocessing | NLTK (English + German stopwords) |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Hugging Face Spaces, GitHub Pages |
| Version Control | Git, GitHub |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Multilingual Customer Support Tickets (Kaggle) |
| Size | 28,587 tickets |
| Languages | English (57%), German (43%) |
| Queue Classes | 10 departments |
| Priority Classes | 3 levels (high, medium, low) |

### Class Distribution

| Department | Tickets | Percentage |
|------------|---------|------------|
| Technical Support | 8,362 | 29.3% |
| Product Support | 5,252 | 18.4% |
| Customer Service | 4,268 | 14.9% |
| IT Support | 3,433 | 12.0% |
| Billing and Payments | 2,788 | 9.8% |
| Returns and Exchanges | 1,437 | 5.0% |
| Service Outages | 1,148 | 4.0% |
| Sales and Pre-Sales | 918 | 3.2% |
| Human Resources | 576 | 2.0% |
| General Inquiry | 405 | 1.4% |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | API information |
| /health | GET | System health check |
| /predict | POST | Single ticket classification |
| /tickets/receive | POST | Receive and auto-route ticket |
| /tickets | GET | Get all tickets for dashboard |
| /tickets/{id} | DELETE | Remove resolved ticket |
| /docs | GET | Interactive API documentation |

Interactive documentation is automatically generated through Swagger UI.

---

## Local Installation

### Clone repository

```bash
git clone https://github.com/Aluka-Analysis/FUTURE_ML_02.git
cd FUTURE_ML_02
Create virtual environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
bash
pip install -r requirements.txt
Run API locally
bash
python run.py
Then open:
```
text
http://localhost:8000
### Open frontend
## Open index.html in your browser.
```
## Running with Docker
### Build container

```bash
docker build -t support-ticket-system .
```
### Run system
```bash
docker run -p 7860:7860 support-ticket-system
```
### Then open:

```
text
http://localhost:7860
```
## Project Structure
text
```
FUTURE_ML_02/
│
├── index.html                    # Frontend (Customer + Agent views)
├── run.py                        # API launcher
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── README.md                     # This file
├── MODEL_CARD.md                 # Detailed model documentation
│
├── src/api/                      # Backend source code
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic models
│   ├── predict.py                # Prediction logic
│   └── preprocess.py             # Text preprocessing
│
├── models/                       # Trained models
│   ├── model_queue.pkl           # Random Forest (240 MB)
│   ├── model_priority.pkl        # Random Forest (121 MB)
│   └── vectorizer.pkl            # TF-IDF vectorizer
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   └── 02_models_training.ipynb  # Model training
│
├── images/charts/                # Visualizations
│   ├── queue_distribution.png
│   ├── priority_distribution.png
│   ├── language_distribution.png
│   ├── confusion_matrix_queue.png
│   ├── confusion_matrix_priority.png
│   └── feature_importance_queue.png
│
└── data/processed/               # Cleaned dataset
    └── cleaned_tickets.csv
```
### Check model_card.md for more information

Author
Aluka Precious Oluchukwu
Machine Learning Engineer

LinkedIn
https://www.linkedin.com/in/aluka-precious-b222a2356

GitHub
https://github.com/Aluka-Analysis

Email
alukaprecious4@gmail.com


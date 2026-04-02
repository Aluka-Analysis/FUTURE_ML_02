
# Model Card: Support Ticket Classification

## Model Overview
- **Task**: Automatically classify support tickets into department (queue) and priority level
- **Models**: Random Forest (n_estimators=100, class_weight='balanced')
- **Training Date**: 2026-04-02
- **Track**: ML | Task 2

## Dataset
- **Total tickets**: 28,587
- **Training set**: 20,010 (70%)
- **Validation set**: 4,288 (15%)
- **Test set**: 4,289 (15%)
- **Classes**: 10 departments, 3 priority levels

## Preprocessing
- **Text cleaning**: Lowercase, remove newlines, extra spaces
- **Stopwords**: 424 combined English + German
- **TF-IDF**: 10,000 features, ngram_range=(1,2), min_df=2, max_df=0.95

## Performance - Queue Classification

| Metric | Validation | Test |
|--------|------------|------|
| Weighted F1 | 0.5730 | 0.5692 |
| Macro F1 | - | 0.5330 |

**Best Departments:**
- Billing and Payments: F1 0.78
- Service Outages: F1 0.68

**Struggles:**
- General Inquiry: F1 0.39 (limited data)
- Human Resources: F1 0.41

## Performance - Priority Classification

| Metric | Validation | Test |
|--------|------------|------|
| Weighted F1 | 0.6419 | 0.6353 |
| Macro F1 | - | 0.6117 |

**Per-Class Performance:**
- High priority: F1 0.69 (recall 72%)
- Medium priority: F1 0.66 (recall 73%)
- Low priority: F1 0.48 (recall 34%)

**Critical Errors:**
- High → Low: 2.0% (costly, acceptable)
- Low → High: 22.5% (less costly)

## Top Features
1. billing (0.0131)
2. service (0.0059)
3. payment (0.0042)
4. software (0.0034)
5. employee (0.0034)

## Limitations
- Minority departments (General Inquiry, HR) have low recall
- High priority tickets occasionally misclassified as low (2%)
- German stopwords applied but German-specific performance not separately validated

## Deployment Files
- `models/model_queue.pkl` (Random Forest)
- `models/model_priority.pkl` (Random Forest)
- `models/vectorizer.pkl` (TF-IDF)

## Business Recommendations
1. **Auto-route**: Billing, Service Outages (high confidence)
2. **Human review**: General Inquiry, HR (low volume, low recall)
3. **Confidence threshold**: Flag predictions <0.7 for review
4. **Monitor**: High→Low misclassifications in production

## Author
Aluka Precious Oluchukwu

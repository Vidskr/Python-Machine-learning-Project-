# Predicting Hospital Readmission Using Machine Learning

##  Project Overview
This project predicts **30-day hospital readmission for diabetic patients** using machine learning models.  

- Identify high-risk patients  
- Understand key contributing factors  

The workflow demonstrates a progression from **basic Python and exploratory data analysis (EDA)** to **advanced machine learning models and neural networks**.

## Research Question
**Can we predict which patients are at higher risk of 30-day hospital readmission, and what patient or hospital factors contribute the most?**

This question enables:
- Exploration from baseline to advanced ML techniques  
- Fairness and equity analysis across demographic groups  
- Policy-relevant interpretation and causal reasoning  

## Dataset

**Source:** UCI – *Diabetes 130-US Hospitals for Years 1999–2008*  
**Size:** ~100,000 hospital encounters

### Features
- **Demographics:** age, race, gender  
- **Diagnoses:** ICD-9 codes (primary and secondary)  
- **Utilization:** number of lab procedures, medications  
- **Hospital Features:** inpatient and outpatient visits  
- **Treatment:** diabetes medications, medication changes  
- **Outcome:** hospital readmission status  

### Why This Dataset?
- Large and diverse population  
- Mixed numeric and categorical variables  
- Direct relevance to healthcare quality and policy decisions  

## Target Variable
We predict **hospital readmission within 30 days**.

| Original Value | Encoded |
|---------------|---------|
| NO            | 0       |
| <30           | 1       |
| >30           | 0       |

This binary encoding allows the use of **standard classification models**.

---

## Project Goals

### Technical Goals
This repository demonstrates a complete **machine learning pipeline**:

- **Python & EDA:** Data loading, inspection, visualization  
- **Preprocessing:** Encoding, scaling, missing value handling  
- **Baseline ML:** Logistic Regression, Random Forest, Gradient Boosting  
- **Advanced ML:** Tuned Random Forest, L1 Logistic Regression  
- **Neural Networks:** Multi-Layer Perceptron (MLP)  
- **Evaluation:** Accuracy, ROC-AUC, classification reports

---

## Phase 1: Project Definition & Dataset Understanding

**Notebook:** `00_data_understanding.ipynb`
### Sections
- **Load Data:** pandas, CSV loading, dataset shape  
- **Basic Inspection:** `.head()`, `.info()`, `.describe()`  
- **Feature Types:** categorical vs numeric  
- **Missing Values:** count and preprocessing needs  
- **Target Exploration:** class balance and distribution  
- **Visualizations:**  
  - Readmission by age  
  - Readmission by race  
  - Readmission by gender  
  - Readmission by time in hospital  

---

## Phase 2–5: Machine Learning Workflow

### Phase 2: Preprocessing
- Encode categorical variables  
- Scale numeric features  
- Train-test split  
---
### Phase 3: Baseline Models
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

**Evaluation Metrics:**
- Accuracy  
- ROC-AUC  
- Classification report  
---

### Phase 4: Advanced Models
- L1-regularized Logistic Regression  
- Tuned Random Forest  
---
### Phase 5: Neural Networks
- Multi-Layer Perceptron (MLP) for classification  
- Training loss tracking  
- Performance evaluation using standard metrics  

Machine Learning | Healthcare Analytics | Policy-Oriented Data Science

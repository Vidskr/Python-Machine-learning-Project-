# Predicting Hospital Readmission Using Machine Learning: A Health & Policy Perspective
Project Overview
This project predicts 30-day hospital readmission for diabetic patients using machine learning models. It combines healthcare data analysis with computational social science insights to identify high-risk patients, understand key contributing factors, and explore potential policy and fairness implications.
The workflow demonstrates a progression from basic Python and data exploration to advanced machine learning models and neural networks.

Research Question
Can we predict which patients are at higher risk of 30-day hospital readmission, and what patient or hospital factors contribute the most?

This question allows:
Exploration of baseline to advanced machine learning techniques
Fairness and equity analysis across demographics
Policy-relevant interpretation and causal reasoning

Dataset
Dataset: UCI “Diabetes 130-US Hospitals for years 1999–2008”
Size: ~100,000 hospital encounters
Features:

Demographics: age, race, gender
Diagnoses: ICD-9 codes for primary and secondary diagnoses
Utilization: number of lab procedures, medications
Hospital features: number of inpatient/outpatient visits
Treatment: diabetes medications, medication changes
Outcome: readmitted (binary: yes/no)

Why this dataset?
Large and diverse
Mixed numeric and categorical features
Directly relevant to healthcare quality and policy decisions

Target Variable
We predict hospital readmission within 30 days:
Original Value	Encoded
| Original Value | Encoded |
| -------------- | ------- |
| NO             | 0       |
| <30            | 1       |
| >30            | 0       |


This binary target allows standard ML models for classification.

Project Goals
Technical Goals

The repository showcases a full machine learning pipeline:

Python & EDA: Data loading, inspection, and visualization

Preprocessing: Encoding, scaling, missing value handling

Baseline ML: Logistic Regression, Random Forest, Gradient Boosting

Advanced ML: Tuned Random Forest, L1 Logistic Regression

Neural Networks: MLPClassifier

Model Evaluation: Accuracy, ROC-AUC, classification reports

Policy & Healthcare Goals

Identify patient groups with higher readmission probability

Inform interventions and resource allocation

Examine fairness across age, gender, and race

Phase 1: Project Definition & Dataset Understanding

Notebook: 00_data_understanding.ipynb

Sections:

Load Data: pandas, load CSV, inspect shape

Basic Data Inspection: .head(), .info(), .describe(), list categorical vs numeric

Missing Values: count missing, inspect columns requiring preprocessing

Explore Target: proportions of readmitted = yes/no, class imbalance, plots

Simple Visualizations: readmission rate by age, race, gender, time in hospital

Phase 2–5: Machine Learning Workflow
Phase 2: Preprocessing

Encode categorical variables

Scale numeric variables

Split data into train/test sets

Phase 3: Baseline Models

Logistic Regression

Random Forest

Gradient Boosting

Evaluate with accuracy, ROC-AUC, and classification report

Phase 4: Advanced Models

L1-regularized Logistic Regression

Tuned Random Forest

Optionally: XGBoost, LightGBM, CatBoost (if installed)

Phase 5: Neural Networks

Multi-Layer Perceptron (MLP) for classification

Track training loss and evaluate metrics

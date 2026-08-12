# CodeAlpha_Credit-Scoring-Model-
# Credit Scoring Model

## Project Overview

This project predicts an individual's credit risk using historical
financial information and machine learning classification algorithms.

The project uses the Give Me Some Credit dataset.

The objective is to identify whether an individual is likely to
experience serious financial distress within the next two years.

---

## Objective

The main objective of this project is to build a machine learning
classification model that can identify customers with different
levels of financial risk.

The project focuses on:

- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Classification
- Model comparison
- Model evaluation
- Credit risk prediction

---

## Dataset

The project uses the Give Me Some Credit dataset.

The training dataset contains financial and demographic information
such as:

- Age
- Monthly income
- Debt ratio
- Revolving credit utilization
- Number of open credit lines
- Number of real estate loans
- Number of dependents
- Past-due payment history

### Target Variable

`SeriousDlqin2yrs`

The target represents whether the person experienced serious
financial delinquency within two years.

- `0` = No serious financial distress
- `1` = Serious financial distress

---

## Features Used

The model uses the following features:

- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

Additional engineered features:

- MonthlyIncomePerDependent
- TotalPastDue
- TotalOpenCreditLines

---

## Machine Learning Algorithms

Three classification algorithms are compared:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model.

### 2. Decision Tree

A tree-based classification algorithm that makes decisions using
feature-based rules.

### 3. Random Forest

An ensemble algorithm that combines multiple decision trees to
improve prediction performance.

---

## Feature Engineering

Two important features were created:

### Monthly Income Per Dependent

This represents the monthly income available relative to the number
of dependents.

### Total Past Due

This combines different categories of late payments:

- 30–59 days late
- 60–89 days late
- 90+ days late

These engineered features help the model understand an individual's
financial behavior more effectively.

---

## Data Preprocessing

The project performs:

1. Missing-value handling
2. Duplicate removal
3. Feature engineering
4. Train-test splitting
5. Feature scaling

Missing numerical values are replaced using the median.

---

## Model Evaluation

The models are evaluated using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many predicted high-risk customers were actually
high-risk.

### Recall

Measures how many actual high-risk customers were correctly
identified.

### F1-Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures the model's ability to distinguish between different
risk classes.

---

## Visualizations

The project includes:

- Target distribution
- Correlation heatmap
- Model comparison chart
- Confusion matrix
- ROC curve
- Feature importance chart

---

## Project Workflow

```text
Dataset
   |
   v
Data Cleaning
   |
   v
Exploratory Data Analysis
   |
   v
Feature Engineering
   |
   v
Train/Test Split
   |
   v
Feature Scaling
   |
   v
Machine Learning Models
   |
   +--> Logistic Regression
   |
   +--> Decision Tree
   |
   +--> Random Forest
   |
   v
Model Comparison
   |
   v
Model Evaluation
   |
   +--> Accuracy
   +--> Precision
   +--> Recall
   +--> F1-Score
   +--> ROC-AUC
   |
   v
Best Model
   |
   v
Credit Risk Prediction

# ============================================
# CREDIT SCORING MODEL
# ============================================

# Import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("credit_data.csv")

print("Dataset loaded successfully!")
print()

print("First 5 rows:")
print(df.head())

print()
print("Dataset shape:", df.shape)


# ============================================
# 2. DATA UNDERSTANDING
# ============================================

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ============================================
# 3. CHECK MISSING VALUES
# ============================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================
# 4. CHECK DUPLICATES
# ============================================

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()


# ============================================
# 5. FEATURE ENGINEERING
# ============================================

# Debt-to-Income Ratio
# This tells us how large the person's debt is
# compared with their income.

df["debt_to_income"] = (
    df["debt"] / df["income"]
)

# Convert payment history into a percentage-based
# feature.

df["payment_reliability"] = (
    df["payment_history"] / 100
)

print("\nNew Features Created:")
print("debt_to_income")
print("payment_reliability")


# ============================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================

# Creditworthiness distribution

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="credit_score"
)

plt.title("Creditworthiness Distribution")

plt.xlabel("Creditworthy")

plt.ylabel("Number of Customers")

plt.show()


# ============================================
# 7. CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()


# ============================================
# 8. DEFINE FEATURES AND TARGET
# ============================================

features = [
    "income",
    "debt",
    "payment_history",
    "credit_utilization",
    "age",
    "loan_count",
    "debt_to_income",
    "payment_reliability"
]

X = df[features]

y = df["credit_score"]


# ============================================
# 9. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================
# 10. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================
# 11. CREATE MODELS
# ============================================

models = {

    "Logistic Regression":
        LogisticRegression(),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


# ============================================
# 12. TRAIN AND EVALUATE MODELS
# ============================================

results = []

trained_models = {}

for name, model in models.items():

    print("\nTraining:", name)

    # Train model

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predictions

    predictions = model.predict(
        X_test_scaled
    )

    # Probability predictions

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # Calculate metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Store results

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })

    trained_models[name] = model


# ============================================
# 13. MODEL COMPARISON
# ============================================

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df)


# ============================================
# 14. VISUALIZE MODEL PERFORMANCE
# ============================================

results_df.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Credit Scoring Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()


# ============================================
# 15. SELECT BEST MODEL
# ============================================

best_model_name = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\nBest Model:", best_model_name)


# ============================================
# 16. DETAILED EVALUATION
# ============================================

best_predictions = best_model.predict(
    X_test_scaled
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# ============================================
# 17. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================
# 18. ROC CURVE
# ============================================

best_probabilities = best_model.predict_proba(
    X_test_scaled
)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    best_probabilities
)

auc_score = roc_auc_score(
    y_test,
    best_probabilities
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# ============================================
# 19. FEATURE IMPORTANCE
# ============================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "Feature": features,

        "Importance":
            best_model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:")

    print(importance)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance"
    )

    plt.show()


# ============================================
# 20. SAMPLE CREDIT PREDICTION
# ============================================

sample_customer = pd.DataFrame({

    "income": [60000],

    "debt": [10000],

    "payment_history": [95],

    "credit_utilization": [25],

    "age": [32],

    "loan_count": [2],

    "debt_to_income": [
        10000 / 60000
    ],

    "payment_reliability": [
        95 / 100
    ]
})


sample_scaled = scaler.transform(
    sample_customer
)


prediction = best_model.predict(
    sample_scaled
)[0]


probability = best_model.predict_proba(
    sample_scaled
)[0][1]


print("\n===================================")

print("CREDIT SCORING RESULT")

print("===================================")

if prediction == 1:

    print(
        "Creditworthiness: CREDITWORTHY"
    )

else:

    print(
        "Creditworthiness: NOT CREDITWORTHY"
    )


print(
    f"Creditworthiness Probability: "
    f"{probability * 100:.2f}%"
)

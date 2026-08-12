# ============================================================
# CREDIT SCORING MODEL
# ============================================================
#
# Objective:
# Predict whether an individual is likely to experience
# financial distress using historical financial information.
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
#
# Evaluation:
# Accuracy
# Precision
# Recall
# F1-Score
# ROC-AUC
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
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

import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Load the training dataset.
df = pd.read_csv("cs-training.csv")

print("=" * 60)
print("CREDIT SCORING MODEL")
print("=" * 60)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

# The dataset contains an unnecessary unnamed index column.
# Remove it if it exists.

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows.

df = df.drop_duplicates()

print("Shape after removing duplicates:")
print(df.shape)


# ============================================================
# 7. TARGET VARIABLE
# ============================================================

# SeriousDlqin2yrs is the target variable.
#
# 0 = No serious financial distress
# 1 = Serious financial distress

target = "SeriousDlqin2yrs"

print("\nTarget Distribution:")
print(df[target].value_counts())

print("\nTarget Percentage:")
print(
    df[target].value_counts(normalize=True) * 100
)


# ============================================================
# 8. TARGET DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=target
)

plt.title("Financial Distress Distribution")

plt.xlabel(
    "Financial Distress Within 2 Years"
)

plt.ylabel("Number of Customers")

plt.show()


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

# Debt-to-income related ratio is already represented by
# DebtRatio in the dataset.
#
# We create additional useful features.

df["MonthlyIncomePerDependent"] = (
    df["MonthlyIncome"] /
    (df["NumberOfDependents"].fillna(0) + 1)
)

df["TotalPastDue"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"] +
    df["NumberOfTime60-89DaysPastDueNotWorse"] +
    df["NumberOfTimes90DaysLate"]
)

df["TotalOpenCreditLines"] = (
    df["NumberOfOpenCreditLinesAndLoans"]
)

print("\nNew Features Created:")
print("- MonthlyIncomePerDependent")
print("- TotalPastDue")
print("- TotalOpenCreditLines")


# ============================================================
# 10. SELECT FEATURES
# ============================================================

features = [

    "RevolvingUtilizationOfUnsecuredLines",

    "age",

    "NumberOfTime30-59DaysPastDueNotWorse",

    "DebtRatio",

    "MonthlyIncome",

    "NumberOfOpenCreditLinesAndLoans",

    "NumberOfTimes90DaysLate",

    "NumberRealEstateLoansOrLines",

    "NumberOfTime60-89DaysPastDueNotWorse",

    "NumberOfDependents",

    "MonthlyIncomePerDependent",

    "TotalPastDue",

    "TotalOpenCreditLines"
]


X = df[features]

y = df[target]


print("\nNumber of Features:", len(features))

print("\nFeatures:")
for feature in features:
    print("-", feature)


# ============================================================
# 11. HANDLE MISSING VALUES
# ============================================================

# Replace missing numerical values with the median.

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)


# ============================================================
# 12. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_imputed,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# 13. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 14. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
}


# ============================================================
# 15. TRAIN MODELS
# ============================================================

results = []

trained_models = {}

probability_predictions = {}

class_predictions = {}


for name, model in models.items():

    print("\nTraining:", name)

    # Train model.

    model.fit(
        X_train_scaled,
        y_train
    )

    # Predict classes.

    predictions = model.predict(
        X_test_scaled
    )

    # Predict probabilities.

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # Calculate evaluation metrics.

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Store results.

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc

    })

    trained_models[name] = model

    probability_predictions[name] = probabilities

    class_predictions[name] = predictions


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 17. MODEL COMPARISON GRAPH
# ============================================================

results_plot = results_df.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
]

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Credit Scoring Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 18. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_predictions = class_predictions[
    best_model_name
]

best_probabilities = probability_predictions[
    best_model_name
]


print("\n" + "=" * 60)

print("BEST MODEL")

print("=" * 60)

print(
    "Best Model:",
    best_model_name
)

print(
    "ROC-AUC:",
    round(
        results_df.iloc[0]["ROC-AUC"],
        4
    )
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "No Financial Distress",
            "Financial Distress"
        ],
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Distress",
        "Distress"
    ],
    yticklabels=[
        "No Distress",
        "Distress"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================================
# 21. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    best_probabilities
)

auc_score = roc_auc_score(
    y_test,
    best_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc_score:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.show()


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance":
            best_model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 60)

    print("FEATURE IMPORTANCE")

    print("=" * 60)

    print(
        importance_df.to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Feature Importance - {best_model_name}"
    )

    plt.show()


# ============================================================
# 23. SAMPLE CUSTOMER PREDICTION
# ============================================================

# Example customer information.

sample_customer = pd.DataFrame({

    "RevolvingUtilizationOfUnsecuredLines":
        [0.25],

    "age":
        [35],

    "NumberOfTime30-59DaysPastDueNotWorse":
        [0],

    "DebtRatio":
        [0.30],

    "MonthlyIncome":
        [5000],

    "NumberOfOpenCreditLinesAndLoans":
        [5],

    "NumberOfTimes90DaysLate":
        [0],

    "NumberRealEstateLoansOrLines":
        [1],

    "NumberOfTime60-89DaysPastDueNotWorse":
        [0],

    "NumberOfDependents":
        [2],

    "MonthlyIncomePerDependent":
        [5000 / 3],

    "TotalPastDue":
        [0],

    "TotalOpenCreditLines":
        [5]
})


# Apply the same preprocessing.

sample_imputed = imputer.transform(
    sample_customer
)

sample_scaled = scaler.transform(
    sample_imputed
)


# Make prediction.

prediction = best_model.predict(
    sample_scaled
)[0]

probability = best_model.predict_proba(
    sample_scaled
)[0][1]


# ============================================================
# 24. FINAL PREDICTION
# ============================================================

print("\n" + "=" * 60)

print("CREDIT RISK PREDICTION")

print("=" * 60)

if prediction == 1:

    print(
        "Prediction: HIGHER FINANCIAL RISK"
    )

else:

    print(
        "Prediction: LOWER FINANCIAL RISK"
    )


print(
    f"Predicted Risk Probability: "
    f"{probability * 100:.2f}%"
)

print("=" * 60)

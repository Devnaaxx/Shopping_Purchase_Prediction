# ==========================================================
# PROJECT: SHOPPING ADDICTION & PURCHASE PREDICTION
# MACHINE LEARNING MODEL TRAINING
# ==========================================================


# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# 2. LOAD DATASET

df = pd.read_csv("online_shoppers_intention.csv")

print("\nDATASET LOADED SUCCESSFULLY!")
print("Dataset Shape:", df.shape)

print("\nFIRST FIVE ROWS")
print(df.head())


# 3. DATA PREPROCESSING

# Remove duplicate records
df = df.drop_duplicates()

# Convert target column into numerical values
df["Revenue"] = df["Revenue"].astype(int)

# Separate features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("\nMISSING VALUES:")
print(X.isnull().sum().sum())


# 4. IDENTIFY COLUMNS

categorical_features = [
    "Month",
    "VisitorType",
    "Weekend"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# 5. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)


# 6. CREATE MACHINE LEARNING MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )
}


# 7. TRAIN AND EVALUATE MODELS

results = []
trained_models = {}

for name, algorithm in models.items():

    print("\nTraining:", name)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             categorical_features)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[name] = pipeline

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))
    print("ROC AUC:", round(auc, 3))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))


# 8. MODEL COMPARISON

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))


# 9. ACCURACY COMPARISON GRAPH

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()


# 10. SELECT BEST MODEL

best_model_name = results_df.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)


# 11. CONFUSION MATRIX

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Purchase", "Purchase"],
    yticklabels=["No Purchase", "Purchase"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()


# 12. SAVE BEST MODEL

os.makedirs("model", exist_ok=True)

joblib.dump(
    best_model,
    "model/best_shopping_model.pkl"
)

joblib.dump(
    best_model_name,
    "model/best_model_name.pkl"
)

joblib.dump(
    results_df,
    "model/model_results.pkl"
)


# 13. FINAL OUTPUT

print("\n======================================")
print("SHOPPING PURCHASE PREDICTION")
print("======================================")

print("Dataset:", df.shape)
print("Best Model:", best_model_name)

print("\nMODEL SAVED SUCCESSFULLY!")
print("Training Completed!")

print("======================================")
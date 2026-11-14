"""
Model selection and evaluation on the engineered Titanic features.

Steps:
1. Load the engineered dataset and split it 80/20
2. Train three models: Logistic Regression, Decision Tree, Random Forest
3. Evaluate with accuracy, precision, recall and F1-score
4. Use 5-fold cross-validation to check for overfitting
5. Compare the models and pick the best one
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "data/engineered_features.csv"
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Survived")
    y = df["Survived"]

    # 80% to train, 20% to test (stratify keeps the survivor ratio equal in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # The models to compare. Logistic Regression is scaled inside a pipeline,
    # so the scaler only learns from the training part of each fold.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    # 5-fold cross-validation: train on 4 parts, test on the 5th, repeat 5 times
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append(
            {
                "Model": name,
                "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
                "CV Accuracy": cv_scores.mean(),
                "CV Std": cv_scores.std(),
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-score": f1_score(y_test, y_pred),
            }
        )

    results = pd.DataFrame(results).set_index("Model").round(3)
    print(results.to_string())

    # Pick the model with the best cross-validated accuracy
    best = results["CV Accuracy"].idxmax()
    print("\nBest model (highest CV accuracy):", best)

    # Save the table and a comparison chart
    results.to_csv("data/model_comparison.csv")
    os.makedirs("images", exist_ok=True)
    results[["CV Accuracy", "Test Accuracy", "F1-score"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("images/model_comparison.png")


if __name__ == "__main__":
    main()

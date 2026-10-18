
"""Train and evaluate classifiers for the educational mRNA-VIP project."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from data_pipeline import load_patient_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42
TARGET = "Reaction_Score"

NUMERIC_FEATURES = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
]
CATEGORICAL_FEATURES = ["Prior_Reaction_History"]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing steps fitted only on the training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a fitted classifier and return its key metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "predictions": predictions,
        "probabilities": probabilities,
        "report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }


def train_and_evaluate() -> None:
    """Train two classifiers, compare them, and save results."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_patient_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Compare candidate models using training data only.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_f1_scores = {}

    for name, classifier in candidates.items():
        cv_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        scores = cross_val_score(
            cv_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )

        cv_f1_scores[name] = scores.mean()

        print(
            f"{name} — 5-fold CV F1: "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )

    best_name = max(cv_f1_scores, key=cv_f1_scores.get)
    print(f"\nSelected by cross-validation: {best_name}")


    results = {}
    fitted_models = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[name] = metrics
        fitted_models[name] = pipeline

        print(f"\n{'=' * 60}")
        print(name)
        print(f"F1-score: {metrics['f1']:.3f}")
        print(f"ROC-AUC:  {metrics['roc_auc']:.3f}")
        print(metrics["report"])

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            metrics["predictions"],
            cmap="Blues",
            colorbar=False,
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(
            FIGURE_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=150,
        )
        plt.close()

    # Save ROC curves for both models.
    plt.figure(figsize=(8, 6))

    for name, metrics in results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, metrics["probabilities"]
        )
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {metrics['roc_auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve.png", dpi=200)
    plt.close()

    comparison = pd.DataFrame(
        {
            name: {
                "F1-score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
            for name, metrics in results.items()
        }
    ).T
    comparison.index.name = "Model"
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")

        # Select by F1-score, as specified in the project handout.
    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model_path = MODEL_DIR / "best_risk_model.joblib"
    joblib.dump(fitted_models[best_name], best_model_path)
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(comparison.round(3).to_string())
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {best_model_path}")
    print(f"Saved ROC curve: {FIGURE_DIR / 'roc_curve.png'}")
    print("\nEducational prototype only; not for clinical decision-making.")


if __name__ == "__main__":
    train_and_evaluate()

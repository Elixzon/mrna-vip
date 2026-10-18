
"""Load the trained model for educational mRNA-VIP predictions."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_risk_model.joblib"

FEATURES = [
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
]


def predict_reaction_risk(features: dict) -> dict:
    """Return a dataset-model prediction for educational demonstration only."""

    missing = [name for name in FEATURES if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Load the model saved by risk_model.py.
    model = joblib.load(MODEL_PATH)

    # Preserve the exact feature names and training order.
    input_df = pd.DataFrame(
        [{name: features[name] for name in FEATURES}],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    positive_index = classes.index(1)
    probability_class_1 = float(probabilities[positive_index])

    return {
        "predicted_class": prediction,
        "class_1_probability": probability_class_1,
        "model": type(model.named_steps["classifier"]).__name__,
        "disclaimer": (
            "Experimental dataset-model output only. Not a clinical risk "
            "assessment, vaccine-safety determination, or medical advice."
        ),
    }

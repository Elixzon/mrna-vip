
import pytest

from src.risk_predictor import FEATURES, predict_reaction_risk


def test_required_features_are_defined():
    assert FEATURES == [
        "Age",
        "Biomarker_Level",
        "Pre_Existing_Conditions",
        "Vaccine_Dose_mcg",
        "Prior_Reaction_History",
    ]


def test_missing_features_raise_value_error():
    with pytest.raises(ValueError, match="Missing"):
        predict_reaction_risk({"Age": 30})


def test_prediction_returns_expected_fields():
    sample = {
        "Age": 30,
        "Biomarker_Level": 1.2,
        "Pre_Existing_Conditions": 0,
        "Vaccine_Dose_mcg": 50,
        "Prior_Reaction_History": "No",
    }

    result = predict_reaction_risk(sample)

    assert result["predicted_class"] in (0, 1)
    assert 0.0 <= result["class_1_probability"] <= 1.0
    assert result["model"] == "LogisticRegression"
    assert "not a clinical risk assessment" in result["disclaimer"].lower()


def test_prediction_is_reproducible():
    sample = {
        "Age": 30,
        "Biomarker_Level": 1.2,
        "Pre_Existing_Conditions": 0,
        "Vaccine_Dose_mcg": 50,
        "Prior_Reaction_History": "No",
    }

    first = predict_reaction_risk(sample)
    second = predict_reaction_risk(sample)

    assert first["predicted_class"] == second["predicted_class"]
    assert first["class_1_probability"] == pytest.approx(
        second["class_1_probability"]
    )

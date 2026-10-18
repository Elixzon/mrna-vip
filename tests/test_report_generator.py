import json
from src.report_generator import save_report
import pytest
from pydantic import ValidationError

from src.report_generator import SafetyReport, validate_report, DISCLAIMER


@pytest.fixture
def sample_report():
    return SafetyReport(
        Eligible_For_Vaccine=True,
        Recommended_Dose_mg=50.0,
        Reasoning_Steps=["Generated educational example."],
        FDA_Safety_Warnings_Cited=[
            "Monitor for adverse reactions."
        ],
        ML_Risk_Assessment={"predicted_class": 1},
        Disclaimer="Original disclaimer",
    )


def test_report_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        SafetyReport(
            Eligible_For_Vaccine=False,
            Recommended_Dose_mg=0.0,
            Reasoning_Steps=[],
            FDA_Safety_Warnings_Cited=[],
            Disclaimer=DISCLAIMER,
            unexpected_field="not allowed",
        )


def test_validation_overrides_eligibility_and_dose(sample_report):
    result = validate_report(
        sample_report,
        retrieved_guideline_text="Monitor for adverse reactions.",
    )

    assert result.Eligible_For_Vaccine is False
    assert result.Recommended_Dose_mg == 0.0
    assert result.Disclaimer == DISCLAIMER
    assert any("Eligibility claim overridden" in step
               for step in result.Reasoning_Steps)
    assert any("Dose claim overridden" in step
               for step in result.Reasoning_Steps)


def test_only_retrieved_evidence_is_cited(sample_report):
    result = validate_report(
        sample_report,
        retrieved_guideline_text="Monitor for adverse reactions.",
    )

    assert result.FDA_Safety_Warnings_Cited == [
        "Monitor for adverse reactions."
    ]


def test_unsupported_citation_is_removed(sample_report):
    sample_report.FDA_Safety_Warnings_Cited = [
        "This statement is not in the source."
    ]

    result = validate_report(
        sample_report,
        retrieved_guideline_text="Actual guideline evidence.",
    )

    assert result.FDA_Safety_Warnings_Cited == []
    assert any(
        "Unsupported source excerpts were removed" in step
        for step in result.Reasoning_Steps
    )


def test_missing_evidence_is_reported(sample_report):
    result = validate_report(sample_report, retrieved_guideline_text="")

    assert result.Eligible_For_Vaccine is False
    assert result.Recommended_Dose_mg == 0.0
    assert any(
        "No retrieved guideline evidence" in step
        for step in result.Reasoning_Steps
    )
    assert result.FDA_Safety_Warnings_Cited == []


def test_negative_dose_bound_is_rejected(sample_report):
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_report(
            sample_report,
            retrieved_guideline_text="Evidence.",
            allowed_max_dose_mg=-1.0,
        )
def test_save_report_creates_json_and_markdown(tmp_path, sample_report):
    json_path, markdown_path = save_report(sample_report, tmp_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert json_path.name == "safety_report.json"
    assert markdown_path.name == "safety_report.md"


def test_saved_json_preserves_report_fields(tmp_path, sample_report):
    json_path, _ = save_report(sample_report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))

    assert saved["Eligible_For_Vaccine"] is True
    assert saved["Recommended_Dose_mg"] == 50.0
    assert saved["ML_Risk_Assessment"] == {"predicted_class": 1}
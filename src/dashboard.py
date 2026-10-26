
"""mRNA-VIP educational project dashboard.

This dashboard demonstrates experimental ML, attention, and RAG/LLM outputs.
It is not a clinical decision-making tool.
"""

import json
import csv
import math
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"
DISCLAIMER = (
    "Educational software demonstration only. This dashboard does not "
    "determine vaccine eligibility, recommend a clinical dose, or provide "
    "medical advice."
)

st.set_page_config(
    page_title="mRNA-VIP | Safety Intelligence",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 mRNA-VIP")
st.subheader("mRNA Vaccine Safety & Dosage Intelligence Platform")
st.warning(DISCLAIMER)

st.caption(
    "Internship demonstration · Experimental dataset model · "
    "Retrieved project-guideline evidence · Not clinically validated"
)


def load_json(path):
    """Load a JSON file and show a useful message if it cannot be read."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.info(f"Optional report not found: {path.relative_to(ROOT)}")
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {path.relative_to(ROOT)}: {error}")
    return None


def load_latency_rows(path):
    """Load measured retrieval latency rows, reporting malformed data."""
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            expected = {
                "question",
                "top_result_section",
                "matches_returned",
                "latency_ms",
            }
            if not expected.issubset(reader.fieldnames or []):
                raise ValueError(f"Expected columns: {', '.join(sorted(expected))}")
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if not row["question"].strip():
                    raise ValueError(f"Missing question on row {line_number}")
                latency = float(row["latency_ms"])
                matches_returned = int(row["matches_returned"])
                if not math.isfinite(latency) or latency < 0:
                    raise ValueError(f"Invalid latency on row {line_number}")
                if matches_returned < 0:
                    raise ValueError(f"Invalid match count on row {line_number}")
                rows.append({"question": row["question"], "latency_ms": latency})
            if not rows:
                raise ValueError("The CSV contains no evaluation rows")
            return rows
    except FileNotFoundError:
        st.info("RAG latency data not found. Run `python -m src.evaluate_rag`.")
    except (OSError, csv.Error, TypeError, ValueError) as error:
        st.error(f"Could not load RAG latency data: {error}")
    return None


# Load available project outputs
model_metrics = load_json(REPORTS / "model_metrics.json")
rag_metrics = load_json(REPORTS / "rag_evaluation.json")

# Panel 1: Model performance
st.header("1. Machine Learning Performance")

roc_path = FIGURES / "roc_curve.png"
if roc_path.exists():
    st.image(str(roc_path), caption="ROC curve from the held-out test set")
else:
    st.info("ROC curve not found. Run src/risk_model.py first.")

if model_metrics:
    st.json(model_metrics)
else:
    st.caption(
        "Detailed model metrics JSON was not found at "
        "outputs/reports/model_metrics.json. The ROC figure is shown when available."
    )

st.subheader("Experimental risk prediction")
st.caption(
    "Optional dataset-model demonstration. Outputs are not clinical risk "
    "assessments or patient-specific guidance."
)
try:
    from src.risk_predictor import FEATURES, predict_reaction_risk

    if (ROOT / "outputs" / "models" / "best_risk_model.joblib").exists():
        with st.form("risk_prediction"):
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            biomarker = st.number_input(
                "Biomarker level", min_value=0.0, value=1.0
            )
            conditions = st.number_input(
                "Pre-existing conditions", min_value=0, value=0
            )
            dose = st.number_input(
                "Vaccine dose (mcg)", min_value=0.0, value=50.0
            )
            prior_reaction = st.selectbox("Prior reaction history", [0, 1])
            submitted = st.form_submit_button("Run educational prediction")
        if submitted:
            features = dict(
                zip(FEATURES, [age, biomarker, conditions, dose, prior_reaction])
            )
            try:
                st.json(predict_reaction_risk(features))
            except (OSError, ValueError, KeyError) as error:
                st.error(f"Could not run the experimental model: {error}")
    else:
        st.info("Risk model artifact not found. Run src/risk_model.py first.")
except ImportError as error:
    st.error(f"Risk prediction dependencies are unavailable: {error}")

# Panel 2: Transformer self-attention
st.header("2. Transformer Self-Attention")

attention_path = FIGURES / "attention_heatmap.png"
if attention_path.exists():
    st.image(
        str(attention_path),
        caption="Illustrative self-attention weights for an RNA sequence",
        width="stretch",
    )
    st.caption(
        "The current implementation uses random, untrained Q/K/V projections. "
        "The heatmap demonstrates attention computation only; it is not "
        "biologically validated or clinically interpretable."
    )
else:
    st.info("Attention heatmap not found. Run src/attention.py first.")

# Panel 3: measured retrieval performance
st.header("3. RAG Evaluation")
if rag_metrics:
    st.json(rag_metrics)

latency_rows = load_latency_rows(ROOT / "outputs" / "rag_test_results.csv")
if latency_rows:
    st.bar_chart(latency_rows, x="question", y="latency_ms", y_label="Latency (ms)")
    st.caption("Measured retrieval latency from outputs/rag_test_results.csv.")

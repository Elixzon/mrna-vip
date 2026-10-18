
"""mRNA-VIP educational project dashboard.

This dashboard demonstrates experimental ML, attention, and RAG/LLM outputs.
It is not a clinical decision-making tool.
"""

import json
import time
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"
SAMPLE = REPORTS / "sample"
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
    """Load a JSON file, returning None if unavailable or invalid."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


# Load available project outputs
model_metrics = load_json(REPORTS / "model_metrics.json")
rag_metrics = load_json(REPORTS / "rag_evaluation.json")
safety_report = load_json(SAMPLE / "safety_report.json")

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

# Panel 2: Transformer self-attention
st.header("2. Transformer Self-Attention")

attention_path = FIGURES / "attention_heatmap.png"
if attention_path.exists():
    st.image(
        str(attention_path),
        caption="Illustrative self-attention weights for an RNA sequence",
        use_container_width=True,
    )
    st.caption(
        "The current implementation uses random, untrained Q/K/V projections. "
        "The heatmap demonstrates attention computation only; it is not "
        "biologically validated or clinically interpretable."
    )
else:
    st.info("Attention heatmap not found. Run src/attention.py first.")

# Panel 3: Actual measured latency

# Panel 3: Actual measured latency
st.header("3. RAG Retrieval vs LLM Generation Latency")

metadata_path = SAMPLE / "report_metadata.json"
latency_metadata = load_json(metadata_path)

if latency_metadata:
    rag_latency = latency_metadata.get("latency_ms")
    llm_latency = latency_metadata.get("llm_latency_ms")

    if (
        isinstance(rag_latency, (int, float))
        and isinstance(llm_latency, (int, float))
    ):
        st.bar_chart(
            {
                "Latency (ms)": {
                    "RAG retrieval": rag_latency,
                    "Gemini generation": llm_latency,
                }
            }
        )
        st.caption(
            f"Measured RAG retrieval: {rag_latency:.2f} ms · "
            f"Measured Gemini generation: {llm_latency:.2f} ms"
        )
    else:
        st.info("Latency fields are missing or invalid in report_metadata.json.")
else:
    st.info(
        "Latency metadata not found. Run "
        "python -m src.report_generator to generate measurements."
    )

# mRNA-VIP: mRNA Vaccine Safety & Dosage Intelligence Platform

An educational AI and data science project combining machine learning, Transformer-style self-attention, retrieval-augmented generation (RAG), and structured report generation.

> **Important disclaimer:** This project is an engineering and educational demonstration only. It is not a clinical decision-support system and must not be used to determine vaccine eligibility, recommend vaccine doses, or make medical decisions.

## Project Overview

The Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform (mRNA-VIP) demonstrates how data science and language-model technologies can be integrated into a single application.

The project includes:

- Data validation, cleaning, and exploratory data analysis.
- Machine-learning models for experimental adverse-reaction risk classification.
- A from-scratch NumPy implementation of scaled dot-product self-attention.
- A retrieval-augmented generation (RAG) pipeline for searching supplied mRNA safety guidelines.
- Schema-validated educational report generation using the Gemini API, with a fallback when generation is unavailable.
- A Streamlit dashboard displaying model evaluation, attention visualization, latency measurements, and an example report.
- Automated tests for core project functionality.

## Technology Stack

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- PyTorch
- Sentence Transformers
- ChromaDB
- Google GenAI SDK
- Pydantic
- Streamlit
- Pytest

## Project Structure

```text
mrna-vip/
├── data/
├── notebooks/
├── src/
│   ├── data_pipeline.py
│   ├── risk_model.py
│   ├── risk_predictor.py
│   ├── attention.py
│   ├── rag_index.py
│   ├── report_generator.py
│   └── dashboard.py
├── tests/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Elixzon/mrna-vip.git
cd mrna-vip
```

Replace the placeholder with your repository URL before publishing this README.

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the environment

Create a local `.env` file using `.env.example` as a template. Add your own Gemini API key if you want to enable live report generation.

Never commit `.env` or expose API keys in source code, screenshots, logs, or public repositories.

### 5. Run the dashboard

```bash
streamlit run src/dashboard.py
```

Follow the local URL printed in the terminal.

## Running Tests

Run the test suite from the project root:

```bash
python -m pytest -v
```

Latest recorded test run: **32 passed, 4 warnings**. The warnings originate from ChromaDB's use of a deprecated Pydantic interface. Re-run the tests to verify the result in your current environment.

## Model Evaluation

The risk-classification workflow compares Logistic Regression and Random Forest using stratified cross-validation and held-out evaluation.

In the recorded experiment, Logistic Regression was selected based on mean cross-validation F1 score. The measured results are experimental results on the supplied dataset, not evidence of clinical validity.

The dashboard includes:

1. ROC curve.
2. Self-attention heatmap.
3. Measured RAG retrieval versus LLM generation latency.

Latency measurements depend on model initialization, hardware, network conditions, and API response times. They should not be interpreted as universal benchmarks.

## Safety and Limitations

- The risk classifier is an experimental model and is not validated for clinical use.
- Predictions must not be used to assess an individual's actual vaccine risk.
- Eligibility is set to false and the displayed dose is set to zero by the report validation safeguards; these are software safety defaults, not clinical recommendations.
- The attention implementation uses randomly initialized projection weights. Its heatmap is illustrative and does not establish biological meaning.
- Guideline citations are checked against retrieved text supplied with this project. This does not independently verify the text against current FDA publications or establish regulatory approval.
- The RAG system may retrieve incomplete or irrelevant evidence.
- Generated reports can contain errors despite schema validation.
- API access and model downloads may require an internet connection.

## Data and Privacy

Use only data that you are authorized to process. Do not upload identifiable patient information to an external model API. Review the dataset's provenance, permissions, and privacy requirements before sharing any data or results.

## Reproducibility

The project records Python package versions in `requirements.txt`. Reproducibility may also depend on the operating system, model downloads, external API availability, and configuration.

## License and Attribution

Review the licenses and attribution requirements for all datasets, source documents, software dependencies, and reused code before distribution. Retain required copyright and license notices.

## Author

Sahil Shaikh

This project was developed as an educational internship project.
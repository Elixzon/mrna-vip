
"""Generate the required three-panel mRNA-VIP dashboard figure.

Educational project visualization only; not clinical guidance.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
RAG_RESULTS = ROOT / "outputs" / "rag_test_results.csv"

DISCLAIMER = (
    "Educational project only — not clinical guidance or a validated "
    "vaccine-safety assessment."
)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    roc_path = FIGURES / "roc_curve.png"
    attention_path = FIGURES / "attention_heatmap.png"

    for path in (roc_path, attention_path, RAG_RESULTS):
        if not path.exists():
            raise FileNotFoundError(f"Required project output not found: {path}")

    with RAG_RESULTS.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"No retrieval measurements found in {RAG_RESULTS}")

    questions = [row["question"] for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]

    # Load existing figures rather than fabricating ROC or attention values.
    roc_image = Image.open(roc_path).convert("RGB")
    attention_image = Image.open(attention_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(roc_image)
    axes[0].set_title("Module 1: ML ROC Curve")
    axes[0].axis("off")

    axes[1].imshow(attention_image)
    axes[1].set_title("Module 2: Self-Attention Heatmap")
    axes[1].axis("off")

    y_positions = np.arange(len(questions))
    axes[2].barh(y_positions, latencies)
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels(
        [question if len(question) <= 42 else question[:39] + "..."
         for question in questions],
        fontsize=8,
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Retrieval latency (ms)")
    axes[2].set_title("Module 3: RAG Retrieval Latency")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("mRNA-VIP — Project Evaluation Dashboard", fontsize=15)
    fig.text(0.5, 0.015, DISCLAIMER, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    output_path = FIGURES / "dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Dashboard saved: {output_path}")
    print("Panels: 3")
    print("DPI: 200")
    print(f"Retrieval measurements plotted: {len(rows)}")


if __name__ == "__main__":
    main()

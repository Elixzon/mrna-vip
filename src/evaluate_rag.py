
"""Evaluate mRNA-VIP retrieval against five manually defined questions.

Educational retrieval evaluation only; not clinical validation.
"""

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.rag_index import build_vector_database, retrieve

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "rag_test_results.csv"
JSON_PATH = ROOT / "outputs" / "reports" / "rag_evaluation.json"

EVALUATION_QUESTIONS = [
    {
        "question": "What are the storage and handling requirements?",
        "expected_section": "Section 1: Storage and Handling Requirements",
    },
    {
        "question": "What dosing guidance is described for different age groups?",
        "expected_section": "Section 2: Dosing Guidance by Age Band",
    },
    {
        "question": "What contraindications and safety warnings are described?",
        "expected_section": "Section 3: Contraindications and Warnings",
    },
    {
        "question": "How should adverse events be documented and reported?",
        "expected_section": "Section 4: Adverse Event Documentation",
    },
    {
        "question": "What should happen after a temperature excursion?",
        "expected_section": "Section 1: Storage and Handling Requirements",
    },
]


def main():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Force a fresh index before evaluating.
    _, collection = build_vector_database(reset=True)

    tests = []
    csv_rows = []

    for item in EVALUATION_QUESTIONS:
        result = retrieve(item["question"], top_k=3)
        matches = result["matches"]

        retrieved_section = (
            matches[0]["metadata"].get("section", "Unknown")
            if matches else "No result"
        )
        passed = retrieved_section == item["expected_section"]

        tests.append({
            "question": item["question"],
            "expected_section": item["expected_section"],
            "retrieved_section": retrieved_section,
            "passed": passed,
            "top1_distance": (
                round(matches[0]["distance"], 4) if matches else None
            ),
            "latency_ms": result["latency_ms"],
            "matches_returned": len(matches),
        })

        csv_rows.append({
            "question": item["question"],
            "top_result_section": retrieved_section,
            "matches_returned": len(matches),
            "latency_ms": result["latency_ms"],
        })

    with CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "top_result_section",
                "matches_returned",
                "latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    correct = sum(test["passed"] for test in tests)
    latencies = [test["latency_ms"] for test in tests]

    report = {
        "project": "mRNA-VIP",
        "module": "Retrieval-Augmented Generation (RAG)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_database": "ChromaDB",
        "top_k": 3,
        "indexed_chunks": collection.count(),
        "evaluation_questions": len(tests),
        "correct_top1_retrievals": correct,
        "top1_accuracy": correct / len(tests) if tests else 0.0,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "notes": [
            "Top-1 accuracy measures expected-section matching on five manually defined questions.",
            "Latency is measured locally and may vary between runs.",
            "This small evaluation set does not establish clinical accuracy or safety.",
            "The source material is for an educational software project, not clinical decision-making.",
        ],
    }

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Indexed chunks: {collection.count()}")
    print(f"Top-1 retrieval accuracy: {correct}/{len(tests)} ({report['top1_accuracy']:.0%})")
    print(f"Mean retrieval latency: {report['mean_latency_ms']} ms")
    for test in tests:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"{status}: {test['question']} -> {test['retrieved_section']}")
    print(f"\nCSV saved: {CSV_PATH}")
    print(f"JSON saved: {JSON_PATH}")
    print("Educational evaluation only; not clinical validation.")


if __name__ == "__main__":
    main()

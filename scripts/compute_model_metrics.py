from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "evaluation" / "evaluation_results.csv"


def compute_metrics(prob_key: str, threshold: float = 0.5):
    y_true = []
    y_pred = []

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob_raw = row.get(prob_key)
            if prob_raw in ("", "None", None):
                continue

            prob = float(prob_raw)
            true_label = row["true_label"]
            true = 1 if true_label == "ai" else 0
            pred = 1 if prob >= threshold else 0

            y_true.append(true)
            y_pred.append(pred)

    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


if __name__ == "__main__":
    print(f"Reading results from {CSV_PATH}\n")
    print("=" * 70)

    for model_name, prob_key in [
        ("Ateeqq", "ateeq_prob"),
        ("wkaandemir", "wkaandemir_prob"),
        ("Ensemble", "ensemble_prob"),
    ]:
        metrics = compute_metrics(prob_key)
        print(f"\n{model_name} (threshold=0.5)")
        print(f"  Accuracy : {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall   : {metrics['recall']:.3f}")
        print(f"  F1 Score : {metrics['f1']:.3f}")
        print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
        print("=" * 70)
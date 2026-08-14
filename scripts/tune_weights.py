from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "evaluation" / "evaluation_results.csv"


def evaluate_weight(ateeq_weight: float, threshold: float = 0.5):
    wka_weight = 1.0 - ateeq_weight
    y_true = []
    y_pred = []

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ateeq_prob"] in ("", "None", None) or row["wkaandemir_prob"] in ("", "None", None):
                continue

            ateeq = float(row["ateeq_prob"])
            wka = float(row["wkaandemir_prob"])
            ensemble = ateeq_weight * ateeq + wka_weight * wka

            true = 1 if row["true_label"] == "ai" else 0
            pred = 1 if ensemble >= threshold else 0

            y_true.append(true)
            y_pred.append(pred)

    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return f1, precision, recall


if __name__ == "__main__":
    print("Searching best Ateeqq weight (wkaandemir = 1 - ateeq_weight)...\n")
    best = []
    for step in range(101):
        ateeq_weight = step / 100.0
        f1, precision, recall = evaluate_weight(ateeq_weight)
        best.append((f1, ateeq_weight, precision, recall))

    best.sort(reverse=True)
    print("Top 5 configurations by F1:")
    for f1, aw, p, r in best[:5]:
        print(f"  Ateeqq weight={aw:.2f}  wka weight={1-aw:.2f}  F1={f1:.3f}  Precision={p:.3f}  Recall={r:.3f}")
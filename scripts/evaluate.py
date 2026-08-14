from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make project root importable when running as a script
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.detector.model import AIDetector    

EVAL_DIR = BASE_DIR / "evaluation"
AI_DIR = EVAL_DIR / "ai"
REAL_DIR = EVAL_DIR / "real"
OUTPUT_CSV = EVAL_DIR / "evaluation_results.csv"

THRESHOLD = 0.5


def collect_images() -> list[tuple[Path, str]]:
    """Return list of (image_path, true_label)."""
    data = []

    for path in AI_DIR.glob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            data.append((path, "ai"))

    for path in REAL_DIR.glob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            data.append((path, "real"))

    return data


def main() -> None:
    detector = AIDetector()
    records = []
    images = collect_images()

    if not images:
        print("No images found in evaluation/ai or evaluation/real")
        return

    print(f"Found {len(images)} images. Running ensemble...\n")

    for image_path, true_label in images:
        detailed = detector.predict_detailed(image_path)
        ensemble = detailed.get("ensemble")
        models = detailed.get("models", {})

        ateeq = models.get("ateeq")
        wkaandemir = models.get("wkaandemir")

        record = {
            "filename": image_path.name,
            "true_label": true_label,
            "ateeq_prob": ateeq,
            "wkaandemir_prob": wkaandemir,
            "ensemble_prob": ensemble,
        }
        records.append(record)

        print(f"{image_path.name:<40} true={true_label:4} ateeq={ateeq:6.3f} wka={wkaandemir:6.3f} ens={ensemble:6.3f}")

    # Save CSV
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "true_label", "ateeq_prob", "wkaandemir_prob", "ensemble_prob"
        ])
        writer.writeheader()
        writer.writerows(records)

    # Compute metrics
    y_true = []
    y_pred = []
    for r in records:
        if r["ensemble_prob"] is None:
            continue
        y_true.append(1 if r["true_label"] == "ai" else 0)
        y_pred.append(1 if r["ensemble_prob"] >= THRESHOLD else 0)

    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    print("\n" + "=" * 60)
    print(f"Confusion Matrix (threshold = {THRESHOLD})")
    print("                 Predicted AI   Predicted Real")
    print(f"Actual AI       {tp:12d} {fn:12d}")
    print(f"Actual Real     {fp:12d} {tn:12d}")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
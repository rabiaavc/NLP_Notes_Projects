
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_data, gold_for
from src.evaluation import score_entities, summarize

MODEL = "dslim/bert-base-NER"
AGGREGATION_STRATEGY = "first"

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
}

# Keep original BERT results unchanged
OUT_DIR = ROOT / "results" / "bert_first"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. LOAD DATASET
# =========================================================

data = load_data("english_test.jsonl")

assert len(data) == 50, (
    f"Expected 50 samples, found {len(data)}"
)

print(f"Dataset loaded: {len(data)} samples")
print(f"Model: {MODEL}")
print(f"Aggregation: {AGGREGATION_STRATEGY}")

# =========================================================
# 3. SELECT DEVICES
# =========================================================

devices = ["cpu"]

if torch.backends.mps.is_available():
    devices.append("mps")

print(f"Available devices: {devices}")

all_results = {}

# =========================================================
# 4. RUN BERT ON CPU AND GPU
# =========================================================

for device in devices:

    print(f"\n{'=' * 55}")
    print(f"BERT {device.upper()} - FIRST AGGREGATION")
    print(f"{'=' * 55}")

    ner = pipeline(
        task="token-classification",
        model=MODEL,
        aggregation_strategy=AGGREGATION_STRATEGY,
        device=device,
    )

    # Warm-up (excluded from latency measurement)
    for row in data[:3]:
        ner(row["text"])

    if device == "mps":
        torch.mps.synchronize()

    details = []

    # -----------------------------------------------------
    # 5. EVALUATE EACH SAMPLE
    # -----------------------------------------------------

    for i, row in enumerate(data, start=1):

        text = row["text"]
        gold = gold_for(row)

        if device == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()

        raw_predictions = ner(text)

        if device == "mps":
            torch.mps.synchronize()

        latency = time.perf_counter() - start_time

        # Convert BERT output to our evaluation format
        predicted = []

        for entity in raw_predictions:

            label = LABEL_MAP.get(
                entity["entity_group"]
            )

            if label is None:
                continue

            begin = int(entity["start"])
            end = int(entity["end"])

            predicted.append({
                "text": text[begin:end],
                "label": label,
                "start": begin,
                "end": end,
            })

        # Exact-span entity evaluation
        stats = score_entities(gold, predicted)

        details.append({
            "sample_id": row["id"],
            "group": row["group"],
            **stats,
            "latency_sec": latency,
            "gold": json.dumps(
                gold, ensure_ascii=False
            ),
            "predicted": json.dumps(
                predicted, ensure_ascii=False
            ),
        })

        print(
            f"{i:02d}/50 | "
            f"{row['id']} | "
            f"F1={stats['f1']:.3f} | "
            f"Time={latency:.3f}s"
        )

    # -----------------------------------------------------
    # 6. AGGREGATE RESULTS
    # -----------------------------------------------------

    summary = summarize(details)

    summary["model"] = MODEL
    summary["device"] = device
    summary["aggregation_strategy"] = (
        AGGREGATION_STRATEGY
    )
    summary["dataset"] = "english_test.jsonl"
    summary["evaluation"] = (
        "Exact-span typed micro-F1"
    )
    summary["torch_version"] = torch.__version__

    all_results[device] = summary

    # -----------------------------------------------------
    # 7. SAVE SAMPLE-LEVEL CSV
    # -----------------------------------------------------

    csv_path = (
        OUT_DIR / f"bert_{device}_detail.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=details[0].keys()
        )

        writer.writeheader()
        writer.writerows(details)

    # -----------------------------------------------------
    # 8. SAVE DEVICE SUMMARY JSON
    # -----------------------------------------------------

    device_summary_path = (
        OUT_DIR / f"bert_{device}_summary.json"
    )

    device_summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------
    # 9. PRINT RESULTS
    # -----------------------------------------------------

    print(f"\n{device.upper()} RESULTS")

    print(
        f"Precision: {summary['precision']:.4f}"
    )
    print(
        f"Recall:    {summary['recall']:.4f}"
    )
    print(
        f"Micro-F1:  {summary['f1']:.4f}"
    )

    print(
        f"TP: {summary['tp']} | "
        f"FP: {summary['fp']} | "
        f"FN: {summary['fn']}"
    )

    print(
        f"Avg time:  "
        f"{summary['avg_latency_sec']:.4f}s"
    )
    print(
        f"Median:    "
        f"{summary['median_latency_sec']:.4f}s"
    )

    print(f"CSV saved: {csv_path}")

    del ner

# =========================================================
# 10. SAVE COMBINED CPU / GPU RESULTS
# =========================================================

summary_path = (
    OUT_DIR / "bert_mac_summary.json"
)

summary_path.write_text(
    json.dumps(
        all_results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETED")
print("=" * 55)

print(f"Model: {MODEL}")
print(f"Aggregation: {AGGREGATION_STRATEGY}")
print(f"Samples: {len(data)}")
print(f"Results saved to: {OUT_DIR}")

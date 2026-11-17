
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_data, gold_for
from src.evaluation import score_entities, summarize

MODEL = "dslim/bert-base-NER"
LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}

data = load_data("english_test.jsonl")
assert len(data) == 50

out_dir = ROOT / "results" / "bert"
out_dir.mkdir(parents=True, exist_ok=True)

devices = ["cpu"]
if torch.backends.mps.is_available():
    devices.append("mps")

all_results = {}

for device in devices:
    print(f"\n=== BERT {device.upper()} ===")

    ner = pipeline(
        "token-classification",
        model=MODEL,
        aggregation_strategy="simple",
        device=device,
    )

    # Warm-up: model loading is excluded from timing
    for row in data[:3]:
        ner(row["text"])

    details = []

    for i, row in enumerate(data, start=1):
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        raw = ner(row["text"])

        if device == "mps":
            torch.mps.synchronize()

        latency = time.perf_counter() - start

        predicted = []
        for p in raw:
            label = LABEL_MAP.get(p["entity_group"])
            if label is None:
                continue

            begin, end = int(p["start"]), int(p["end"])
            predicted.append({
                "text": row["text"][begin:end],
                "start": begin,
                "end": end,
                "label": label,
            })

        gold = gold_for(row)
        stats = score_entities(gold, predicted)

        details.append({
            "sample_id": row["id"],
            "group": row["group"],
            **stats,
            "latency_sec": latency,
            "gold": json.dumps(gold, ensure_ascii=False),
            "predicted": json.dumps(predicted, ensure_ascii=False),
        })

        print(f"{i:02d}/50 | F1={stats['f1']:.3f} | {latency:.3f}s")

    summary = summarize(details)
    summary["model"] = MODEL
    summary["device"] = device
    summary["dataset"] = "english_test.jsonl"
    summary["evaluation"] = "exact-span typed micro F1"
    summary["torch_version"] = torch.__version__

    all_results[device] = summary

    csv_path = out_dir / f"bert_{device}_detail.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)

    print(f"\n{device.upper()} RESULTS")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"Micro-F1:  {summary['f1']:.4f}")
    print(f"Avg time:  {summary['avg_latency_sec']:.4f}s")
    print(f"Median:    {summary['median_latency_sec']:.4f}s")

    del ner

summary_path = out_dir / "bert_mac_summary.json"
summary_path.write_text(
    json.dumps(all_results, indent=2),
    encoding="utf-8",
)

print(f"\nResults saved to {out_dir}")

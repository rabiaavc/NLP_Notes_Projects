#!/usr/bin/env python3
"""Pilot model selection on DEVELOPMENT data only (never on final test data).

Usage from project root:
  python scripts/pilot_compare_models.py --model qwen2.5:3b
  python scripts/pilot_compare_models.py --model qwen3.5:4b-q4_K_M

Requires: pip install -r requirements.txt ; Ollama running locally.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import CORE_LABELS, gold_for, load_data
from src.evaluation import align_generated_entities, score_entities
from src.models import build_prompt, build_schema


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONE Ollama model on held-out development examples")
    parser.add_argument("--model", required=True, help="Local Ollama model tag")
    parser.add_argument("--limit", type=int, default=5, help="Maximum English development examples (1-5)")
    parser.add_argument("--think-mode", choices=["off", "default"], default="off",
                        help="Disable reasoning or use model's default behavior")
    args = parser.parse_args()
    if not 1 <= args.limit <= 5:
        parser.error("--limit must be between 1 and 5 (five English development examples)")

    from jsonschema import Draft202012Validator
    from ollama import Client

    items = [r for r in load_data("development.jsonl") if r["language"] == "en"][: args.limit]
    schema = build_schema(CORE_LABELS)
    client = Client(host="http://localhost:11434", timeout=240)
    records = []
    for i, row in enumerate(items, 1):
        prompt = build_prompt(row["text"], CORE_LABELS, language="en")
        kwargs = dict(model=args.model,
                      messages=[{"role": "user", "content": prompt}],
                      format=schema,
                      options={"temperature": 0.0, "seed": 42, ""
                      "": 2048,
                               "num_predict": 1536},
                      stream=False)
        if args.think_mode == "off":
            kwargs["think"] = False
        start = time.perf_counter()
        try:
            resp = client.chat(**kwargs)
        except Exception as exc:
            raise SystemExit(f"Ollama request failed for {args.model}, sample {row['id']}: {exc}") from exc
        elapsed = time.perf_counter() - start
        raw = resp.message.content or ""
        try:
            parsed = json.loads(raw)
            raw_valid = True
        except (TypeError, ValueError):
            parsed = None
            raw_valid = False
        schema_valid = raw_valid and Draft202012Validator(schema).is_valid(parsed)
        pred, unsupported = align_generated_entities(
            row["text"], parsed["entities"] if schema_valid else [], CORE_LABELS
        )
        stats = score_entities(gold_for(row), pred)
        stats["fp"] += len(unsupported)
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        record = {
            "sample_id": row["id"], "model": args.model,
            "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1,
            "raw_json_valid": bool(raw_valid), "schema_valid": bool(schema_valid),
            "unsupported_count": len(unsupported), "latency_sec": round(elapsed, 3),
            "gold": json.dumps(gold_for(row), ensure_ascii=False),
            "predicted": json.dumps(pred, ensure_ascii=False),
            "unsupported": json.dumps(unsupported, ensure_ascii=False),
            "raw_response": raw,
        }
        records.append(record)
        print(f"{i}/{len(items)}  {row['id']}  F1={f1:.3f}  schema_valid={schema_valid}  time={elapsed:.1f}s")

    tp = sum(x["tp"] for x in records)
    fp = sum(x["fp"] for x in records)
    fn = sum(x["fn"] for x in records)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    summary = {
        "model": args.model, "dataset": "development.jsonl (English only)",
        "n": len(records), "temperature": 0.0, "seed": 42,
        "num_ctx": 2048, "num_predict": 1536, "think_mode": args.think_mode,
        "evaluation": "exact-span typed micro F1; ungrounded strings count as FP",
        "precision": p, "recall": r, "micro_f1": f1,
        "schema_valid_rate": sum(x["schema_valid"] for x in records) / len(records),
        "unsupported_total": sum(x["unsupported_count"] for x in records),
        "average_latency_sec": sum(x["latency_sec"] for x in records) / len(records),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    name = f"pilot_{safe}_{args.think_mode}"
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n=== DEVELOPMENT SET SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: results/{name}.json and results/{name}.csv")


if __name__ == "__main__":
    main()

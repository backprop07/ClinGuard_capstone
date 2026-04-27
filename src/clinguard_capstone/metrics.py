from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


MODEL_RUNS = {
    "gpt54_codex": "GPT-5.4",
    "deepseek_chat": "DeepSeek-V3.2",
    "qwen35_9b": "Qwen3.5-9B",
}


def load_metrics(results_dir: Path) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for run_id, label in MODEL_RUNS.items():
        path = results_dir / run_id / "metrics.json"
        with path.open("r", encoding="utf-8") as handle:
            metrics[label] = json.load(handle)
    return metrics


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_text_only_cases(path: Path) -> tuple[int, int]:
    total = 0
    cases_with_images = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            images = item.get("patient_assets", {}).get("images", [])
            if images:
                cases_with_images += 1
    return total, cases_with_images

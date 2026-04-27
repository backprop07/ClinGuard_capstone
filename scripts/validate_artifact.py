from __future__ import annotations

import json
from pathlib import Path

from clinguard_capstone.metrics import MODEL_RUNS, validate_text_only_cases


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases = ROOT / "data" / "medqa_text" / "cases.jsonl"
    total, cases_with_images = validate_text_only_cases(cases)
    if total != 300:
        raise SystemExit(f"Expected 300 MedQA text cases, found {total}.")
    if cases_with_images:
        raise SystemExit(f"Expected text-only cases, found {cases_with_images} cases with images.")

    for run_id in MODEL_RUNS:
        metrics_path = ROOT / "results" / run_id / "metrics.json"
        csv_path = ROOT / "results" / run_id / "metrics.csv"
        if not metrics_path.exists() or not csv_path.exists():
            raise SystemExit(f"Missing metrics artifacts for {run_id}.")
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("counts", {}).get("attempted") != 300:
            raise SystemExit(f"{run_id} did not attempt 300 cases.")

    print("Artifact validation passed: 300 text-only MedQA cases and three complete result sets.")


if __name__ == "__main__":
    main()


"""Measure allowed-model token use on a balanced labeled sample.

The benchmark intentionally records raw answers and API usage instead of using an
LLM judge. That keeps the paid experiment focused on model generation; answers
can then be reviewed against each task's ground-truth rubric.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.conversation import ChatMessage
from app.fireworks_client import FireworksClient, FireworksConfig, _parse_allowed_models
from app.models import Tier
from app.output_limits import requested_output_limit


PROFILES: dict[str, tuple[str, str | None]] = {
    "m3-default": ("minimax-m3", None),
    "m3-no-thinking": ("minimax-m3", "none"),
    "gemma-26": ("gemma-4-26b-a4b-it", None),
    "gemma-31": ("gemma-4-31b-it", None),
    "kimi-code": ("kimi-k2p7-code", None),
}


def _model_for(allowed: list[str], identifier: str) -> str:
    matches = [model for model in allowed if model.lower().endswith(identifier)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one allowed model containing {identifier!r}; found {matches!r}")
    return matches[0]


def _balanced_sample(tasks: list[dict[str, Any]], limit_per_category: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for task in tasks:
        category = task["category"]
        if counts[category] < limit_per_category:
            selected.append(task)
            counts[category] += 1
    return selected


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Fireworks routing experiments")
    parser.add_argument("--tasks", type=Path, default=Path("../../work/tutorial-router-eval/data/queries_raw.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/fireworks-benchmark-results.json"))
    parser.add_argument("--profiles", nargs="+", default=["m3-default", "m3-no-thinking"])
    parser.add_argument("--limit-per-category", type=int, default=2)
    args = parser.parse_args()

    unknown_profiles = set(args.profiles) - PROFILES.keys()
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {sorted(unknown_profiles)}")
    if args.limit_per_category < 1:
        raise ValueError("--limit-per-category must be positive")

    tasks = _balanced_sample(json.loads(args.tasks.read_text()), args.limit_per_category)
    base = FireworksConfig.from_environment()
    allowed = _parse_allowed_models(os.environ["ALLOWED_MODELS"])
    prior_records = json.loads(args.output.read_text()) if args.output.exists() else []
    completed = {(record["profile"], record["id"]) for record in prior_records}
    clients: dict[str, FireworksClient] = {}

    try:
        for profile in args.profiles:
            model_identifier, reasoning_effort = PROFILES[profile]
            model = _model_for(allowed, model_identifier)
            config = FireworksConfig(
                api_key=base.api_key,
                base_url=base.base_url,
                small_model=model,
                large_model=model,
                max_completion_tokens=base.max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
            clients[profile] = FireworksClient(config)

        for profile, client in clients.items():
            for task in tasks:
                key = (profile, task["id"])
                if key in completed:
                    continue
                prompt = task["prompt"]
                record: dict[str, Any] = {
                    "id": task["id"],
                    "category": task["category"],
                    "profile": profile,
                    "token_limit": requested_output_limit(prompt),
                    "ground_truth": task["ground_truth"],
                }
                try:
                    answer = client.complete(
                        [{"role": "user", "content": prompt}],
                        Tier.SMALL,
                        conversation_id=f"benchmark:{profile}:{task['id']}",
                        max_tokens=requested_output_limit(prompt),
                    )
                    record.update(
                        {
                            "tokens": answer.token_count,
                            "retried_after_truncation": answer.metadata.get("retried_after_truncation", False),
                            "answer": answer.answer,
                        }
                    )
                except Exception as exc:
                    record["error"] = str(exc)
                prior_records.append(record)
                completed.add(key)
                _write(args.output, prior_records)
                print(f"{profile} {task['id']} {record.get('tokens', 'error')} tokens", flush=True)
    finally:
        for client in clients.values():
            client.http.close()

    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "tokens": 0, "retries": 0})
    for record in prior_records:
        if record["profile"] in args.profiles:
            group = summary[record["profile"]]
            group["tasks"] += 1
            group["tokens"] += record.get("tokens", 0)
            group["retries"] += int(record.get("retried_after_truncation", False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

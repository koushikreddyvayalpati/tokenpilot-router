from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.cache import JsonAnswerCache
from app.fireworks_client import FireworksClient, FireworksConfig
from app.graph_router import LangGraphRouter

INPUT_PATH = Path("/input/tasks.json")
OUTPUT_PATH = Path("/output/results.json")


def _tasks_from_payload(payload: Any) -> list[dict[str, Any]]:
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON list or an object with a tasks list")
    normalized: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be an object")
        task_id = task.get("task_id", task.get("id"))
        prompt = task.get("prompt", task.get("query"))
        if not isinstance(task_id, str | int) or not str(task_id).strip():
            raise ValueError(f"task {index} is missing task_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {index} is missing prompt")
        normalized.append({"task_id": str(task_id), "prompt": prompt})
    return normalized


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    temporary.replace(path)


def run_batch(input_path: Path, output_path: Path, router: LangGraphRouter) -> list[dict[str, str]]:
    tasks = _tasks_from_payload(json.loads(input_path.read_text()))
    results: list[dict[str, str]] = []
    for task in tasks:
        state = router.run(task["prompt"], task_id=task["task_id"], conversation_id=f"task:{task['task_id']}")
        results.append({"task_id": task["task_id"], "answer": state["answer"].answer})
    _atomic_json_write(output_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 batch submission runner")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    config = FireworksConfig.from_environment()
    router = LangGraphRouter(
        fireworks=FireworksClient(config),
        answer_cache=JsonAnswerCache(Path("/tmp/track1-router/answers.json")),
    )
    run_batch(args.input, args.output, router)


if __name__ == "__main__":
    main()

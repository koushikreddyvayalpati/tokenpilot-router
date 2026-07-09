from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.cache import JsonAnswerCache
from app.graph_router import LangGraphRouter
from app.models import EvalTask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="eval/tasks_sample.json")
    parser.add_argument("--cache", default=".router-cache/answers.json")
    args = parser.parse_args()
    tasks = [EvalTask.model_validate(item) for item in json.loads(Path(args.tasks).read_text())]
    router = LangGraphRouter(answer_cache=JsonAnswerCache(Path(args.cache)))
    results = []
    for task in tasks:
        try:
            state = router.run(task.prompt, task.id)
            answer = state["answer"]
            exact = task.expected is None or answer.answer.strip().lower() == task.expected.lower()
            results.append(
                {
                    "id": task.id,
                    "attempts": state["attempts"],
                    "answer": answer.answer,
                    "confidence": answer.confidence,
                    "tokens": state["token_count"],
                    "exact": exact,
                }
            )
        except Exception as exc:
            results.append({"id": task.id, "error": str(exc)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.cache import JsonAnswerCache
from app.graph_router import LangGraphRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 LangGraph model router")
    parser.add_argument("prompt", nargs="?", help="Task prompt to route")
    parser.add_argument("--cache", default=".router-cache/answers.json")
    parser.add_argument("--conversation-id", default="manual")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("prompt is required")
    router = LangGraphRouter(answer_cache=JsonAnswerCache(Path(args.cache)))
    result = router.run(args.prompt, conversation_id=args.conversation_id)
    answer = result["answer"]
    print(json.dumps({"answer": answer.model_dump(mode="json"), "attempts": result["attempts"], "tokens": result["token_count"]}, indent=2))


if __name__ == "__main__":
    main()

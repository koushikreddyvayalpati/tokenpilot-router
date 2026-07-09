from __future__ import annotations

import argparse
from pathlib import Path

from app.cache import JsonAnswerCache
from app.graph_router import LangGraphRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Track 1 routing chat")
    parser.add_argument("--conversation-id", default="interactive")
    parser.add_argument("--cache", default=".router-cache/answers.json")
    args = parser.parse_args()

    router = LangGraphRouter(answer_cache=JsonAnswerCache(Path(args.cache)))
    print("Type :clear to discard this chat history, or press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if not prompt:
            continue
        if prompt == ":clear":
            router.conversations.clear(args.conversation_id)
            print("history cleared")
            continue

        try:
            state = router.run(prompt, conversation_id=args.conversation_id)
        except Exception as exc:
            print(f"error: {exc}")
            continue
        answer = state["answer"]
        cached_output = " cache-hit" if answer.cache_hit else ""
        cached_prompt_tokens = answer.metadata.get("prompt_cache_hit_tokens")
        cached_input = f", {cached_prompt_tokens} cached-input" if cached_prompt_tokens else ""
        print(f"assistant [{answer.tier.value}, {state['token_count']} tokens{cached_input}{cached_output}]> {answer.answer}")


if __name__ == "__main__":
    main()

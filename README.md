# Track 1 LangGraph Router

This is a submission-ready routing agent for the AMD/Fireworks Track 1 challenge. It reads `/input/tasks.json`, writes `/output/results.json`, and minimizes paid Fireworks tokens by trying deterministic/local answers first, escalating only when confidence is too low, and using two cache layers:

- LangGraph node cache for duplicate work inside a graph process.
- JSON persistent answer cache for repeat prompts across runs.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=$PWD
python -m app.cli "What is 17 * 23?"
pytest
```

## Continue a chat efficiently

Start one process per active conversation. The router retains the prior turns in memory, keeps a stable Fireworks cache-affinity key for that conversation, and limits history to recent whole messages. This lets Fireworks reuse the common prompt prefix while avoiding an extra paid summarization call.

```bash
export FIREWORKS_API_KEY=your_key_here
python -m app.chat_cli --conversation-id customer-123
```

Use `:clear` in the chat to discard its in-memory history. Conversation history is intentionally not written to disk; the answer cache is persisted separately.

For paid tiers:

```bash
export FIREWORKS_API_KEY=your_key_here
python -m eval.run_eval --tasks eval/tasks_sample.json

# Stress test hard coding, proof, security, and systems prompts.
python -m eval.run_eval --tasks eval/tasks_hard.json
```

## Docker

```bash
docker buildx build --platform linux/amd64 -t track1-router .
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS='["accounts/fireworks/models/your-cheap-model", "accounts/fireworks/models/your-large-model"]' \
  -v "$PWD/eval:/input:ro" \
  -v "$PWD/output:/output" \
  track1-router
```

The judging harness supplies `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS`. The runner chooses its tiers only from `ALLOWED_MODELS`, processes `tasks.json`, and writes an atomic `/output/results.json` list of `{ "task_id", "answer" }` records.

## Tuning points

- Replace `app/local_model.py` with an AMD GPU pod local model adapter.
- Tune `confidence_threshold` in `LangGraphRouter`.
- Add representative hard cases to `eval/tasks_sample.json`.
- Keep the system prompt and conversation history prefix stable. The Fireworks API uses `prompt_cache_key` and `x-session-affinity` to route related turns to the same cache replica.

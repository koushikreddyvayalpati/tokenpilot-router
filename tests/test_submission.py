import json
from pathlib import Path

from app.cache import JsonAnswerCache
from app.fireworks_client import _parse_allowed_models, _select_model_tiers
from app.graph_router import LangGraphRouter
from app.submission import run_batch


def test_submission_runner_writes_harness_result_shape(tmp_path: Path) -> None:
    input_path = tmp_path / "tasks.json"
    output_path = tmp_path / "output" / "results.json"
    input_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "a", "prompt": "What is 17 * 23?"},
                    {"task_id": 2, "prompt": "Calculate (18 + 6) / 3."},
                ]
            }
        )
    )
    router = LangGraphRouter(answer_cache=JsonAnswerCache(tmp_path / "answers.json"))

    results = run_batch(input_path, output_path, router)

    assert results == [{"task_id": "a", "answer": "391"}, {"task_id": "2", "answer": "8"}]
    assert json.loads(output_path.read_text()) == results


def test_allowed_model_parser_and_tiers() -> None:
    models = _parse_allowed_models('["accounts/x/model-8b", "accounts/x/model-70b"]')
    assert _select_model_tiers(models) == ("accounts/x/model-8b", "accounts/x/model-70b")
    assert _parse_allowed_models("cheap-model, large-model") == ["cheap-model", "large-model"]

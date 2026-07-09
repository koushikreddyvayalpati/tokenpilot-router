from pathlib import Path

from app.cache import JsonAnswerCache


def test_json_cache_round_trip(tmp_path: Path) -> None:
    cache = JsonAnswerCache(tmp_path / "answers.json")
    assert cache.get("hello", "local") is None
    cache.set("hello", "local", {"answer": "world"})
    assert cache.get("hello", "local") == {"answer": "world"}


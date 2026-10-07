"""The OpenRouter client: what a call costs is written down, per caller."""

import json

import httpx

import paths
from tailor import llm


def test_every_call_is_recorded_with_its_caller_and_its_cost(monkeypatch, tmp_path):
    body = {
        "model": "z-ai/glm-5.3", "provider": "Z.AI",
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 300, "cost": 0.0021,
                  "prompt_tokens_details": {"cached_tokens": 900},
                  "completion_tokens_details": {"reasoning_tokens": 250}},
    }
    monkeypatch.setattr(llm.httpx, "post",
                        lambda *a, **k: httpx.Response(200, json=body))
    monkeypatch.setattr(paths, "DATA", tmp_path)

    assert llm.complete("system", "user", model="z-ai/glm-5.3") == "ok"

    row = json.loads((tmp_path / llm.USAGE_FILE).read_text().splitlines()[0])
    assert row["kind"] == "test_llm.test_every_call_is_recorded_with_its_caller_and_its_cost"
    assert (row["prompt"], row["cached"], row["completion"], row["reasoning"]) == (1000, 900, 300, 250)
    assert row["cost"] == 0.0021 and row["provider"] == "Z.AI"


def test_calls_go_to_the_cheapest_provider_unless_told_otherwise(monkeypatch):
    payload = llm._payload([{"role": "user", "content": "hi"}], "z-ai/glm-5.3", 0.3, 100, None)
    assert payload["provider"] == {"sort": llm.PROVIDER_SORT}
    monkeypatch.setattr(llm, "PROVIDER_SORT", "")
    payload = llm._payload([{"role": "user", "content": "hi"}], "z-ai/glm-5.3", 0.3, 100, None)
    assert "provider" not in payload

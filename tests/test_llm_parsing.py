"""Tests for the LLM client's JSON-fence stripping and parsing.

Doesn't hit the real API — we mock the OpenAI client's response.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm import LLMClient


def _mock_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = content
    return resp


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DO_INFERENCE_API_KEY", "test-key")
    return LLMClient(model="test-model")


def test_plain_json_parses(client):
    with patch.object(client.client.chat.completions, "create", return_value=_mock_response('{"foo": "bar"}')):
        result = client.chat_json(system="s", user="u")
        assert result == {"foo": "bar"}


def test_json_in_markdown_fence_parses(client):
    fenced = '```json\n{"foo": "bar"}\n```'
    with patch.object(client.client.chat.completions, "create", return_value=_mock_response(fenced)):
        result = client.chat_json(system="s", user="u")
        assert result == {"foo": "bar"}


def test_bare_fence_parses(client):
    fenced = '```\n{"foo": "bar"}\n```'
    with patch.object(client.client.chat.completions, "create", return_value=_mock_response(fenced)):
        result = client.chat_json(system="s", user="u")
        assert result == {"foo": "bar"}


def test_invalid_json_raises(client):
    with patch.object(client.client.chat.completions, "create", return_value=_mock_response("not json at all")):
        with pytest.raises(ValueError):
            client.chat_json(system="s", user="u")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DO_INFERENCE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing DigitalOcean inference API key"):
        LLMClient(model="test-model")

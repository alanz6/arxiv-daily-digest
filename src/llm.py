"""LLM client.

Wraps the OpenAI Python SDK pointed at DigitalOcean's serverless inference
endpoint (or any OpenAI-compatible host). All LLM-calling modules in this
project go through this client.

DigitalOcean Serverless Inference: https://docs.digitalocean.com/products/genai-platform/
- Base URL is typically https://inference.do-ai.run/v1
- API key from DO control panel
- Models: llama3.3-70b-instruct, openai-gpt-4o, etc. (see DO docs for current list)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL = "llama3.3-70b-instruct"


class LLMClient:
    """OpenAI-compatible client with a `chat_json` helper for structured output."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or os.environ.get("DO_INFERENCE_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("DO_INFERENCE_API_KEY")
        self.base_url = base_url or os.environ.get("DO_INFERENCE_BASE_URL", DEFAULT_BASE_URL)
        if not self.api_key:
            raise RuntimeError(
                "Missing DigitalOcean inference API key. "
                "Set DO_INFERENCE_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_json(
        self,
        system: str,
        user: str,
        schema_hint: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> dict:
        """Send a chat request and parse the response as JSON.

        We use OpenAI's JSON mode (`response_format={"type": "json_object"}`)
        which most DigitalOcean-hosted models support. The expected shape is
        described in the prompt via `schema_hint`.

        Raises ValueError if the response isn't valid JSON.
        """
        system_text = system
        if schema_hint:
            system_text += (
                "\n\nReturn ONLY a JSON object matching this shape (no prose, "
                "no markdown fences):\n"
                + schema_hint
            )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
        )
        return _parse_json_response(response.choices[0].message.content)

    async def chat_json_async(
        self,
        system: str,
        user: str,
        schema_hint: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> dict:
        """Async version of `chat_json`. Same contract, used by parallel scoring."""
        system_text = system
        if schema_hint:
            system_text += (
                "\n\nReturn ONLY a JSON object matching this shape (no prose, "
                "no markdown fences):\n"
                + schema_hint
            )

        response = await self.async_client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user},
            ],
        )
        # Log the finish reason if it's not "stop" — useful for diagnosing
        # truncation that isn't a max_tokens issue.
        finish = response.choices[0].finish_reason
        if finish and finish != "stop":
            logger.warning("LLM finish_reason=%s (expected 'stop')", finish)
        return _parse_json_response(response.choices[0].message.content)


def _parse_json_response(content: Optional[str]) -> dict:
    """Strip markdown fences (some models add them even under JSON mode) and json.loads."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("LLM returned non-JSON: %s", text[:500])
        raise ValueError(f"LLM did not return valid JSON: {e}") from e

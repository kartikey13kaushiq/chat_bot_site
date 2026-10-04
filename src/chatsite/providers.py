"""LLM providers behind one streaming interface.

``OpenAICompatibleProvider`` speaks the ``/v1/chat/completions`` streaming protocol, which covers OpenAI,
Azure OpenAI, Groq, Together, Mistral, and local servers such as Ollama, vLLM and LM Studio.
``AnthropicProvider`` speaks the Anthropic Messages API. ``EchoProvider`` works offline for demos and tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ProviderError(RuntimeError):
    """The upstream model API failed; the message is safe to show to users."""


class ChatProvider(Protocol):
    name: str

    def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data:`` payloads of a server-sent event stream."""
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            yield line[5:].strip()


async def _raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.status_code >= 400:
        body = (await response.aread()).decode(errors="replace")[:300]
        if response.status_code in (401, 403):
            raise ProviderError(f"{provider} rejected the API key")
        if response.status_code == 429:
            raise ProviderError(f"{provider} is rate limiting requests; try again shortly")
        raise ProviderError(f"{provider} returned HTTP {response.status_code}: {body}")


class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.7,
        client: httpx.AsyncClient | None = None,
        name: str = "openai",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.name = name
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                await _raise_for_status(response, self.name)
                async for data in _sse_data(response):
                    if data == "[DONE]":
                        return
                    choices = json.loads(data).get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise ProviderError(f"could not reach {self.name}: {type(e).__name__}") from e


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10))

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "stream": True,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/v1/messages", json=payload, headers=headers
            ) as response:
                await _raise_for_status(response, self.name)
                async for data in _sse_data(response):
                    event = json.loads(data)
                    if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta":
                        yield event["delta"]["text"]
                    elif event.get("type") == "error":
                        raise ProviderError(f"anthropic: {event['error'].get('message', 'stream error')}")
                    elif event.get("type") == "message_stop":
                        return
        except httpx.HTTPError as e:
            raise ProviderError(f"could not reach anthropic: {type(e).__name__}") from e


class EchoProvider:
    """No model: answers from the conversation itself. Lets the site run with zero configuration."""

    name = "echo"

    async def stream(self, system: str, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        last = messages[-1].content if messages else ""
        turns = sum(1 for m in messages if m.role == "user")
        reply = (
            f'(demo mode, no model configured) You said: "{last}". '
            f"This is message {turns} of our conversation. Set CHAT_PROVIDER to openai, anthropic or ollama "
            "to talk to a real model."
        )
        for word in reply.split(" "):
            yield word + " "


def provider_from_env(env: dict[str, str] | None = None) -> ChatProvider:
    env = dict(os.environ if env is None else env)
    kind = env.get("CHAT_PROVIDER", "echo").lower()
    if kind == "echo":
        return EchoProvider()
    if kind == "anthropic":
        return AnthropicProvider(
            model=env.get("CHAT_MODEL", "claude-sonnet-5-5"), api_key=_require(env, "ANTHROPIC_API_KEY")
        )
    if kind == "ollama":
        return OpenAICompatibleProvider(
            model=env.get("CHAT_MODEL", "llama3.1"),
            base_url=env.get("CHAT_BASE_URL", "http://localhost:11434/v1"),
            name="ollama",
        )
    if kind in ("openai", "openai-compatible"):
        return OpenAICompatibleProvider(
            model=env.get("CHAT_MODEL", "gpt-4.1-mini"),
            base_url=env.get("CHAT_BASE_URL", "https://api.openai.com/v1"),
            api_key=env.get("OPENAI_API_KEY") or env.get("CHAT_API_KEY"),
            name=kind,
        )
    raise ValueError(f"unknown CHAT_PROVIDER {kind!r} (expected echo, openai, openai-compatible, ollama, anthropic)")


def _require(env: dict[str, str], key: str) -> str:
    value = env.get(key)
    if not value:
        raise ValueError(f"{key} must be set")
    return value

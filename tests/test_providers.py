import json

import httpx
import pytest

from chatsite.providers import (
    AnthropicProvider,
    ChatMessage,
    EchoProvider,
    OpenAICompatibleProvider,
    ProviderError,
    provider_from_env,
)

HISTORY = [ChatMessage("user", "Hi"), ChatMessage("assistant", "Hello!"), ChatMessage("user", "Tell me a joke")]


def sse(*events: str) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode()


async def collect(provider, system="Be brief.", messages=HISTORY):
    return [chunk async for chunk in provider.stream(system, messages)]


async def test_openai_compatible_streams_deltas_and_sends_the_system_prompt():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Why "}}]},
            {"choices": [{"delta": {"content": "not?"}}]},
            {"choices": []},
        ]
        return httpx.Response(200, content=sse(*map(json.dumps, chunks), "[DONE]"))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider("gpt-x", base_url="https://llm.test/v1/", api_key="sk-1", client=client)
    assert await collect(provider) == ["Why ", "not?"]
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-1"
    assert seen["body"]["stream"] is True
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert [m["role"] for m in seen["body"]["messages"][1:]] == ["user", "assistant", "user"]


async def test_local_servers_need_no_key():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, content=sse(json.dumps({"choices": [{"delta": {"content": "ok"}}]}), "[DONE]"))

    provider = OpenAICompatibleProvider(
        "llama", base_url="http://ollama.test/v1", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await collect(provider) == ["ok"]


async def test_anthropic_streams_text_deltas():
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        events = [
            {"type": "message_start", "message": {}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Knock "}},
            {"type": "ping"},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "knock."}},
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(200, content=body.encode())

    provider = AnthropicProvider(
        "claude-x", api_key="ak", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    assert await collect(provider) == ["Knock ", "knock."]
    assert seen["headers"]["x-api-key"] == "ak"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "Be brief."
    assert all(m["role"] != "system" for m in seen["body"]["messages"])


async def test_anthropic_stream_errors_surface():
    def handler(request):
        return httpx.Response(200, content=sse(json.dumps({"type": "error", "error": {"message": "overloaded"}})))

    provider = AnthropicProvider("c", api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ProviderError, match="overloaded"):
        await collect(provider)


@pytest.mark.parametrize(("status", "text"), [(401, "API key"), (429, "rate limiting"), (500, "HTTP 500")])
async def test_http_errors_become_readable_provider_errors(status, text):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, text="nope")))
    with pytest.raises(ProviderError, match=text):
        await collect(OpenAICompatibleProvider("m", base_url="https://x.test", client=client))


async def test_network_failures_become_provider_errors():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="could not reach"):
        await collect(AnthropicProvider("c", api_key="k", client=client))
    with pytest.raises(ProviderError, match="could not reach"):
        await collect(OpenAICompatibleProvider("m", client=client))


async def test_echo_provider_needs_nothing():
    text = "".join(await collect(EchoProvider()))
    assert "Tell me a joke" in text and "message 2" in text


def test_provider_selection_from_environment():
    assert provider_from_env({}).name == "echo"
    assert provider_from_env({"CHAT_PROVIDER": "ollama"}).base_url == "http://localhost:11434/v1"
    openai = provider_from_env(
        {
            "CHAT_PROVIDER": "openai-compatible",
            "CHAT_BASE_URL": "https://api.groq.com/openai/v1",
            "CHAT_API_KEY": "g",
            "CHAT_MODEL": "llama-3.3-70b",
        }
    )
    assert (openai.base_url, openai.api_key, openai.model) == ("https://api.groq.com/openai/v1", "g", "llama-3.3-70b")
    assert provider_from_env({"CHAT_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}).name == "anthropic"
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        provider_from_env({"CHAT_PROVIDER": "anthropic"})
    with pytest.raises(ValueError, match="unknown"):
        provider_from_env({"CHAT_PROVIDER": "skynet"})

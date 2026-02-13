import asyncio
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.openai_api import OpenAIAPIChannel
from nanobot.config.schema import OpenAIAPIConfig


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


def test_openai_api_channel_authorization() -> None:
    channel = OpenAIAPIChannel(
        OpenAIAPIConfig(enabled=True, api_key="secret-token"),
        MessageBus(),
    )

    assert channel.is_request_authorized("Bearer secret-token") is True
    assert channel.is_request_authorized("Bearer wrong-token") is False
    assert channel.is_request_authorized(None) is False


def test_openai_api_channel_messages_to_prompt() -> None:
    single = OpenAIAPIChannel._messages_to_prompt(
        [{"role": "user", "content": "你好，介绍一下你自己"}]
    )
    assert single == "你好，介绍一下你自己"

    multi = OpenAIAPIChannel._messages_to_prompt(
        [
            {"role": "system", "content": "你是助手"},
            {"role": "assistant", "content": "你好"},
            {"role": "user", "content": [{"type": "text", "text": "再说一次"}]},
        ]
    )
    assert "OpenAI Chat Completions" in multi
    assert "[system] 你是助手" in multi
    assert "[assistant] 你好" in multi
    assert "[user] 再说一次" in multi


def test_openai_api_channel_rejects_unauthorized_http_call() -> None:
    channel = OpenAIAPIChannel(
        OpenAIAPIConfig(enabled=True, api_key="secret-token"),
        MessageBus(),
    )

    status, payload = channel.handle_chat_completion_http(
        body=b'{"messages":[{"role":"user","content":"hi"}]}',
        client_ip="127.0.0.1",
        authorization=None,
    )

    assert status == 401
    assert payload["error"]["type"] == "authentication_error"


async def test_openai_api_channel_send_resolves_pending_waiter() -> None:
    channel = OpenAIAPIChannel(OpenAIAPIConfig(enabled=True), MessageBus())
    waiter = asyncio.get_running_loop().create_future()
    channel._pending["req-1"] = waiter

    await channel.send(
        OutboundMessage(
            channel="openai_api",
            chat_id="alice",
            content="收到",
            metadata={"request_id": "req-1"},
        )
    )

    assert await waiter == "收到"

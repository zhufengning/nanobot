import asyncio
import os
import uuid
from pathlib import Path
from typing import Any

from nanobot.agent.loop import AgentLoop
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.openai_api import OpenAIAPIChannel
from nanobot.config.loader import _parse_env_line, load_dotenv
from nanobot.config.schema import OpenAIAPIConfig
from nanobot.providers.base import LLMProvider, LLMResponse


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


class DummyProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        return LLMResponse(content="ok")

    def get_default_model(self) -> str:
        return "dummy-model"


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


def test_agent_loop_can_disable_web_search_tool() -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=Path.cwd() / "workspace",
        web_search_enabled=False,
    )

    assert "web_search" not in loop.tools.tool_names
    assert "web_fetch" in loop.tools.tool_names


def test_agent_loop_can_disable_multiple_builtin_tools() -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=DummyProvider(),
        workspace=Path.cwd() / "workspace",
        filesystem_enabled=False,
        exec_enabled=False,
        web_search_enabled=False,
        web_fetch_enabled=False,
        message_enabled=False,
        spawn_enabled=False,
        cron_enabled=False,
    )

    disabled = {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "exec",
        "web_search",
        "web_fetch",
        "message",
        "spawn",
        "cron",
    }
    assert disabled.isdisjoint(set(loop.tools.tool_names))


def test_subagent_manager_can_disable_web_search_tool() -> None:
    manager = SubagentManager(
        provider=DummyProvider(),
        workspace=Path.cwd(),
        bus=MessageBus(),
        web_search_enabled=False,
    )
    tools = manager._build_tools()

    assert "web_search" not in tools.tool_names
    assert "web_fetch" in tools.tool_names


def test_subagent_manager_can_disable_files_exec_and_fetch_tools() -> None:
    manager = SubagentManager(
        provider=DummyProvider(),
        workspace=Path.cwd(),
        bus=MessageBus(),
        filesystem_enabled=False,
        exec_enabled=False,
        web_search_enabled=False,
        web_fetch_enabled=False,
    )
    tools = manager._build_tools()

    disabled = {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "exec",
        "web_search",
        "web_fetch",
    }
    assert disabled.isdisjoint(set(tools.tool_names))


def test_parse_env_line_supports_export_and_quotes() -> None:
    assert _parse_env_line("export DEMO_KEY=value123") == ("DEMO_KEY", "value123")
    assert _parse_env_line("QUOTED='hello world'") == ("QUOTED", "hello world")
    assert _parse_env_line("EMPTY=") == ("EMPTY", "")
    assert _parse_env_line("# comment") is None


def test_load_dotenv_from_custom_path_without_override() -> None:
    env_file = Path.cwd() / "workspace" / f"test-dotenv-{uuid.uuid4().hex}.env"
    env_file.write_text("DOTENV_ALPHA=one\nDOTENV_BETA='two words'\n", encoding="utf-8")

    os.environ.pop("DOTENV_ALPHA", None)
    os.environ["DOTENV_BETA"] = "keep-original"

    try:
        loaded = load_dotenv(env_path=env_file, override=False)
        assert loaded == env_file
        assert os.environ.get("DOTENV_ALPHA") == "one"
        assert os.environ.get("DOTENV_BETA") == "keep-original"
    finally:
        os.environ.pop("DOTENV_ALPHA", None)
        os.environ.pop("DOTENV_BETA", None)
        if env_file.exists():
            env_file.unlink()

"""OpenAI-compatible Chat Completions channel (HTTP)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import OpenAIAPIConfig


class _OpenAIHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying a channel reference."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], channel: "OpenAIAPIChannel"):
        super().__init__(server_address, _OpenAIRequestHandler)
        self.channel = channel


class _OpenAIRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible endpoints."""

    server: _OpenAIHTTPServer
    server_version = "nanobot-openai/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok"})
            return

        if path == "/v1/models":
            if not self.server.channel.is_request_authorized(self.headers.get("Authorization")):
                self._send_json(401, self.server.channel.error_payload("Unauthorized", "authentication_error"))
                return
            self._send_json(200, self.server.channel.build_models_payload())
            return

        self._send_json(404, self.server.channel.error_payload("Not found", "invalid_request_error"))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/v1/chat/completions":
            self._send_json(404, self.server.channel.error_payload("Not found", "invalid_request_error"))
            return

        is_stream = self.server.channel.request_is_stream(self._peek_body())
        status, payload = self.server.channel.handle_chat_completion_http(
            body=self._read_body(),
            client_ip=self.client_address[0] if self.client_address else "unknown",
            authorization=self.headers.get("Authorization"),
        )
        if status != 200:
            self._send_json(status, payload)
            return

        if is_stream:
            self._send_sse_from_completion(payload)
            return

        self._send_json(status, payload)

    def _peek_body(self) -> bytes:
        """Read request body once and cache it for later handlers."""
        if hasattr(self, "_cached_body"):
            return self._cached_body  # type: ignore[attr-defined]
        body = self._read_body()
        setattr(self, "_cached_body", body)
        return body

    def _read_body(self) -> bytes:
        if hasattr(self, "_cached_body"):
            return self._cached_body  # type: ignore[attr-defined]
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return b""
        if content_length <= 0:
            return b""
        body = self.rfile.read(content_length)
        setattr(self, "_cached_body", body)
        return body

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_sse_from_completion(self, completion: dict[str, Any]) -> None:
        """Send OpenAI-compatible SSE chunks from a completion payload."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # End stream by closing the HTTP connection after [DONE].
        self.send_header("Connection", "close")
        self.end_headers()

        for chunk in self.server.channel.build_stream_chunks(completion):
            data = json.dumps(chunk, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(f"OpenAI API channel HTTP: {fmt % args}")


class OpenAIAPIChannel(BaseChannel):
    """
    OpenAI-compatible channel exposing `/v1/chat/completions`.

    Request format:
    - OpenAI Chat Completions compatible JSON payload.
    - `stream=true` is supported via SSE.
    """

    name = "openai_api"

    def __init__(self, config: OpenAIAPIConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: OpenAIAPIConfig = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: _OpenAIHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._pending: dict[str, asyncio.Future[str]] = {}

    async def start(self) -> None:
        """Start HTTP server and keep running until stopped."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()

        try:
            self._server = _OpenAIHTTPServer((self.config.host, self.config.port), self)
        except OSError as e:
            self._running = False
            logger.error(
                f"Failed to start OpenAI API channel on {self.config.host}:{self.config.port}: {e}"
            )
            return

        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="nanobot-openai-api",
            daemon=True,
        )
        self._server_thread.start()

        logger.info(
            f"OpenAI API channel listening on http://{self.config.host}:{self.config.port}/v1/chat/completions"
        )

        while self._running:
            await asyncio.sleep(0.25)

    async def stop(self) -> None:
        """Stop HTTP server and clear pending requests."""
        self._running = False

        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

        if self._server:
            await asyncio.to_thread(self._server.shutdown)
            await asyncio.to_thread(self._server.server_close)
            self._server = None

        if self._server_thread and self._server_thread.is_alive():
            await asyncio.to_thread(self._server_thread.join, 2.0)
        self._server_thread = None

    async def send(self, msg: OutboundMessage) -> None:
        """Resolve waiting HTTP request with agent response content."""
        request_id = str((msg.metadata or {}).get("request_id", "")).strip()
        if not request_id:
            return

        waiter = self._pending.get(request_id)
        if waiter and not waiter.done():
            waiter.set_result(msg.content or "")

    def is_request_authorized(self, authorization: str | None) -> bool:
        """Validate optional Bearer auth."""
        token = self.config.api_key.strip()
        if not token:
            return True
        if not authorization:
            return False
        return authorization.strip() == f"Bearer {token}"

    def handle_chat_completion_http(
        self,
        body: bytes,
        client_ip: str,
        authorization: str | None,
    ) -> tuple[int, dict[str, Any]]:
        """Parse request and execute it on the asyncio loop."""
        if not self.is_request_authorized(authorization):
            return 401, self.error_payload("Unauthorized", "authentication_error")

        if not body:
            return 400, self.error_payload("Request body is required")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 400, self.error_payload("Request body must be valid JSON")

        if not isinstance(payload, dict):
            return 400, self.error_payload("JSON root must be an object")

        if not self._loop or not self._running:
            return 503, self.error_payload("OpenAI API channel is not ready", "server_error")

        result_future = asyncio.run_coroutine_threadsafe(
            self._handle_chat_completion(payload, client_ip),
            self._loop,
        )
        try:
            result = result_future.result(
                timeout=max(1, int(self.config.request_timeout_seconds) + 5)
            )
            return 200, result
        except FutureTimeoutError:
            result_future.cancel()
            return 504, self.error_payload("Upstream timeout", "server_error")
        except ValueError as e:
            return 400, self.error_payload(str(e), "invalid_request_error")
        except PermissionError as e:
            return 403, self.error_payload(str(e), "insufficient_permissions")
        except Exception as e:  # pragma: no cover - defensive safeguard
            logger.exception(f"OpenAI API request failed: {e}")
            return 500, self.error_payload("Internal server error", "server_error")

    @staticmethod
    def request_is_stream(body: bytes) -> bool:
        """Best-effort check whether request asks for stream mode."""
        if not body:
            return False
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(payload, dict) and payload.get("stream") is True

    async def _handle_chat_completion(
        self,
        payload: dict[str, Any],
        client_ip: str,
    ) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Field 'messages' must be a non-empty array.")

        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            model = self.config.default_model
        model = model.strip()

        sender_id = self._resolve_sender(payload, client_ip)
        chat_id = self._resolve_chat_id(payload, sender_id)
        if not self.is_allowed(sender_id):
            raise PermissionError(
                f"Sender '{sender_id}' is not allowed for channel '{self.name}'."
            )

        prompt = self._messages_to_prompt(messages)
        request_id = uuid.uuid4().hex
        response_waiter = asyncio.get_running_loop().create_future()
        self._pending[request_id] = response_waiter

        try:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=prompt,
                metadata={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "source": "openai_api",
                    # OpenAI clients usually send full conversation history in each request.
                    # Mark request as stateless so agent-side session history is not reused.
                    "stateless": True,
                    # Return only after all spawned subagents complete.
                    "wait_for_subagents": True,
                },
            )
            answer = await asyncio.wait_for(
                response_waiter,
                timeout=max(1, int(self.config.request_timeout_seconds)),
            )
        finally:
            self._pending.pop(request_id, None)

        return self._build_completion_payload(model=model, content=answer)

    def build_models_payload(self) -> dict[str, Any]:
        """Return OpenAI-compatible model list payload."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.default_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }

    @staticmethod
    def error_payload(message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
        """Return OpenAI-style error payload."""
        return {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        }

    @staticmethod
    def _resolve_sender(payload: dict[str, Any], client_ip: str) -> str:
        user = payload.get("user")
        if isinstance(user, str) and user.strip():
            return user.strip()
        return client_ip or "openai-client"

    @staticmethod
    def _resolve_chat_id(payload: dict[str, Any], fallback_sender: str) -> str:
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
        user = payload.get("user")
        if isinstance(user, str) and user.strip():
            return user.strip()
        return fallback_sender or "openai-default"

    @classmethod
    def _messages_to_prompt(cls, messages: list[Any]) -> str:
        rendered: list[tuple[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().lower() or "user"
            content = cls._content_to_text(item.get("content"))
            if content:
                rendered.append((role, content))

        if not rendered:
            raise ValueError("No textual content found in 'messages'.")

        if len(rendered) == 1 and rendered[0][0] == "user":
            return rendered[0][1]

        history = []
        for role, text in rendered:
            history.append(f"[{role}] {text}")
        return "\n\n".join(history)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
            return "\n".join(texts).strip()

        return ""

    @staticmethod
    def _build_completion_payload(model: str, content: str) -> dict[str, Any]:
        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def build_stream_chunks(completion: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a completion payload into OpenAI-compatible stream chunks."""
        completion_id = str(completion.get("id") or f"chatcmpl-{uuid.uuid4().hex}")
        created = int(completion.get("created") or int(time.time()))
        model = str(completion.get("model") or "")

        content = ""
        finish_reason = "stop"
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            if isinstance(first_choice, dict):
                finish_reason = str(first_choice.get("finish_reason") or "stop")
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = str(message.get("content") or "")

        chunks: list[dict[str, Any]] = [
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        ]

        if content:
            chunks.append(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
            )

        chunks.append(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        return chunks

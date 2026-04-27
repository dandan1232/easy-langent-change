#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""个人记忆助手 Web 后端。

使用 Python 标准库提供 HTTP 服务，避免为了演示额外引入 Web 框架。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from assistant import LongTermMemoryStore, PersonalMemoryAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_MEMORY_FILE = PROJECT_ROOT / "memory_store.json"


class AssistantService:
    def __init__(self, memory_file: Path, window_size: int) -> None:
        self.memory_file = memory_file
        self.window_size = window_size
        self._assistant: PersonalMemoryAssistant | None = None

    @property
    def assistant(self) -> PersonalMemoryAssistant:
        if self._assistant is None:
            self._assistant = PersonalMemoryAssistant(
                memory_file=self.memory_file,
                window_size=self.window_size,
            )
        return self._assistant

    def chat(self, message: str) -> dict[str, Any]:
        return self.assistant.chat(message)

    def memories(self) -> dict[str, Any]:
        store = (
            self.assistant.long_term_memory
            if self._assistant is not None
            else LongTermMemoryStore(self.memory_file)
        )
        return {
            "memory_text": store.as_prompt_text(),
            "memory_data": store.data,
        }

    def clear(self) -> None:
        if self._assistant is not None:
            self._assistant.clear_all()
            return
        LongTermMemoryStore(self.memory_file).clear()


class PersonalMemoryRequestHandler(BaseHTTPRequestHandler):
    service: AssistantService

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/memories":
            self._send_json(self.service.memories())
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            self._handle_chat()
            return
        if path == "/api/clear":
            self.service.clear()
            self._send_json({"ok": True})
            return
        self._send_json({"error": "接口不存在。"}, HTTPStatus.NOT_FOUND)

    def _handle_chat(self) -> None:
        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"error": "请输入要发送的内容。"}, HTTPStatus.BAD_REQUEST)
                return

            result = self.service.chat(message)
            self._send_json(
                {
                    "reply": result["answer"]["reply"],
                    "visible_reply": result["visible_reply"],
                    "suggestions": result["answer"]["suggestions"],
                    "matched_memories": result["answer"]["matched_memories"],
                    "need_follow_up": result["answer"]["need_follow_up"],
                    "follow_up_question": result["answer"]["follow_up_question"],
                    "new_memories": result["new_memories"],
                    "memories": self.service.memories(),
                }
            )
        except RuntimeError as error:
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        except Exception as error:
            self._send_json({"error": f"请求处理失败：{error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_static(self, request_path: str) -> None:
        if request_path == "/":
            file_path = FRONTEND_DIR / "index.html"
        elif request_path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        else:
            relative_path = unquote(request_path.lstrip("/"))
            if relative_path.startswith("frontend/"):
                relative_path = relative_path.removeprefix("frontend/")
            file_path = (FRONTEND_DIR / relative_path).resolve()
            if not str(file_path).startswith(str(FRONTEND_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[PersonalMemoryAssistant] {self.address_string()} - {format % args}")


def run_server() -> None:
    parser = argparse.ArgumentParser(description="个人记忆助手 Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8010, help="监听端口")
    parser.add_argument(
        "--memory-file",
        default=str(DEFAULT_MEMORY_FILE),
        help="长期记忆 JSON 文件路径",
    )
    parser.add_argument("--window-size", type=int, default=4, help="保留最近几轮对话")
    args = parser.parse_args()

    PersonalMemoryRequestHandler.service = AssistantService(
        memory_file=Path(args.memory_file),
        window_size=args.window_size,
    )
    server = ThreadingHTTPServer((args.host, args.port), PersonalMemoryRequestHandler)
    print(f"个人记忆助手已启动：http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务。")
    server.serve_forever()


if __name__ == "__main__":
    run_server()

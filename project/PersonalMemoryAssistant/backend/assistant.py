#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
个人记忆助手：基于长期记忆 + 窗口对话记忆的多轮问答 Demo。

运行前请在项目根目录或当前目录配置 .env：
API_KEY=你的模型 API Key
BASE_URL=模型服务地址，可选
MODEL=模型名称，例如 deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    ConversationBufferWindowMemory = None


MEMORY_CATEGORIES = {
    "preferences": "偏好：喜欢/不喜欢、常用选择、口味、风格",
    "plans": "计划：近期想做、想买、想学、想去",
    "constraints": "限制：身体状态、预算、时间、禁忌",
    "facts": "事实：稳定个人信息、重要日期、常用信息",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WindowConversationMemory:
    """兼容 LangChain 1.x 中 ConversationBufferWindowMemory 迁移后的最小窗口记忆。"""

    def __init__(self, k: int = 4, memory_key: str = "chat_history") -> None:
        self.k = k
        self.memory_key = memory_key
        self.messages: list[Any] = []

    def load_memory_variables(self, _: dict[str, Any]) -> dict[str, list[Any]]:
        return {self.memory_key: self.messages[-2 * self.k :]}

    def save_context(self, inputs: dict[str, str], outputs: dict[str, str]) -> None:
        self.messages.append(HumanMessage(content=inputs["user_input"]))
        self.messages.append(AIMessage(content=outputs["output"]))
        self.messages = self.messages[-2 * self.k :]

    def clear(self) -> None:
        self.messages.clear()


class LongTermMemoryStore:
    """用 JSON 文件保存用户长期记忆。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> dict[str, list[dict[str, str]]]:
        if not self.path.exists():
            return {category: [] for category in MEMORY_CATEGORIES}

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for category in MEMORY_CATEGORIES:
            data.setdefault(category, [])
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        self.data = {category: [] for category in MEMORY_CATEGORIES}
        self.save()

    def as_prompt_text(self) -> str:
        lines: list[str] = []
        for category, description in MEMORY_CATEGORIES.items():
            items = self.data.get(category, [])
            if not items:
                continue
            lines.append(f"【{category}】{description}")
            for index, item in enumerate(items, 1):
                lines.append(f"{index}. {item['content']}（记录于 {item['updated_at']}）")
        return "\n".join(lines) if lines else "暂无长期记忆。"

    def add_memories(self, memories: list[dict[str, Any]]) -> list[str]:
        added: list[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        for memory in memories:
            category = str(memory.get("category", "")).strip()
            content = str(memory.get("content", "")).strip()
            if category not in MEMORY_CATEGORIES or not content:
                continue

            if self._contains(category, content):
                continue

            self.data[category].append(
                {
                    "content": content,
                    "reason": str(memory.get("reason", "")).strip(),
                    "updated_at": now,
                }
            )
            added.append(content)

        if added:
            self.save()
        return added

    def _contains(self, category: str, content: str) -> bool:
        normalized = content.replace(" ", "")
        for item in self.data.get(category, []):
            existing = item.get("content", "").replace(" ", "")
            if normalized == existing or normalized in existing or existing in normalized:
                return True
        return False


class PersonalMemoryAssistant:
    def __init__(self, memory_file: Path, window_size: int = 4) -> None:
        env_file = PROJECT_ROOT / ".env"
        backend_env_file = Path(__file__).resolve().parent / ".env"
        if backend_env_file.exists():
            load_dotenv(backend_env_file)
        elif env_file.exists():
            load_dotenv(env_file)
        else:
            load_dotenv()
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
        model = os.getenv("MODEL", "deepseek-chat")

        if not api_key:
            raise RuntimeError("请先在 .env 中配置 API_KEY 或 OPENAI_API_KEY。")

        self.long_term_memory = LongTermMemoryStore(memory_file)
        self.window_memory = self._create_window_memory(window_size)
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.3,
        )

        self.answer_parser = JsonOutputParser()
        self.extract_parser = JsonOutputParser()
        self.answer_chain = self._build_answer_chain()
        self.memory_extract_chain = self._build_memory_extract_chain()

    def _create_window_memory(self, window_size: int) -> Any:
        if ConversationBufferWindowMemory is None:
            return WindowConversationMemory(k=window_size, memory_key="chat_history")

        return ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            input_key="user_input",
            output_key="output",
            return_messages=True,
        )

    def _build_answer_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个个人记忆助手，负责在自然对话中理解用户当前需求，并结合长期记忆给出有用提醒。\n"
                    "你只能使用用户本人提供过的信息，不要编造记忆。\n"
                    "如果当前问题指代不清，或长期记忆中存在多个可能对象，请主动追问。\n"
                    "今天是：{today}\n\n"
                    "长期记忆：\n{long_term_memory}\n\n"
                    "输出要求：\n"
                    "1. reply 要自然、简洁，像日常助手提醒，不要机械复述。\n"
                    "2. matched_memories 只放确实用到的长期记忆。\n"
                    "3. suggestions 给 0-3 条可执行建议。\n"
                    "4. 如果需要追问，need_follow_up 为 true，并填写 follow_up_question。\n"
                    "{format_instructions}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_input}"),
            ]
        ).partial(format_instructions=self.answer_parser.get_format_instructions())

        return (
            RunnableLambda(self._load_answer_inputs)
            | prompt
            | self.llm
            | self.answer_parser
            | RunnableLambda(self._normalize_answer)
        )

    def _build_memory_extract_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是记忆抽取器。请从用户本轮输入中提取值得长期保存的个人记忆。\n"
                    "只保存对未来回答有帮助的信息，例如偏好、计划、限制、稳定事实。\n"
                    "不要保存闲聊、临时问题、助手回答、无意义寒暄。\n"
                    "如果没有新记忆，返回空列表。\n\n"
                    "可用分类：\n{category_descriptions}\n\n"
                    "已有长期记忆：\n{long_term_memory}\n\n"
                    "今天是：{today}\n"
                    "{format_instructions}",
                ),
                (
                    "human",
                    "用户输入：{user_input}\n\n"
                    "助手回复：{assistant_reply}\n\n"
                    "请输出 JSON，字段为 new_memories。new_memories 是数组，每项包含 category、content、reason。",
                ),
            ]
        ).partial(format_instructions=self.extract_parser.get_format_instructions())

        return (
            RunnableLambda(self._load_extract_inputs)
            | prompt
            | self.llm
            | self.extract_parser
            | RunnableLambda(self._normalize_extracted_memories)
        )

    def _load_answer_inputs(self, inputs: dict[str, str]) -> dict[str, Any]:
        memory_variables = self.window_memory.load_memory_variables({})
        return {
            "user_input": inputs["user_input"],
            "chat_history": memory_variables.get("chat_history", []),
            "long_term_memory": self.long_term_memory.as_prompt_text(),
            "today": datetime.now().strftime("%Y-%m-%d"),
        }

    def _load_extract_inputs(self, inputs: dict[str, str]) -> dict[str, Any]:
        descriptions = "\n".join(
            f"- {category}: {description}"
            for category, description in MEMORY_CATEGORIES.items()
        )
        return {
            "user_input": inputs["user_input"],
            "assistant_reply": inputs["assistant_reply"],
            "long_term_memory": self.long_term_memory.as_prompt_text(),
            "category_descriptions": descriptions,
            "today": datetime.now().strftime("%Y-%m-%d"),
        }

    def _normalize_answer(self, parsed: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}

        return {
            "reply": str(parsed.get("reply", "")).strip() or "我还需要更多信息才能准确回答。",
            "matched_memories": self._as_string_list(parsed.get("matched_memories", [])),
            "suggestions": self._as_string_list(parsed.get("suggestions", []))[:3],
            "need_follow_up": bool(parsed.get("need_follow_up", False)),
            "follow_up_question": str(parsed.get("follow_up_question", "")).strip(),
        }

    def _normalize_extracted_memories(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(parsed, dict):
            return []

        memories = parsed.get("new_memories", [])
        if not isinstance(memories, list):
            return []
        return [memory for memory in memories if isinstance(memory, dict)]

    def _as_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def chat(self, user_input: str) -> dict[str, Any]:
        answer = self.answer_chain.invoke({"user_input": user_input})
        visible_reply = self.format_answer(answer)

        extracted = self.memory_extract_chain.invoke(
            {
                "user_input": user_input,
                "assistant_reply": answer["reply"],
            }
        )
        new_memories = self.long_term_memory.add_memories(extracted)

        self.window_memory.save_context(
            {"user_input": user_input},
            {"output": visible_reply},
        )

        return {
            "answer": answer,
            "visible_reply": visible_reply,
            "new_memories": new_memories,
        }

    def format_answer(self, answer: dict[str, Any]) -> str:
        lines = [answer["reply"]]

        if answer["suggestions"]:
            lines.append("\n建议：")
            for suggestion in answer["suggestions"]:
                lines.append(f"- {suggestion}")

        if answer["need_follow_up"] and answer["follow_up_question"]:
            lines.append(f"\n追问：{answer['follow_up_question']}")

        return "\n".join(lines)

    def print_memories(self) -> None:
        print("\n当前长期记忆：")
        print(self.long_term_memory.as_prompt_text())

    def clear_all(self) -> None:
        self.window_memory.clear()
        self.long_term_memory.clear()


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="个人记忆助手 CLI")
    parser.add_argument(
        "--memory-file",
        default=str(PROJECT_ROOT / "memory_store.json"),
        help="长期记忆 JSON 文件路径",
    )
    parser.add_argument("--window-size", type=int, default=4, help="保留最近几轮对话")
    args = parser.parse_args()

    assistant = PersonalMemoryAssistant(
        memory_file=Path(args.memory_file),
        window_size=args.window_size,
    )

    print("个人记忆助手已启动。输入 /exit 退出，/memories 查看记忆，/clear 清空记忆。")
    while True:
        user_input = input("\n你：").strip()
        if not user_input:
            continue
        if user_input in {"/exit", "exit", "退出"}:
            break
        if user_input == "/memories":
            assistant.print_memories()
            continue
        if user_input == "/clear":
            assistant.clear_all()
            print("已清空长期记忆和本轮窗口对话。")
            continue

        result = assistant.chat(user_input)
        print(f"\n助手：{result['visible_reply']}")
        if result["new_memories"]:
            print("\n已更新记忆：")
            for memory in result["new_memories"]:
                print(f"- {memory}")


if __name__ == "__main__":
    run_cli()

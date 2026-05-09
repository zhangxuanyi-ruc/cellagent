"""LLM API 客户端抽象接口与 Mock 实现.

设计原则:
  - 抽象基类 LLMClient 定义统一接口，不依赖具体提供商
  - MockLLMClient 用于状态机开发和测试（无需真实 API key）
  - 后续接入 OpenAI/Claude/本地模型时，只需实现子类

使用方式:
  from src.llm.clients import LLMClient, MockLLMClient

  # 开发测试期
  llm = MockLLMClient(fixture_dir="tests/fixtures/llm")

  # 生产环境（后续接入）
  # llm = OpenAIClient(model="gpt-4o", api_key=...)
  # llm = ClaudeClient(model="claude-3-5-sonnet", api_key=...)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LLMClient(ABC):
    """LLM 客户端抽象基类.

    所有具体实现（OpenAI、Claude、本地模型）必须实现 chat() 方法。
    """

    def __init__(self, model: str, temperature: float = 0.3, timeout: int = 60):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str | dict:
        """发送对话请求.

        Args:
            messages: OpenAI 格式的消息列表，如
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            json_mode: 是否强制返回 JSON 格式

        Returns:
            json_mode=False: 返回文本字符串
            json_mode=True:  返回解析后的 dict
        """
        raise NotImplementedError

    def batch_chat(
        self,
        messages_list: list[list[dict[str, str]]],
        json_mode: bool = False,
    ) -> list[str | dict]:
        """批量发送对话请求（串行，带简单重试）."""
        return [self.chat(msgs, json_mode=json_mode) for msgs in messages_list]


class MockLLMClient(LLMClient):
    """Mock LLM 客户端，用于开发和单测.

    根据 prompt 内容的关键词匹配 fixture 文件，返回固定响应。
    无需 API key，离线可用。
    """

    def __init__(
        self,
        fixture_dir: str | Path = "tests/fixtures/llm",
        model: str = "mock",
        temperature: float = 0.0,
    ):
        super().__init__(model=model, temperature=temperature)
        self.fixture_dir = Path(fixture_dir)
        self._cache: dict[str, str | dict] = {}

    def _load_fixture(self, name: str) -> str | dict | None:
        """加载 fixture 文件."""
        path = self.fixture_dir / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("response", data)

        path_txt = self.fixture_dir / f"{name}.txt"
        if path_txt.exists():
            return path_txt.read_text(encoding="utf-8")

        return None

    def _match_fixture(self, messages: list[dict[str, str]]) -> str | dict | None:
        """根据消息内容匹配 fixture."""
        # 简单的关键词匹配
        content = " ".join(m.get("content", "") for m in messages).lower()

        if "initial" in content or "infer" in content:
            return self._load_fixture("initial_inference")
        if "evaluat" in content or "score" in content:
            return self._load_fixture("evaluation")
        if "reflect" in content:
            return self._load_fixture("reflection")
        if "function" in content and "consist" in content:
            return self._load_fixture("function_judge")
        if "tissue" in content:
            return self._load_fixture("tissue_judge")
        if "conflict" in content:
            return self._load_fixture("conflict_arbitration")

        return self._load_fixture("default")

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str | dict:
        """返回匹配的 fixture 响应，若无匹配则返回默认响应."""
        # 用消息内容的 hash 作为缓存键
        key = hash(json.dumps(messages, sort_keys=True))
        if key in self._cache:
            result = self._cache[key]
        else:
            result = self._match_fixture(messages)
            if result is None:
                result = self._default_response(json_mode)
            self._cache[key] = result

        if json_mode and isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": "invalid json", "raw": result}
        return result

    def _default_response(self, json_mode: bool) -> str | dict:
        """默认响应."""
        if json_mode:
            return {
                "score": 20,
                "reasoning": "Mock response: default pass",
                "veto_triggered": False,
            }
        return "Mock response: this is a placeholder from MockLLMClient."

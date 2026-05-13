"""LLM API 客户端抽象接口与具体实现.

设计原则:
  - 抽象基类 LLMClient 定义统一接口，不依赖具体提供商
  - MockLLMClient 用于状态机开发和测试（无需真实 API key）
  - OpenAICompatibleClient 对接任何 OpenAI-compatible API（本地 vLLM、Tailscale 等）

使用方式:
  from src.llm.clients import LLMClient, MockLLMClient, OpenAICompatibleClient

  # 开发测试期
  llm = MockLLMClient(fixture_dir="tests/fixtures/llm")

  # 本地部署（OpenAI-compatible）
  llm = OpenAICompatibleClient(
      base_url="https://localagent.tail053d0c.ts.net/v1",
      model="default",
      tailscale_host="localagent.tail053d0c.ts.net",
      tailscale_ip="100.105.94.49",
  )
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


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible API 客户端.

    支持本地 vLLM、TGI、Tailscale 代理等任何 OpenAI-compatible endpoint。
    可选的 Tailscale hostname resolution patch，用于通过内网 IP 连接 Tailscale 节点。

    Args:
        base_url: API base URL, e.g. "https://host/v1" or "http://localhost:8000/v1"
        model: 模型名称，若为 None 则自动调用 /models 获取第一个可用模型
        api_key: API key，本地部署通常传 "unused" 或任意字符串
        temperature: 采样温度
        timeout: 请求超时秒数
        max_retries: 重试次数
        tailscale_host: Tailscale hostname（如需 patch DNS）
        tailscale_ip: Tailscale 内网 IP（如需 patch DNS）
        enable_thinking: 是否启用模型的思考过程（传给 extra_body）
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        api_key: str = "unused",
        temperature: float = 0.3,
        timeout: int = 120,
        max_retries: int = 3,
        tailscale_host: str | None = None,
        tailscale_ip: str | None = None,
        enable_thinking: bool = False,
        socks5_proxy: str | None = None,
        verify_ssl: bool = True,
    ):
        super().__init__(model=model or "auto", temperature=temperature, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.tailscale_host = tailscale_host
        self.tailscale_ip = tailscale_ip
        self.enable_thinking = enable_thinking
        self.socks5_proxy = socks5_proxy
        self.verify_ssl = verify_ssl
        self._client = self._create_client()
        self._model: str | None = model

    def _create_client(self):
        import socket

        import httpx
        from openai import OpenAI

        if self.tailscale_host and self.tailscale_ip:
            orig_getaddrinfo = socket.getaddrinfo
            host, ip = self.tailscale_host, self.tailscale_ip

            def _patched_getaddrinfo(hostname, port, *args, **kwargs):
                if hostname == host:
                    return orig_getaddrinfo(ip, port, *args, **kwargs)
                return orig_getaddrinfo(hostname, port, *args, **kwargs)

            socket.getaddrinfo = _patched_getaddrinfo

        if self.socks5_proxy:
            http_client = httpx.Client(
                proxy=self.socks5_proxy,
                trust_env=False,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        else:
            http_client = httpx.Client(
                trust_env=False,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=http_client,
            max_retries=self.max_retries,
        )

    @property
    def model(self) -> str:
        if self._model is None:
            models = self._client.models.list()
            self._model = models.data[0].id if models.data else "default"
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str | dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        extra_body: dict[str, Any] = {}
        if not self.enable_thinking:
            extra_body["enable_thinking"] = False
        if extra_body:
            kwargs["extra_body"] = extra_body

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # Qwen3 模型输出在 reasoning 字段，不在 content
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        text = content or reasoning

        if json_mode:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "invalid json", "raw": text}
        return text


def build_llm_client_from_config(config: dict[str, Any]) -> LLMClient | None:
    """Build the configured LLM client for judge/reflection stages.

    Returns None when llm.enabled is explicitly false. The default provider is
    openai_compatible because local Qwen/vLLM endpoints expose that interface.
    """
    llm_cfg = config.get("llm", {}) or {}
    if llm_cfg.get("enabled") is False:
        return None
    provider = llm_cfg.get("provider", "openai_compatible")
    if provider == "mock":
        mock_cfg = llm_cfg.get("mock", {}) or {}
        return MockLLMClient(
            fixture_dir=mock_cfg.get("fixture_dir", "tests/fixtures/llm"),
            model=mock_cfg.get("model", "mock"),
            temperature=float(mock_cfg.get("temperature", 0.0)),
        )
    if provider != "openai_compatible":
        raise ValueError(f"Unsupported llm.provider: {provider!r}")
    raw = llm_cfg.get("openai_compatible", {}) or {}
    if not raw.get("base_url"):
        raise ValueError("llm.openai_compatible.base_url is required when LLM judge is enabled.")
    return OpenAICompatibleClient(
        base_url=raw["base_url"],
        model=raw.get("model"),
        api_key=raw.get("api_key", "unused"),
        temperature=float(raw.get("temperature", 0.3)),
        timeout=int(raw.get("timeout", 120)),
        max_retries=int(raw.get("max_retries", 3)),
        tailscale_host=raw.get("tailscale_host"),
        tailscale_ip=raw.get("tailscale_ip"),
        enable_thinking=bool(raw.get("enable_thinking", False)),
        socks5_proxy=raw.get("socks5_proxy"),
        verify_ssl=bool(raw.get("verify_ssl", True)),
    )

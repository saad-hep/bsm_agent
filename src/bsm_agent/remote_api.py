"""Remote chat-model helpers for BSM agent integrations.

This mirrors the remote Open WebUI and OpenAI-compatible support used in
``rough_agent.py`` so the same API base can be reused from the ``bsm_agent``
package itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import os
import re
import subprocess
from typing import Any


PREFERRED_MODELS = (
    "qwen3:8b",
    "qwen3.6:35b-a3b",
    "qwen3.6:27b",
    "qwen3.5:35b",
)

@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str
    api_base: str | None = None
    api_key: str | None = None
    api_email: str | None = None
    api_password: str | None = None

def _normalize_remote_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return None
    return cleaned


def normalize_openai_base_url(base_url: str | None) -> str | None:
    cleaned = _normalize_remote_base_url(base_url)
    if cleaned and not cleaned.endswith("/v1"):
        cleaned = f"{cleaned}/v1"
    return cleaned


def resolve_openwebui_timeout() -> float:
    raw_value = os.environ.get("BSM_AGENT_OPENWEBUI_TIMEOUT")
    if raw_value is None or not raw_value.strip():
        return 300.0
    try:
        timeout = float(raw_value)
    except ValueError:
        return 300.0
    return timeout if timeout > 0 else 300.0


def normalize_model_alias(model: str) -> str:
    model = model.strip()
    qwen_match = re.fullmatch(r"qwen:(\d+(?:\.\d+)?):(.*)", model)
    if qwen_match:
        return f"qwen{qwen_match.group(1)}:{qwen_match.group(2)}"
    return model


def list_installed_ollama_models() -> list[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return []

    models: list[str] = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def pick_default_model(installed: list[str]) -> str:
    for candidate in PREFERRED_MODELS:
        if candidate in installed:
            return candidate
    if installed:
        return installed[0]
    raise RuntimeError(
        "No Ollama models are installed. Install one first, e.g. "
        "'ollama pull qwen3:8b' or 'ollama pull qwen3.5:35b'."
    )


def format_model_error(model: str | None) -> str:
    installed = list_installed_ollama_models()
    requested = "auto" if model is None or not model.strip() else model
    normalized = normalize_model_alias(requested)
    candidates = difflib.get_close_matches(normalized, installed, n=3, cutoff=0.4)

    parts = [f"Invalid Ollama model '{requested}'."]
    if normalized != requested:
        parts.append(f"Normalized form tried: '{normalized}'.")
    if installed:
        parts.append("Installed models: " + ", ".join(installed) + ".")
    parts.append(
        "Remote models can be selected with "
        "'remote:<model>', 'openai:<model>', or 'anthropic:<model>' plus the required auth."
    )
    if candidates:
        parts.append("Closest matches: " + ", ".join(candidates) + ".")
    return " ".join(parts)


def resolve_remote_settings(
    api_base: str | None = None,
    api_key: str | None = None,
    api_email: str | None = None,
    api_password: str | None = None,
) -> dict[str, str | None]:
    resolved_base = _normalize_remote_base_url(
        api_base
        or os.environ.get("BSM_AGENT_OPENAI_BASE_URL")
    )
    resolved_key = (
        api_key
        or os.environ.get("BSM_AGENT_OPENAI_API_KEY")
    )
    resolved_email = (
        api_email
        or os.environ.get("BSM_AGENT_OPENAI_EMAIL")
    )
    resolved_password = (
        api_password
        or os.environ.get("BSM_AGENT_OPENAI_PASSWORD")
    )

    if not resolved_base:
        raise RuntimeError(
            "Remote model selected but no API base URL was provided. "
            "Set BSM_AGENT_OPENAI_BASE_URL or pass api_base."
        )
    if not resolved_key and not (resolved_email and resolved_password):
        raise RuntimeError(
            "Remote model selected but no authentication was provided. "
            "Set BSM_AGENT_OPENAI_API_KEY, or provide "
            "BSM_AGENT_OPENAI_EMAIL and BSM_AGENT_OPENAI_PASSWORD."
        )
    return {
        "api_base": resolved_base,
        "api_key": resolved_key,
        "api_email": resolved_email,
        "api_password": resolved_password,
    }


def resolve_openai_settings(
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, str | None]:
    resolved_base = _normalize_remote_base_url(
        api_base
        or os.environ.get("BSM_AGENT_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    resolved_key = (
        api_key
        or os.environ.get("BSM_AGENT_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not resolved_key:
        raise RuntimeError(
            "OpenAI model selected but no API key was provided. "
            "Set BSM_AGENT_OPENAI_API_KEY or OPENAI_API_KEY."
        )

    return {
        "api_base": resolved_base,
        "api_key": resolved_key,
    }


def resolve_anthropic_settings(
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, str | None]:
    resolved_base = _normalize_remote_base_url(
        api_base
        or os.environ.get("BSM_AGENT_ANTHROPIC_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
    )
    resolved_key = (
        api_key
        or os.environ.get("BSM_AGENT_ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )

    if not resolved_key:
        raise RuntimeError(
            "Anthropic model selected but no API key was provided. "
            "Set BSM_AGENT_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY."
        )

    return {
        "api_base": resolved_base,
        "api_key": resolved_key,
    }


def resolve_model_target(
    model: str | None,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    api_email: str | None = None,
    api_password: str | None = None,
    installed_models: list[str] | None = None,
) -> ModelTarget:
    installed = installed_models if installed_models is not None else list_installed_ollama_models()
    installed_map = {name.lower(): name for name in installed}

    requested = (model or "").strip()
    if not requested:
        return ModelTarget(provider="ollama", model=pick_default_model(installed))

    normalized = normalize_model_alias(requested)
    lowered = normalized.lower()

    if requested.startswith("openai:"):
        settings = resolve_openai_settings(api_base, api_key)
        return ModelTarget(
            provider="openai",
            model=requested.split(":", 1)[1],
            api_base=settings["api_base"],
            api_key=settings["api_key"],
        )

    if requested.startswith("anthropic:"):
        settings = resolve_anthropic_settings(api_base, api_key)
        return ModelTarget(
            provider="anthropic",
            model=requested.split(":", 1)[1],
            api_base=settings["api_base"],
            api_key=settings["api_key"],
        )

    if normalized in installed:
        return ModelTarget(provider="ollama", model=normalized)
    if lowered in installed_map:
        return ModelTarget(provider="ollama", model=installed_map[lowered])

    if requested.startswith("remote:"):
        settings = resolve_remote_settings(api_base, api_key, api_email, api_password)
        return ModelTarget(
            provider="openwebui",
            model=requested.split(":", 1)[1],
            api_base=settings["api_base"],
            api_key=settings["api_key"],
            api_email=settings["api_email"],
            api_password=settings["api_password"],
        )

    return ModelTarget(provider="ollama", model=normalized)


class OpenAICompatibleChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0,
        tools: list[Any] | None = None,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.tools = list(tools or [])
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def bind_tools(self, tools: list[Any]):
        return OpenAICompatibleChatModel(
            self.model,
            api_key=self.client.api_key,
            base_url=str(self.client.base_url),
            temperature=self.temperature,
            tools=tools,
        )

    def _convert_message(self, message: Any) -> dict[str, Any]:
        role = getattr(message, "type", None)
        if role == "system":
            return {"role": "system", "content": message.content}
        if role == "human":
            return {"role": "user", "content": message.content}
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if role == "ai":
            payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in tool_calls
                ]
            return payload
        return {"role": "user", "content": str(message)}

    def _convert_tools(self) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool_obj in self.tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                        "parameters": tool_obj.tool_call_schema.model_json_schema(),
                    },
                }
            )
        return converted

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[self._convert_message(message) for message in messages],
            tools=self._convert_tools() or None,
        )
        message = response.choices[0].message
        tool_calls = []
        for tool_call in message.tool_calls or []:
            args = tool_call.function.arguments or "{}"
            tool_calls.append(
                {
                    "name": tool_call.function.name,
                    "args": json.loads(args),
                    "id": tool_call.id,
                    "type": "tool_call",
                }
            )
        return {
            "content": message.content or "",
            "tool_calls": tool_calls,
        }


class AnthropicChatModel:
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0,
        tools: list[Any] | None = None,
        max_tokens: int = 4096,
    ):
        from anthropic import Anthropic

        self.model = model
        self.temperature = temperature
        self.tools = list(tools or [])
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    def bind_tools(self, tools: list[Any]):
        return AnthropicChatModel(
            self.model,
            api_key=self.client.api_key,
            base_url=str(self.client.base_url),
            temperature=self.temperature,
            tools=tools,
            max_tokens=self.max_tokens,
        )

    def _convert_tools(self) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool_obj in self.tools:
            converted.append(
                {
                    "name": tool_obj.name,
                    "description": tool_obj.description or "",
                    "input_schema": tool_obj.tool_call_schema.model_json_schema(),
                }
            )
        return converted

    def _message_to_payload(
        self,
        message: Any,
    ) -> tuple[str | None, dict[str, Any] | None]:
        role = getattr(message, "type", None)
        if role == "system":
            return message.content, None
        if role == "human":
            return None, {"role": "user", "content": [{"type": "text", "text": message.content}]}
        if role == "tool":
            return None, {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content,
                    }
                ],
            }
        if role == "ai":
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in getattr(message, "tool_calls", None) or []:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call.get("args", {}),
                    }
                )
            return None, {"role": "assistant", "content": content or [{"type": "text", "text": ""}]}
        return None, {"role": "user", "content": [{"type": "text", "text": str(message)}]}

    def _convert_messages(self, messages: list[Any]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            system_text, payload = self._message_to_payload(message)
            if system_text is not None:
                system_parts.append(system_text)
                continue
            if payload is None:
                continue
            if converted and converted[-1]["role"] == payload["role"]:
                converted[-1]["content"].extend(payload["content"])
            else:
                converted.append(payload)
        system = "\n\n".join(part for part in system_parts if part.strip()) or None
        return system, converted

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        system, converted_messages = self._convert_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=converted_messages,
            tools=self._convert_tools() or None,
        )

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "name": block.name,
                        "args": dict(block.input),
                        "id": block.id,
                        "type": "tool_call",
                    }
                )

        return {
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
        }


class OpenWebUIChatModel:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        email: str | None = None,
        password: str | None = None,
        temperature: float = 0,
        tools: list[Any] | None = None,
    ):
        import httpx

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.tools = list(tools or [])
        self.email = email
        self.password = password
        self.timeout = resolve_openwebui_timeout()
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self.api_key = api_key or self._login_for_token()

    def _login_for_token(self) -> str:
        if not self.email or not self.password:
            raise RuntimeError(
                "Open WebUI requires authentication. Provide an API token via "
                "BSM_AGENT_OPENAI_API_KEY/api_key, or provide "
                "BSM_AGENT_OPENAI_EMAIL and BSM_AGENT_OPENAI_PASSWORD."
            )

        response = self.client.post(
            "/api/v1/auths/signin",
            json={"email": self.email, "password": self.password},
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token") or payload.get("access_token")
        if not token and isinstance(payload.get("data"), dict):
            token = payload["data"].get("token") or payload["data"].get("access_token")
        if not token:
            raise RuntimeError("Open WebUI sign-in succeeded but no token was returned.")
        return token

    def bind_tools(self, tools: list[Any]):
        return OpenWebUIChatModel(
            self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            email=self.email,
            password=self.password,
            temperature=self.temperature,
            tools=tools,
        )

    def _convert_message(self, message: Any) -> dict[str, Any]:
        role = getattr(message, "type", None)
        if role == "system":
            return {"role": "system", "content": message.content}
        if role == "human":
            return {"role": "user", "content": message.content}
        if role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
        if role == "ai":
            payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in tool_calls
                ]
            return payload
        return {"role": "user", "content": str(message)}

    def _convert_tools(self) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for tool_obj in self.tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_obj.name,
                        "description": tool_obj.description or "",
                        "parameters": tool_obj.tool_call_schema.model_json_schema(),
                    },
                }
            )
        return converted

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        import httpx

        try:
            response = self.client.post(
                "/api/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [self._convert_message(message) for message in messages],
                    "tools": self._convert_tools() or None,
                },
            )
        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                "Open WebUI request timed out while waiting for the model response "
                f"after {self.timeout:g} seconds. Increase BSM_AGENT_OPENWEBUI_TIMEOUT "
                "if the remote model is slow to answer."
            ) from exc
        if response.status_code >= 4000:
            detail = response.text.strip()
            if len(detail) > 4000:
                detail = detail[:397] + "..."
            if response.status_code == 4000 and "Model not found" in detail:
                detail = (
                    f"{detail}. The requested Open WebUI model id was '{self.model}'. "
                    "Pass the exact id with '--model remote:<model-id>' or set "
                    "BSM_AGENT_REMOTE_DEFAULT_MODEL to the Open WebUI model id you want "
                    "bare 'gpt-oss:20b' to target."
                )
            raise RuntimeError(
                f"Open WebUI request failed with HTTP {response.status_code}: {detail}"
            )

        payload = response.json()
        message = payload["choices"][0]["message"]
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            function_payload = tool_call.get("function", {})
            args = function_payload.get("arguments") or "{}"
            tool_calls.append(
                {
                    "name": function_payload["name"],
                    "args": json.loads(args),
                    "id": tool_call["id"],
                    "type": "tool_call",
                }
            )
        return {
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        }


def build_chat_model(
    target: ModelTarget,
    *,
    temperature: float = 0,
    tools: list[Any] | None = None,
) -> Any:
    if target.provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=target.model, temperature=temperature, num_ctx=8192)
        return llm.bind_tools(tools or [])

    if target.provider == "openwebui":
        return OpenWebUIChatModel(
            model=target.model,
            api_key=target.api_key,
            email=target.api_email,
            password=target.api_password,
            base_url=target.api_base or "",
            temperature=temperature,
            tools=tools,
        )

    if target.provider == "anthropic":
        return AnthropicChatModel(
            model=target.model,
            api_key=target.api_key or "",
            base_url=target.api_base,
            temperature=temperature,
            tools=tools,
        )

    return OpenAICompatibleChatModel(
        model=target.model,
        api_key=target.api_key or "",
        base_url=normalize_openai_base_url(target.api_base),
        temperature=temperature,
        tools=tools,
    )

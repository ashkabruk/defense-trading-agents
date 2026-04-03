from __future__ import annotations

import json
import os
from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.models.config import ModelConfig
from src.models.llm import ChatMessage, LLMResponse, ToolDefinition, ToolInvocation


class LLMClient:
    """Unified async LLM client with provider-swappable routing."""

    async def chat(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        tools: list[ToolDefinition] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        provider = model_config.provider.lower()
        if provider == "deepseek":
            return await self._chat_openai_compatible(messages, model_config, tools, response_format)
        if provider == "gemini":
            return await self._chat_gemini(messages, model_config, response_format)
        if provider == "anthropic":
            return await self._chat_anthropic(messages, model_config, response_format)

        raise ValueError(f"Unsupported provider: {model_config.provider}")

    async def _chat_openai_compatible(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        tools: list[ToolDefinition] | None,
        response_format: type[BaseModel] | None,
    ) -> LLMResponse:
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing required API key env var: {model_config.api_key_env}")

        client = AsyncOpenAI(api_key=api_key, base_url=model_config.base_url)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in tools
            ]

        kwargs: dict[str, Any] = {
            "model": model_config.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                },
            }

        if model_config.provider.lower() == "deepseek":
            kwargs.pop("response_format", None)

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolInvocation] = []
        if message.tool_calls:
            for call in message.tool_calls:
                parsed_args = json.loads(call.function.arguments) if call.function.arguments else {}
                tool_calls.append(
                    ToolInvocation(id=call.id, name=call.function.name, arguments=parsed_args)
                )

        return LLMResponse(
            model=model_config.model,
            provider=model_config.provider,
            content=message.content or "",
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            raw=response.model_dump(),
        )

    async def _chat_gemini(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        response_format: type[BaseModel] | None,
    ) -> LLMResponse:
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing required API key env var: {model_config.api_key_env}")

        client = genai.Client(api_key=api_key)
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        response = await client.aio.models.generate_content(
            model=model_config.model,
            contents=prompt,
            config={
                "temperature": model_config.temperature,
                "max_output_tokens": model_config.max_tokens,
                "response_mime_type": "application/json" if response_format else "text/plain",
            },
        )

        content = response.text or ""
        if response_format is not None:
            # Validate structured output early to fail fast for caller retries.
            response_format.model_validate_json(content)

        return LLMResponse(
            model=model_config.model,
            provider=model_config.provider,
            content=content,
            finish_reason=getattr(response, "finish_reason", None),
            raw=response.to_json_dict() if hasattr(response, "to_json_dict") else None,
        )

    async def _chat_anthropic(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        response_format: type[BaseModel] | None,
    ) -> LLMResponse:
        api_key = os.environ.get(model_config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing required API key env var: {model_config.api_key_env}")

        client = AsyncAnthropic(api_key=api_key)

        system_messages = [m.content for m in messages if m.role == "system"]
        conversation = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"user", "assistant"}
        ]

        response = await client.messages.create(
            model=model_config.model,
            system="\n".join(system_messages),
            messages=conversation,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )

        text_chunks = [block.text for block in response.content if block.type == "text"]
        content = "\n".join(text_chunks)

        if response_format is not None:
            response_format.model_validate_json(content)

        return LLMResponse(
            model=model_config.model,
            provider=model_config.provider,
            content=content,
            finish_reason=response.stop_reason,
            raw=response.model_dump(),
        )

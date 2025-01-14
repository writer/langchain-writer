import json
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to Writer dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """

    message_dict = {"role": "", "content": message.content}

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool["id"],
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": tool["args"]},
                }
                for tool in message.tool_calls
            ]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        raise ValueError(f"Got unknown message type: {type(message)}")

    if message.name:
        message_dict["name"] = message.name

    return message_dict


def convert_dict_to_message(response_message: Any) -> BaseMessage:
    """Convert a Writer message or dictionary to a LangChain message.

    Args:
        response_message: The response dictionary.

    Returns:
        The LangChain message.
    """

    if not isinstance(response_message, dict):
        response_message = json.loads(
            json.dumps(response_message, default=lambda o: o.__dict__)
        )

    role = response_message.get("role", "") or ""
    content = response_message.get("content", "") or ""

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs = {}

        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := response_message.get("tool_calls", []):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        additional_kwargs = {}
        if "name" in response_message:
            additional_kwargs["name"] = response_message["name"]
        return ToolMessage(
            content=content,
            tool_call_id=response_message.get("tool_call_id", ""),
            name=response_message.get("name", ""),
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=content, role=role)


def convert_dict_chunk_to_message_chunk(chunk: Any) -> BaseMessageChunk:
    if not isinstance(chunk, dict):
        chunk = json.loads(json.dumps(chunk, default=lambda o: o.__dict__))

    delta = chunk["choices"][0]["delta"]

    role = cast(str, delta.get("role"))
    content = cast(str, delta.get("content") or "")

    if role == "user":
        return HumanMessageChunk(content=content)
    elif role == "assistant":
        if usage := chunk.get("usage"):
            usage_metadata = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        else:
            usage_metadata = None

        additional_kwargs = {}
        tool_call_chunks = []
        if raw_tool_calls := delta.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for rtc in raw_tool_calls:
                try:
                    tool_call_chunks.append(
                        create_tool_call_chunk(
                            name=rtc["function"].get("name"),
                            args=rtc["function"].get("arguments"),
                            id=rtc.get("id"),
                            index=rtc.get("index"),
                        )
                    )
                except KeyError:
                    pass
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
        )
    elif role == "system":
        return SystemMessageChunk(content=content)
    elif role == "tool":
        return ToolMessageChunk(content=content, tool_call_id=delta["tool_call_id"])
    elif role:
        return ChatMessageChunk(content=content, role=role)

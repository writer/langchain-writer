import json
from typing import Any, Dict, List, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk


def _convert_message_to_dict(message: BaseMessage) -> dict:
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


def _convert_dict_to_message(response_message: Any) -> BaseMessage:
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
    content = response_message.get("content", "") or ""  # check on

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs = {}
        if tool_calls := response_message.get("tool_calls", []):
            additional_kwargs["tool_calls"] = tool_calls
        # TODO check on tool calls parsing
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
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


# TODO check on chunk creation
def _convert_dict_chunk_to_message_chunk(chunk: Any) -> BaseMessageChunk:
    if not isinstance(chunk, dict):
        chunk = json.loads(json.dumps(chunk, default=lambda o: o.__dict__))
    choice = chunk["choices"][0]
    _dict = choice["delta"]
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    tool_call_chunks: List[ToolCallChunk] = []
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if raw_tool_calls := _dict.get("tool_calls"):
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
    if role == "user":
        return HumanMessageChunk(content=content)
    elif role == "assistant":
        if usage := chunk.get("usage"):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
            }
        else:
            usage_metadata = None
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,  # type: ignore[arg-type]
        )
    elif role == "system":
        return SystemMessageChunk(content=content)
    elif role == "function":
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool":
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role:
        return ChatMessageChunk(content=content, role=role)

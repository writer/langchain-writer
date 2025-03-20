"""Writer chat models."""

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field
from writerai.types.chat_completion_chunk import Choice, ChoiceDelta

from langchain_writer.base import BaseWriter
from langchain_writer.tools import GraphTool, LLMTool, NoCodeAppTool


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to Writer dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """

    message_dict = {"role": "", "content": format_message_content(message.content)}

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
                    "function": {"name": tool["name"], "arguments": str(tool["args"])},
                }
                for tool in message.tool_calls
            ]

        if graph_data := message.additional_kwargs.get("graph_data"):
            message_dict["graph_data"] = graph_data

        if llm_calls := message.additional_kwargs.get("llm_data"):
            message_dict["llm_data"] = llm_calls

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


def convert_dict_to_message(response_message: dict) -> BaseMessage:
    """Convert a Writer dictionary to a LangChain message.

    Args:
        response_message: The response dictionary.

    Returns:
        The LangChain message.
    """

    if not isinstance(response_message, dict):
        raise ValueError(f"Expected 'dict' but got: {type(response_message)}")

    role = response_message.get("role", "") or ""
    content = response_message.get("content", "") or ""

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs = {}
        tool_calls = []
        invalid_tool_calls = []

        if raw_function_calls := response_message.get("tool_calls", []):
            additional_kwargs["tool_calls"] = raw_function_calls
            for raw_tool_call in raw_function_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )

        if raw_graph_calls := response_message.get("graph_data", {}):
            additional_kwargs["graph_data"] = raw_graph_calls

        if raw_llm_calls := response_message.get("llm_data", {}):
            additional_kwargs["llm_data"] = raw_llm_calls

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


def convert_dict_chunk_to_message_chunk(chunk: dict) -> BaseMessageChunk:
    """Convert a Writer chunk dictionary to a LangChain message.

    Args:
        chunk: The response dictionary.

    Returns:
        The LangChain message chunk.
    """

    if not isinstance(chunk, dict):
        raise ValueError(f"Expected 'dict' but got: {type(chunk)}")

    delta = chunk["choices"][0]["delta"]

    role = delta.get("role", "") or ""
    content = delta.get("content", "") or ""

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

        if raw_graph_calls := delta.get("graph_data", {}):
            additional_kwargs["graph_data"] = raw_graph_calls

        if raw_llm_calls := delta.get("llm_data", {}):
            additional_kwargs["llm_data"] = raw_llm_calls

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
    else:
        return ChatMessageChunk(content=content, role=role)


def create_chat_generation_chunk(
    chunk: Union[dict, BaseModel]
) -> tuple[ChatGeneration, dict]:
    """Convert a Writer message to a LangChain ChatGeneration chunk and logprobs.

    Args:
        chunk: The response dictionary.

    Returns:
        The LangChain ChatGeneration chunk and loprobs.
    """
    if not isinstance(chunk, dict):
        chunk = chunk.model_dump()

    message_chunk = convert_dict_chunk_to_message_chunk(chunk)

    generation_info = {}
    if finish_reason := chunk["choices"][0].get("finish_reason", ""):
        generation_info["finish_reason"] = finish_reason
    if logprobs := chunk["choices"][0].get("logprobs", {}):
        generation_info["logprobs"] = logprobs
    if usage := chunk.get("usage"):
        usage_metadata = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        generation_info["token_usage"] = usage_metadata

    generation_chunk = ChatGenerationChunk(
        message=message_chunk,
        generation_info=generation_info,
    )

    return generation_chunk, logprobs


def format_message_content(content: Any) -> str:
    """Format Lang Chain message content. Sanitize if from unnecessary elements.

    Args:
        content: Lang Chain message content.

    Returns:
        Formatted content in string format.
    """

    if content and isinstance(content, list):
        formatted_content = ""
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            elif (
                isinstance(block, dict) and "type" in block and block["type"] == "text"
            ):
                formatted_content += f" {block['text']}"
    else:
        formatted_content = content

    return formatted_content


def format_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable]
) -> dict[str, Any]:
    """Format tool and taking into consideration it's type e.g. 'function, 'graph', 'llm'.

    Args:
        tool: Tool to format.

    Returns:
        OpenAI typed dict with tool definition.
    """

    dict_tool = {}
    if isinstance(tool, BaseModel):
        dict_tool = tool.model_dump()
    elif isinstance(tool, BaseTool):
        dict_tool = tool.__dict__
    elif isinstance(tool, dict):
        dict_tool = tool

    if isinstance(tool, GraphTool):
        return {
            "type": "graph",
            "function": {
                "description": dict_tool.get("description", ""),
                "graph_ids": dict_tool.get("graph_ids", []),
                "subqueries": dict_tool.get("subqueries", False),
            },
        }
    elif isinstance(tool, LLMTool):
        return {
            "type": "llm",
            "function": {
                "description": dict_tool.get("description", ""),
                "model": dict_tool.get("model_name", ""),
            },
        }
    elif isinstance(tool, NoCodeAppTool):
        return tool.to_openai_tool()
    else:
        return convert_to_openai_tool(tool)


class ChatWriter(BaseWriter, BaseChatModel):
    """`Writer` chat model integration.

    Setup:
        Install ``langchain-writer`` and set environment variable ``WRITER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-writer

            export WRITER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_writer import ChatWriter

            llm = ChatWriter(
                model_name="...",
                temperature=0,
                max_tokens=None,
                request_timeout=None,
                max_retries=2,
                api_key="...",
                other params...

            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]

            llm.invoke(messages)

        .. code-block:: python

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk

        .. code-block:: python

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            stream:
            async for chunk in (await llm.astream(messages))

            batch:
            await llm.abatch([messages])

        .. code-block:: python

    Tool calling:
        .. code-block:: python

        from langchain_writer.tools import GraphTool
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''
            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

        graph_tool = GraphTool(graph_ids=['id1', 'id2'])

        Besides 'function' WriterChat supports 'graph' and 'llm' tool types.
        To use them pass GraphTool/LLMTool object from langchain_writer package
        as one of tools list elements of bind_tools() function params.

        llm_with_tools = llm.bind_tools([graph_tool, GetWeather])

        .. code-block:: python

        See ``ChatWriter.bind_tools()`` method for more.dict

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

    """

    """Model name to use."""
    model_name: str = Field(default="palmyra-x-004", alias="model")

    """What sampling temperature to use."""
    temperature: float = Field(default=0.7, ge=0, le=1)

    """Holds any model parameters valid for `create` call not explicitly specified."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    """Number of chat completions to generate for each prompt."""
    n: int = Field(default=1, ge=1)

    """Maximum number of tokens to generate."""
    max_tokens: Optional[int] = Field(default=None, ge=1)

    """Default stop sequences."""
    stop: Optional[Union[str, List[str]]] = Field(default=None, alias="stop_sequences")

    """Return logprobs or not"""
    logprobs: bool = Field(default=True)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "writer-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Writer API."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": self.n,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "logprobs": self.logprobs,
            **self.model_kwargs,
        }
        return params

    def _convert_messages_to_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Convert a list of LangChain messages to List of Writer dicts."""
        params = self._default_params
        if stop:
            params["stop"] = stop

        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()

        token_usage = response.get("usage", {})
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            if token_usage:
                message.usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)

        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.client.chat.chat(messages=message_dicts, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {**params, **kwargs}
        response = await self.async_client.chat.chat(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        response = self.client.chat.chat(messages=message_dicts, **params)

        for chunk in response:
            if len(chunk.choices) == 0 and chunk.usage is None:
                continue

            if len(chunk.choices) == 0 and chunk.usage:
                chunk.choices = [Choice(delta=ChoiceDelta(role="assistant"), index=0)]

            generation_chunk, logprobs = create_chat_generation_chunk(chunk)
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            yield generation_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._convert_messages_to_dicts(messages, stop)
        params = {
            **params,
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        response = await self.async_client.chat.chat(messages=message_dicts, **params)

        async for chunk in response:
            if len(chunk.choices) == 0 and chunk.usage is None:
                continue

            if len(chunk.choices) == 0 and chunk.usage:
                chunk.choices = [Choice(delta=ChoiceDelta(role="assistant"), index=0)]

            generation_chunk, logprobs = create_chat_generation_chunk(chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk, logprobs=logprobs
                )
            yield generation_chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        *,
        tool_choice: Optional[Union[str, Literal["auto", "none"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model.

        Args:
            tools: Tools to bind to the model
            tool_choice: Which tool to require ('auto', 'none', or specific tool name)

        Returns:
            Runnable Binding
        """
        formatted_tools = [format_tool(tool) for tool in tools]
        if tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            tool_choice = (
                tool_choice
                if tool_choice in ("auto", "none", "required")
                else {"type": "function", "function": {"name": tool_choice}}
            )
            kwargs["tool_choice"] = tool_choice

        return super().bind(tools=formatted_tools, **kwargs)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        system_fingerprint = None
        model_name = None
        for output in llm_outputs:
            if output is None:
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if v:
                        if k in overall_token_usage:
                            overall_token_usage[k] += v
                        else:
                            overall_token_usage[k] = v
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
            if model_name is None:
                model_name = output.get("model_name")
        combined = {"token_usage": overall_token_usage}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        if model_name:
            combined["model_name"] = model_name
        return combined

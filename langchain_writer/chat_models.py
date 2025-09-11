"""Writer chat models."""

import ast
import base64
from operator import itemgetter
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
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, Field
from writerai.types.chat_completion_chunk import Choice, ChoiceDelta

from langchain_writer.base import BaseWriter
from langchain_writer.tools import (
    GraphTool,
    LLMTool,
    NoCodeAppTool,
    TranslationTool,
    WebSearchTool,
)


def convert_message_to_dict(message: BaseMessage, model: str) -> dict:
    """Convert a LangChain message to Writer dictionary.

    Args:
        message: The LangChain message.
        model: Name of model that will handle converted messages

    Returns:
        The dictionary.
    """

    message_dict = {
        "role": "",
        "content": format_message_content(message.content, model),
    }

    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls:
            args_alias = "parameters" if model == "palmyra-x5" else "arguments"
            message_dict["tool_calls"] = [
                {
                    "id": tool["id"],
                    "type": "function",
                    "function": {"name": tool["name"], args_alias: str(tool["args"])},
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
                    tool_call = parse_tool_call(raw_tool_call, return_id=True)
                    tool_args = tool_call.get("args")
                    if isinstance(
                        tool_args, str
                    ):  # In case if LLM returned tool call args as TWO times dumped JSON
                        # trying to parse it for the second time. Using `literal_eval` because model may return
                        # a string with single quotes like "{'a': 3, 'b': 4}" and it's impossible to use json.loads
                        args = ast.literal_eval(tool_args)
                        tool_call["args"] = args
                    tool_calls.append(tool_call)
                except Exception as e:
                    invalid_tool_calls.append(
                        dict(make_invalid_tool_call(raw_tool_call, str(e)))
                    )

        if raw_graph_calls := response_message.get("graph_data", {}):
            additional_kwargs["graph_data"] = raw_graph_calls

        if raw_llm_calls := response_message.get("llm_data", {}):
            additional_kwargs["llm_data"] = raw_llm_calls

        if raw_translation_data := response_message.get("translation_data", {}):
            additional_kwargs["translation_data"] = raw_translation_data

        if raw_image_data := response_message.get("image_data", {}):
            additional_kwargs["image_data"] = raw_image_data

        if raw_web_search_data := response_message.get("web_search_data", {}):
            additional_kwargs["web_search_data"] = raw_web_search_data

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
        response_metadata = {}

        if usage := chunk.get("usage"):
            usage_metadata = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        else:
            usage_metadata = None

        response_metadata["token_usage"] = chunk.get("usage")

        if chunk["choices"][0].get("finish_reason"):
            response_metadata["model_name"] = chunk.get("model")
            response_metadata["system_fingerprint"] = chunk.get("system_fingerprint")
            response_metadata["finish_reason"] = chunk["choices"][0].get(
                "finish_reason"
            )
            response_metadata["logprobs"] = chunk["choices"][0].get("logprobs")

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

        if raw_translation_data := delta.get("translation_data", {}):
            additional_kwargs["translation_data"] = raw_translation_data

        if raw_image_data := delta.get("image_data", {}):
            additional_kwargs["image_data"] = raw_image_data

        if raw_web_search_data := delta.get("web_search_data", {}):
            additional_kwargs["web_search_data"] = raw_web_search_data

        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
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

    generation_chunk = ChatGenerationChunk(
        message=message_chunk,
        generation_info=(
            message_chunk.response_metadata if message_chunk.response_metadata else {}
        ),
    )

    return generation_chunk, (
        message_chunk.response_metadata.get("logprobs", {})
        if message_chunk.response_metadata
        else {}
    )


def format_message_content(
    content: Any, model: str
) -> Union[str, list[dict[str, Any]]]:
    """Format Lang Chain message content. Sanitize if from unnecessary elements.

    Args:
        content: Lang Chain message content.

    Returns:
        Formatted content in vision or non vision format.
    """

    if model == "palmyra-x5":
        return format_content_for_vision(content)
    else:
        return forman_content_for_non_vision(content)


def format_content_for_vision(content: Any) -> Union[str, list[dict[str, Any]]]:
    if content:
        if isinstance(content, list):
            formatted_content = []
            for block in content:
                if isinstance(block, dict) and "type" in block:
                    if block["type"] == "text" and "text" in block:
                        formatted_content += [block]
                    elif (
                        block["type"] == "image_url"
                        and "image_url" in block
                        and "url" in block["image_url"]
                    ):
                        formatted_content += [block]
                    elif block["type"] == "image" and "url" in block:
                        formatted_content += [
                            {"type": "image_url", "image_url": {"url": block["url"]}}
                        ]
                    elif (
                        block["type"] == "image"
                        and "base64" in block
                        and "mime_type" in block
                    ):
                        formatted_content += [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block['mime_type']};base64,{base64.b64encode(block['base64']).decode('utf-8')}",
                                },
                            }
                        ]

                    else:
                        raise TypeError(
                            f"Unsupported content type: {block['type']}, 'text'; 'image_url' or 'image' expected."
                            f" Or unsupported block format: {block}."
                        )
                elif isinstance(block, str):
                    formatted_content += [{"type": "text", "text": block}]
                else:
                    raise TypeError(
                        f"Unsupported content type: {type(content)}. Expected dict with 'type' key or str."
                    )
        elif isinstance(content, str):
            formatted_content = content
        else:
            raise TypeError(
                f"Unsupported content type: {type(content)}. Expected list or str."
            )

        return formatted_content
    else:
        return ""


def forman_content_for_non_vision(content: Any) -> str:
    if content:
        if isinstance(content, list):
            formatted_content = ""
            for block in content:
                if isinstance(block, dict) and "type" in block:
                    if block["type"] == "text":
                        formatted_content += f" {block['text']}"
                    else:
                        raise TypeError(
                            f"Unsupported content type: {block['type']}. Expected 'text' type."
                        )
                elif isinstance(block, str):
                    formatted_content += f" {block}"
                else:
                    raise TypeError(
                        f"Unsupported content type: {type(content)}. Expected dict with 'type' key or str."
                    )
        elif isinstance(content, str):
            formatted_content = content
        else:
            raise TypeError(
                f"Unsupported content type: {type(content)}. Expected list or str."
            )

        return formatted_content
    else:
        return ""


def format_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable]
) -> dict[str, Any]:
    """Format tool and taking into consideration it's type e.g. 'function, 'graph', 'llm'.

    Args:
        tool: Tool to format.

    Returns:
        Typed dict with tool definition.
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
            "type": dict_tool.get("type", ""),
            "function": {
                "name": dict_tool.get("name", ""),
                "description": dict_tool.get("description", ""),
                "graph_ids": dict_tool.get("graph_ids", []),
                "subqueries": dict_tool.get("subqueries", False),
            },
        }
    elif isinstance(tool, LLMTool):
        return {
            "type": dict_tool.get("type", ""),
            "function": {
                "name": dict_tool.get("name", ""),
                "description": dict_tool.get("description", ""),
                "model": dict_tool.get("model_name", ""),
            },
        }
    elif isinstance(tool, NoCodeAppTool):
        return tool.to_openai_tool()
    elif isinstance(tool, TranslationTool):
        return {
            "type": dict_tool.get("type", ""),
            "function": {
                "name": dict_tool.get("name", ""),
                "description": dict_tool.get("description", ""),
                "model": dict_tool.get("model", ""),
                "formality": dict_tool.get("formality", False),
                "length_control": dict_tool.get("length_control", False),
                "mask_profanity": dict_tool.get("mask_profanity", False),
                "source_language_code": dict_tool.get("source_language_code"),
                "target_language_code": dict_tool.get("target_language_code"),
            },
        }
    elif isinstance(tool, WebSearchTool):
        web_search_tool_dict = {
            "type": dict_tool.get("type", ""),
            "function": {
                # "name": dict_tool.get("name", ""),
                # "description": dict_tool.get("description", ""),
            },
        }

        if include_domains := dict_tool.get("include_domains"):
            web_search_tool_dict["function"].update(
                {"include_domains": include_domains}
            )
        if exclude_domains := dict_tool.get("exclude_domains"):
            web_search_tool_dict["function"].update(
                {"exclude_domains": exclude_domains}
            )

        return web_search_tool_dict
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
    model_name: str = Field(default="palmyra-x4", alias="model")

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

        message_dicts = [
            convert_message_to_dict(m, params.get("model")) for m in messages
        ]
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

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["json_schema", "function_calling"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - a function schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise, the model output will be a
                dict and will not be validated.

            method:
                The method for steering model generation, either "function_calling"
                or "json_mode/json_schema". If "function_calling" then the schema will be converted
                to a function dict and the returned model will make use of the
                function-calling API. If "json_schema" then JSON mode will be
                used. Note that if using "json_schema" then you must pass schema of
                desired output format.
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_writer import ChatWriter
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    # If we provide default values and/or descriptions for fields, these will be passed
                    # to the model. This is an important part of improving a model's ability to
                    # correctly return structured outputs.
                    justification: Optional[str] = Field(
                        default=None, description="A justification for the answer."
                    )


                llm = ChatWriter()
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound.
                                    The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_writer import ChatWriter
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatWriter()
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #    'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-zsHrm9K2nfygJMQylrBLhb47lPZskFvW', 'function': {'arguments': '{"answer": "They weigh the same.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The common misconception arises from the difference in volume between bricks and feathers, but the weight specified is the same for both."}', 'name': 'AnswerWithJustification'}, 'type': 'function', 'index': None}], 'graph_data': {'sources': None, 'status': None, 'subqueries': None}}, response_metadata={'token_usage': {'completion_tokens': 70, 'prompt_tokens': 202, 'total_tokens': 272, 'completion_tokens_details': None, 'prompt_token_details': None}, 'model_name': 'palmyra-x5', 'system_fingerprint': 'v1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-44ae92bf-2eed-47c6-bdcd-81165f826b48-0', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same.', 'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The common misconception arises from the difference in volume between bricks and feathers, but the weight specified is the same for both.'}, 'id': 'chatcmpl-tool-zsHrm9K2nfygJMQylrBLhb47lPZskFvW', 'type': 'tool_call'}], usage_metadata={'input_tokens': 202, 'output_tokens': 70, 'total_tokens': 272}),
                #    'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The common misconception arises from the difference in volume between bricks and feathers, but the weight specified is the same for both.'),
                #    'parsing_error': None
                #   }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Annotated, TypedDict

                from langchain_writer import ChatWriter


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatWriter()
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=Function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_writer import ChatWriter

                schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatWriter()
                structured_llm = llm.with_structured_output(schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=Pydantic class, method="json_schema", include_raw=True:
            .. code-block::

                from langchain_writer import ChatWriter
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatWriter()
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_schema",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #    'raw': AIMessage(content='{"answer": "They weigh the same.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The common misconception arises from the difference in volume between the two items; bricks are much denser than feathers, so a pound of feathers will occupy a much larger volume than a pound of bricks."}', additional_kwargs={'graph_data': {'sources': None, 'status': None, 'subqueries': None}}, response_metadata={'token_usage': {'completion_tokens': 69, 'prompt_tokens': 66, 'total_tokens': 135, 'completion_tokens_details': None, 'prompt_token_details': None}, 'model_name': 'palmyra-x5', 'system_fingerprint': 'v1', 'finish_reason': 'stop', 'logprobs': None}, id='run-8f9190db-5bc5-43f5-9575-2dc4919594bd-0', usage_metadata={'input_tokens': 66, 'output_tokens': 69, 'total_tokens': 135}),
                #    'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The common misconception arises from the difference in volume between the two items; bricks are much denser than feathers, so a pound of feathers will occupy a much larger volume than a pound of bricks.'),
                #    'parsing_error': None
                # }

        Example: schema=Function schema, method="json_schema", include_raw=False:
            .. code-block::

                from langchain_writer import ChatWriter

                schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
                }

                structured_llm = llm.with_structured_output(schema=schema, method="json_schema")

                llm = ChatWriter()
                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #    'answer': 'Neither, they both weigh the same.',
                #    'justification': 'A pound is a unit of weight, and both a pound of bricks and a pound of feathers weigh exactly one pound. The common misconception arises from the difference in volume between bricks and feathers, where a pound of feathers will occupy much more space than a pound of bricks due to their lightness.'
                # }
        """
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")

        is_pydantic_schema = _is_pydantic_class(schema)

        if method in ["json_schema", "json_mode"]:
            if schema is None:
                raise ValueError(
                    "Schema must be specified when method is 'json_schema/json_mode'. "
                    "Received None."
                )

            response_format = _convert_to_openai_response_format(schema, strict=True)

            llm = self.bind(
                response_format=response_format,
                structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )

        elif method == "function_calling":
            if schema is None:
                raise ValueError(
                    "Schema must be specified when method is 'function_calling'. "
                    "Received None."
                )

            formatted_tool = _convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]

            llm = self.bind_tools(
                [formatted_tool],
                tool_choice=tool_name,
                structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )

            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling', 'json_schema' or "
                f"'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

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


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _convert_to_openai_tool(
    tool: Union[dict[str, Any], type[BaseModel], Callable, BaseTool],
    *,
    strict: Optional[bool] = None,
) -> dict[str, Any]:
    if (
        isinstance(tool, dict)
        and tool.get("json_schema")
        and tool["json_schema"].get("schema", {}).get("properties")
    ):
        tool = {
            "parameters": tool["json_schema"]["schema"],
            "name": tool["json_schema"].get("name"),
        }
    return convert_to_openai_tool(tool, strict=strict)


def _convert_to_openai_response_format(
    schema: Union[Dict[str, Any], Type], *, strict: Optional[bool] = None
) -> Dict:
    if (
        isinstance(schema, dict)
        and "json_schema" in schema
        and schema.get("type") == "json_schema"
    ):
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_tool(schema, strict=strict)["function"]
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    if (
        strict is not None
        and "strict" in response_format["json_schema"]
        and strict is not response_format["json_schema"].get("strict")
    ):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format

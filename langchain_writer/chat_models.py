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
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self
from writerai import AsyncWriter, Writer

from langchain_writer.utils import (
    convert_dict_chunk_to_message_chunk,
    convert_dict_to_message,
    convert_message_to_dict,
)


class ChatWriter(BaseChatModel):
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

                model="...",

                temperature=0,

                max_tokens=None,

                timeout=None,

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

            # TODO: Example output.

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):

                print(chunk)

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            stream = llm.stream(messages)

            full = next(stream)

            for chunk in stream:

                full += chunk

            full

        .. code-block:: python

            # TODO: Example output.

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)


            stream:

            async for chunk in (await llm.astream(messages))


            batch:

            await llm.abatch([messages])

        .. code-block:: python

            # TODO: Example output.

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):

                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            class GetPopulation(BaseModel):

                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])

            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")

            ai_msg.tool_calls

        .. code-block:: python

              # TODO: Example output.

        See ``ChatWriter.bind_tools()`` method for more.

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):

                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")

                punchline: str = Field(description="The punchline to the joke")

                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_llm = llm.with_structured_output(Joke)

            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            # TODO: Example output.

        See ``ChatWriter.with_structured_output()`` for more.

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)

            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    Logprobs:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.

            logprobs_llm = llm.bind(logprobs=True)

            ai_msg = logprobs_llm.invoke(messages)

            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

              # TODO: Example output.

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)

            ai_msg.response_metadata

        .. code-block:: python

             # TODO: Example output.

    """  # noqa: E501

    client: Writer = Field(default=None, exclude=True)  #: :meta private:
    async_client: AsyncWriter = Field(default=None, exclude=True)  #: :meta private:

    """Writer API key.
    Automatically read from env variable `WRITER_API_KEY` if not provided.
    """
    api_key: SecretStr = Field(
        default_factory=secret_from_env(
            "WRITER_API_KEY",
            error_message=(
                "You must specify an api key. "
                "You can pass it an argument as `api_key=...` or "
                "set the environment variable `WRITER_API_KEY`."
            ),
        ),
    )

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

    """Timeout for requests to Writer completion API. Can be float, httpx.Timeout or
        None."""
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )

    """Base URL path for API requests."""
    api_base: Optional[str] = Field(
        alias="base_url", default_factory=from_env("WRITER_BASE_URL", default=None)
    )

    """Maximum number of retries to make when generating."""
    max_retries: int = Field(default=2, gt=0)

    """Return logprobs or not"""
    logprobs: bool = Field(default=True)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists in environment."""

        client_params = {
            "api_key": (self.api_key.get_secret_value() if self.api_key else None),
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "base_url": self.api_base,
        }

        if not self.client:
            self.client = Writer(**client_params)
        if not self.async_client:
            self.async_client = AsyncWriter(**client_params)
        return self

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
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "n": self.n,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "logprobs": self.logprobs,
            **self.model_kwargs,
        }

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
        params = {**params, **kwargs, "stream": True}

        response = self.client.chat.chat(messages=message_dicts, **params)

        for chunk in response:
            if len(chunk.choices) == 0:
                continue
            message_chunk = convert_dict_chunk_to_message_chunk(chunk)
            generation_info = {}
            if finish_reason := chunk.choices[0].finish_reason:
                generation_info["finish_reason"] = finish_reason
            if logprobs := chunk.choices[0].logprobs:
                generation_info["logprobs"] = logprobs
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info,
            )

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
        params = {**params, **kwargs, "stream": True}

        response = await self.async_client.chat.chat(messages=message_dicts, **params)

        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            message_chunk = convert_dict_chunk_to_message_chunk(chunk)
            generation_info = {}
            if finish_reason := chunk.choices[0].finish_reason:
                generation_info["finish_reason"] = finish_reason
            if logprobs := chunk.choices[0].logprobs:
                generation_info["logprobs"] = logprobs
            generation_chunk = ChatGenerationChunk(
                message=message_chunk,
                generation_info=generation_info,
            )

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
            **kwargs: Additional parameters to pass to the chat model

        Returns:
            A runnable that will use the tools
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            kwargs["tool_choice"] = (
                (tool_choice)
                if tool_choice in ("auto", "none")
                else {"type": "function", "function": {"name": tool_choice}}
            )

        return super().bind(tools=formatted_tools, **kwargs)

    # def with_structured_output(
    #     self,
    #     schema: Optional[Union[Dict, Type[BaseModel]]] = None,
    #     *,
    #     method: Literal["function_calling", "json_mode"] = "function_calling",
    #     include_raw: bool = False,
    #     **kwargs: Any,
    # ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]: ...

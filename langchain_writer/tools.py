"""Writer tools."""

from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr
from writerai import AsyncWriter, Writer
from writerai.types.application_retrieve_response import Input as ResponseInput
from writerai.types.applications.job_create_params import Input


class GraphTool(BaseTool):
    """`Writer` GraphTool. It's main feature is 'graph' type and running workflow
    (it runs only in the WriterChat environment and doesn't support direct calls).
    For more information visit tool calling section of `langchain_writer.ChatWriter()`
    docstring or `ChatWriter` docs.

    Setup:
        Install ``langchain-writer``.

        .. code-block:: bash

            pip install -U langchain-writer

    Instantiate:
        To instantiate tool you should pass required parameter `graph_ids`.
        Other parameters are optional.

        .. code-block:: python

            from langchain_writer.tools import GraphTool


            graph_tool = GraphTool(

                graph_ids=["your-graph-id1", "your-graph-id2"],

                name="CV knowledge graph",

                other params...

            )

    """

    """Tool type."""
    type: Literal["graph"] = "graph"

    """Tool name."""
    name: str = "Knowledge Graph"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "Graph of files from which model can fetch data to compose response on user request."
    )

    """list of grap ids to handle request"""
    graph_ids: list[str]

    """Whether include the subqueries used by Palmyra in the response"""
    subqueries: bool = Field(default=False)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Writer GraphTool does not support direct invocations."
        )


class LLMTool(BaseTool):
    """`Writer` LLMTool. It's main feature is 'llm' type and running workflow
    (it runs only in the WriterChat environment and doesn't support direct calls).
    For more information visit tool calling section of `langchain_writer.ChatWriter()`
    docstring or `ChatWriter` docs.
    """

    """Tool type."""
    type: Literal["llm"] = "llm"

    """Tool name."""
    name: str = "Large Language Model"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "LLM tool provide way to perform sub call of another type of model to compose response on user request."
    )

    """Name of LLM to invoke."""
    model_name: Literal[
        "palmyra-x5",
        "palmyra-x4",
        "palmyra-x-003-instruct",
        "palmyra-med",
        "palmyra-fin",
        "palmyra-creative",
    ] = "palmyra-x5"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        descriptions = {
            "palmyra-x5": "Writer's newest and most advanced model for building and scaling AI agents, featuring "
            "a 1 million token context window, adaptive reasoning, and industry-leading speed and cost efficiency.",
            "palmyra-x4": "Enterprise-grade language model for answering "
            "global questions and compose suggestions linked with various topics.",
            "palmyra-x-003-instruct": "Top-performing instruct language model, built "
            "specifically for structured text completion rather than conversational use.",
            "palmyra-med": "Language model, engineered to support clinical and "
            "administrative workflows with high accuracy in medical terminology, coding, and analysis",
            "palmyra-fin": "Language model, designed to support critical financial workflows "
            "with precision in terminology, document analysis, and complex investment analysis.",
            "palmyra-creative": "Language model, engineered to elevate creative "
            "thinking and writing across diverse professional contexts",
        }

        description = kwargs.get("description")
        if description:
            self.description = description
        else:
            self.description = descriptions[self.model_name]

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Writer LLMTool does not support direct invocations.")


class NoCodeAppTool(BaseTool):
    """`Writer` NoCodeAppTool allows developers to use Palmyra powered zero-code apps as LLM tools.
    For more information visit tool calling section of `langchain_writer.ChatWriter()`
    docstring or `ChatWriter` docs.

    Setup:
        Install ``langchain-writer`` and set environment variable ``WRITER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-writer

            export WRITER_API_KEY="your-api-key"

    Instantiate:
        To instantiate tool you should pass required parameter `app_id` and `api_key`
        if you don't have it as environment variable. Other parameters are optional.
        Attribute `app_inputs` will be auto initialized on instantiation.

        .. code-block:: python

            from langchain_writer.tools import NoCodeAppTool


            no_code_app_tool = NoCodeAppTool(

                api_key="your-api-key",

                app_id="your-app-id",

                name="Poem generator app",

                other params...

            )

    Invoke:
        To run no-code app tool you should pass inputs dict which match app inputs as element
        of `tool_input` dict with `key=inputs` as following:

        .. code-block:: python

            no_code_app_tool.run(tool_input={"inputs": {"Input name": "Input value"}})

        .. code-block:: python
    """

    """Tool type."""
    type: Literal["function"] = "function"

    """Tool name that is passed to model when performing tool calling."""
    name: str = "No-code application"

    """The description that is passed to the model when performing tool calling."""
    description: str = "No-code application powered by Palmyra"

    """App ID"""
    app_id: str = ""

    """App input args"""
    app_inputs: list[ResponseInput] = Field(default_factory=list)

    """Writer API key"""
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

    client: Writer = Field(default=None, exclude=True)  #: :meta private:
    async_client: AsyncWriter = Field(default=None, exclude=True)  #: :meta private:

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if not kwargs.get("client"):
            self.client = Writer(api_key=self.api_key.get_secret_value())
        if not kwargs.get("async_client"):
            self.async_client = AsyncWriter(api_key=self.api_key.get_secret_value())

        app_metadata = self.client.applications.retrieve(self.app_id)
        self.app_inputs = app_metadata.inputs

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.applications.generate_content(
            application_id=self.app_id, inputs=self.convert_inputs(kwargs.get("inputs"))
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self.async_client.applications.generate_content(
            application_id=self.app_id, inputs=self.convert_inputs(kwargs.get("inputs"))
        )

    def convert_inputs(self, inputs: dict) -> list[Input]:
        """Convert inputs from dict into Writer objects.

        Args:
            inputs: No-code application inputs stored in dict required to run no-code app tool:

             inputs = {

             "Input name": "Input value",

             "Another input name": ["List", "of", "string", "values"],

             }

             Input values should be strings or lists of strings.

        Returns:
            List of Writer input objects.
        """
        if not inputs or not isinstance(inputs, dict):
            raise ValueError(
                "To run no-code app tool you must pass non-empty inputs dict."
            )

        converted_inputs = []
        for app_input in self.app_inputs:
            input_from_dict = inputs.get(app_input.name)

            if input_from_dict:
                if isinstance(input_from_dict, str):
                    converted_inputs.append(
                        Input(id=app_input.name, value=[input_from_dict])
                    )
                elif isinstance(input_from_dict, list) and all(
                    [isinstance(el, str) for el in input_from_dict]
                ):
                    converted_inputs.append(
                        Input(id=app_input.name, value=input_from_dict)
                    )
                else:
                    raise ValueError(
                        f"Input should be a string or list of strings. But {app_input.name} is {type(input_from_dict)}"
                    )

            elif app_input.required:
                raise ValueError(
                    f"Argument `{app_input.name}` is required to run this no-code app tool."
                )

        return converted_inputs

    def to_openai_tool(self) -> dict[str, Any]:
        properties = {}
        required = []

        for app_input in self.app_inputs:
            properties.update(
                {
                    app_input.name: {
                        "type": "string",
                        "description": app_input.description,
                    }
                }
            )

            if app_input.required:
                required.append(app_input.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class TranslationTool(BaseTool):
    """`Writer` TranslationTool. The translation tool for chat completions allows you
    to translate text during a conversation with a Palmyra model. While Palmyra X models
    can perform translation tasks, they are not optimized for these tasks and may not
    perform well without correct prompting. Palmyra Translate is a dedicated model optimized
    for translation use cases. (it runs only in the WriterChat environment and doesn't
    support direct calls).
    """

    """Tool type."""
    type: Literal["translation"] = "translation"

    """Tool name that is passed to model when performing tool calling."""
    name: str = "Translation tool"

    """The description that is passed to the model when performing tool calling."""
    description: str = "Texts translator powered by palmyra-translation"

    """Name of model used as translator"""
    model: Literal["palmyra-translate"] = "palmyra-translate"

    """Whether the translation should be formal or informal, if the target language supports it."""
    formality: bool = False

    """Whether to control the length of the translation, if the target language supports it."""
    length_control: bool = False

    """Whether to mask profanity in the translation, if the target language supports it."""
    mask_profanity: bool = False

    """The language code of the text you want to translate. If you don’t provide a source language,
    the model automatically detects the language of the text you want to translate."""
    source_language_code: Optional[str] = None

    """The language code you want to translate the text to. If you don’t provide a target language,
    the model automatically selects the most appropriate language based on the message you provide
    to the chat completion endpoint."""
    target_language_code: Optional[str] = None

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Writer TranslationTool does not support direct invocations."
        )


class WebSearchTool(BaseTool):
    """`Writer` WebSearchTool. The web search tool for chat completions allows you
    to receive most relevant information from web. It surfs the web for information
    about a given query and return relevant results with source URLs.
    """

    """Tool type."""
    type: Literal["web_search"] = "web_search"

    """Tool name that is passed to model when performing tool calling."""
    name: str = "Web Search tool"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "Web Search engine that allows you to surf the web for information"
    )

    """An array of domains to include in the search results."""
    include_domains: list[str] = Field(default_factory=list)

    """An array of domains to exclude from the search results."""
    exclude_domains: list[str] = Field(default_factory=list)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Writer WebSearchTool does not support direct invocations."
        )

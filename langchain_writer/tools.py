"""Writer tools."""

from typing import Any, Literal

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
        "palmyra-x-004",
        "palmyra-x-003-instruct",
        "palmyra-med",
        "palmyra-fin-32k",
        "palmyra-creative",
        "palmyra-vision",
    ] = "palmyra-x-004"

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Writer LLMTool does not support direct invocations.")


class NoCodeAppTool(BaseTool):
    """`Writer` NoCodeAppTool allows developers to use Palmyra powered zero-code apps as LLM tools.
    For more information visit tool calling section of `langchain_writer.ChatWriter()`
    docstring or `ChatWriter` docs.
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

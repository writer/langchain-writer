"""Writer tools."""

from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self
from writerai import AsyncWriter, Writer


class GraphToolInput(BaseModel):
    """Input schema for Writer Knowledge Graph tool."""

    question: str = Field(..., description="Question sent to graph.")


class GraphTool(BaseTool):  # type: ignore[override]
    """Writer Knowledge Graph tool.

    Setup:
        Install ``langchain-writer`` and set environment variable ``WRITER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-writer

            export WRITER_API_KEY="your-api-key"

    Instantiation:

        .. code-block:: python

            tool = WriterTool(

                graph_ids=["id1", "id2"],

                subqueries=True

            )

    Invocation with args:

        .. code-block:: python

            tool.invoke(question="")

        .. code-block:: python

            Question(

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke(
                {
                    "args": {"question": "How to stay healthy?"},
                    "id": "1",
                    "name": tool.name,
                    "type": "tool_call"
                }
            )

        .. code-block:: python
    """

    """The name that is passed to the model when performing tool calling."""
    name: str = "Knowledge graph"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "Graph of files from which model can fetch data to compose response on user request."
    )

    """The schema that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = GraphToolInput

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

    """list of grap ids to handle request"""
    graph_ids: list[str]

    """Whether include the subqueries used by Palmyra in the response"""
    subqueries: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists in environment."""

        api_key = self.api_key.get_secret_value() if self.api_key else None

        if not self.client:
            self.client = Writer(api_key=api_key)
        if not self.async_client:
            self.async_client = AsyncWriter(api_key=api_key)
        return self

    def _run(
        self, question: str, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        response = self.client.graphs.question(
            graph_ids=self.graph_ids,
            question=question,
            stream=False,
            subqueries=self.subqueries,
        )
        dict_response = response.model_dump()
        return dict_response

    async def _arun(
        self,
        question: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        response = await self.async_client.graphs.question(
            graph_ids=self.graph_ids,
            question=question,
            stream=False,
            subqueries=self.subqueries,
        )
        dict_response = response.model_dump()
        return dict_response

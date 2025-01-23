"""Writer tools."""

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import Field


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

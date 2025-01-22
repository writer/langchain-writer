"""Writer tools."""

from typing import Literal

from pydantic import BaseModel, Field


class GraphTool(BaseModel):
    """Tool type."""

    type: Literal["graph"] = "graph"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "Graph of files from which model can fetch data to compose response on user request."
    )

    """list of grap ids to handle request"""
    graph_ids: list[str]

    """Whether include the subqueries used by Palmyra in the response"""
    subqueries: bool = Field(default=False)

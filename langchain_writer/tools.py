"""Writer tools."""

from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_writer.base import BaseWriter


class GraphToolInput(BaseModel):
    """Input schema for Writer Knowledge Graph tool."""

    question: str = Field(..., description="Question sent to graph.")


class GraphTool(BaseWriter, BaseTool):
    """
    Writer Knowledge Graph tool.

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

    Returned dict sample:

        .. code-block:: json

            {
                "answer": "Knowledge Graph text response",
                "question": "Your sent question",
                "sources": [
                    {
                        "file_id": "ID of file used by KG to compose an answer",
                        "snippet": "File data snippet used by KG to compose an answer"
                    },
                    {
                        "file_id": "Another ID of file used by KG to compose an answer",
                        "snippet": "Another file data snippet used by KG to compose an answer"
                    }
                ],
                "subqueries": [
                    {
                        "query": "Subquery question used by KG to compose a final answer",
                        "answer": "Subquery answer used by KG to compose a final answer",
                        "sources": [
                            {
                                "file_id": "ID of file used by KG in subquery to compose an answer",
                                "snippet": "File data snippet used by KG in subquery to compose an answer"
                            },
                            {
                                "file_id": "Another ID of file used by KG in subquery to compose an answer",
                                "snippet": "Another file data snippet used by KG in subquery to compose an answer"
                            }
                        ]
                    }
                ]
            }
    """

    """The name that is passed to the model when performing tool calling."""
    name: str = "Knowledge graph"

    """The description that is passed to the model when performing tool calling."""
    description: str = (
        "Graph of files from which model can fetch data to compose response on user request."
    )

    """The schema that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = GraphToolInput

    """list of grap ids to handle request"""
    graph_ids: list[str]

    """Whether include the subqueries used by Palmyra in the response"""
    subqueries: bool = Field(default=False)

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

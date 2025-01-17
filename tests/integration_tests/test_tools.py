"""Integration tests Writer Graph Tool wrapper

You need WRITER_API_KEY set in your environment
and fill GRAPH_IDS variable to run these tests.
"""

import json

import pytest
from langchain_core.messages import ToolMessage

from langchain_writer.tools import GraphTool

GRAPH_IDS = []


def test_graph_tool_invoke_directly(graph_tool: GraphTool):
    response = graph_tool.invoke({"question": "How to improve my physical conditions?"})

    assert isinstance(response, dict)
    assert isinstance(response["answer"], str)
    assert len(response["answer"]) > 0


@pytest.mark.asyncio
async def test_graph_tool_async_invoke_directly(graph_tool: GraphTool):
    response = await graph_tool.ainvoke(
        {"question": "How to improve my physical conditions?"}
    )

    assert isinstance(response, dict)
    assert isinstance(response["answer"], str)
    assert len(response["answer"]) > 0


def test_graph_tool_invoke_tool_call(graph_tool: GraphTool):
    model_generated_tool_call = {
        "args": {
            "question": "How to improve my physical conditions?",
        },
        "id": "id",
        "name": graph_tool.name,
        "type": "tool_call",
    }
    response = graph_tool.invoke(model_generated_tool_call)

    assert isinstance(response, ToolMessage)
    content = json.loads(response.content)
    assert isinstance(content["answer"], str)
    assert len(content["answer"]) > 0


@pytest.mark.asyncio
async def test_graph_tool_async_invoke_tool_call(graph_tool: GraphTool):
    model_generated_tool_call = {
        "args": {
            "question": "How to improve my physical conditions?",
        },
        "id": "id",
        "name": graph_tool.name,
        "type": "tool_call",
    }
    response = await graph_tool.ainvoke(model_generated_tool_call)

    assert isinstance(response, ToolMessage)
    content = json.loads(response.content)
    assert isinstance(content["answer"], str)
    assert len(content["answer"]) > 0

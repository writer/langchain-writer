"""Unit tests Writer Graph Tool wrapper

You need WRITER_API_KEY set in your environment to run these tests.
"""

import os

import pytest
from pydantic import SecretStr

from langchain_writer.tools import GraphTool
from tests.unit_tests.test_chat_models import keep_api_key


@keep_api_key
def test_graph_tool_api_key_in_env():
    os.environ["WRITER_API_KEY"] = "API key"

    tool = GraphTool(graph_ids=["id1", "id2"])

    assert tool.api_key.get_secret_value() == "API key"


@keep_api_key
def test_graph_tool_api_key_not_in_env_error():
    os.environ.pop("WRITER_API_KEY", None)

    with pytest.raises(ValueError):
        GraphTool(graph_ids=["id1", "id2"])


@keep_api_key
def test_graph_tool_api_key_not_in_env_success():
    os.environ.pop("WRITER_API_KEY", None)

    tool = GraphTool(api_key=SecretStr("API key"), graph_ids=["id1", "id2"])

    assert tool.api_key.get_secret_value() == "API key"

import pytest

from langchain_writer import ChatWriter, GraphTool
from tests.integration_tests.test_tools import GRAPH_IDS


@pytest.fixture(scope="function")
def chat_writer():
    return ChatWriter()


@pytest.fixture(scope="function")
def graph_tool():
    return GraphTool(graph_ids=GRAPH_IDS)

import pytest
from langchain_core.documents.base import Blob

from langchain_writer import ChatWriter, GraphTool, WriterTextSplitter
from langchain_writer.pdf_parser import PDFParser
from langchain_writer.tools import LLMTool

GRAPH_IDS = ["id_1", "id_2"]


@pytest.fixture(scope="function")
def chat_writer():
    return ChatWriter()


@pytest.fixture(scope="function")
def graph_tool():
    return GraphTool(graph_ids=GRAPH_IDS)


@pytest.fixture(scope="function")
def llm_tool():
    return LLMTool()


@pytest.fixture(scope="function")
def text_splitter():
    return WriterTextSplitter()


@pytest.fixture(scope="function")
def pdf_parser():
    return PDFParser()


@pytest.fixture(scope="function")
def pdf_file():
    return Blob.from_path("tests/data/sample.pdf")


@pytest.fixture(scope="function")
def text_file():
    return Blob.from_path("tests/data/text_to_split.txt")

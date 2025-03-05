import os

import pytest
from dotenv import load_dotenv
from langchain_core.documents.base import Blob

from langchain_writer import ChatWriter, GraphTool, NoCodeAppTool, WriterTextSplitter
from langchain_writer.pdf_parser import PDFParser
from langchain_writer.tools import LLMTool

load_dotenv()

GRAPH_IDS = os.environ.get("GRAPH_IDS").split(" ")
TEXT_GENERATION_APP_ID = os.getenv("TEXT_GENERATION_APP_ID")
RESEARCH_APP_ID = os.getenv("RESEARCH_APP_ID")


@pytest.fixture(scope="function")
def chat_writer():
    return ChatWriter()


@pytest.fixture(scope="function")
def graph_tool():
    return GraphTool(graph_ids=GRAPH_IDS)


@pytest.fixture(scope="function")
def llm_tool():
    return LLMTool()


@pytest.fixture(scope="function", params=[TEXT_GENERATION_APP_ID, RESEARCH_APP_ID])
def no_code_app_tool(request):
    return NoCodeAppTool(app_id=request.param)


def get_app_inputs(app: NoCodeAppTool):
    inputs = {}
    for app_input in app.app_inputs:
        inputs.update({app_input.name: "fake input"})
    return inputs


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

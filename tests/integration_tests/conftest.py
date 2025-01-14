import pytest

from langchain_writer import ChatWriter


@pytest.fixture(scope="function")
def chat_writer():
    return ChatWriter()

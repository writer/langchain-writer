import pytest
from dotenv import load_dotenv

from langchain_writer import ChatWriter

load_dotenv()


@pytest.fixture(scope="function")
def chat_writer_non_vision_model():
    return ChatWriter(model="palmyra-x4")


@pytest.fixture(scope="function")
def chat_writer_vision_model():
    return ChatWriter(model="palmyra-x5")

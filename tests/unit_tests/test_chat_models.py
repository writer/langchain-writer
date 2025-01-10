"""Test chat model integration."""

from typing import Type

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_writer.chat_models import ChatWriter


class TestChatWriterUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatWriter]:
        return ChatWriter

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }

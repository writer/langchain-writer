from typing import Dict, Optional, Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_writer import ChatWriter


class TestWriterChatStandard(ChatModelIntegrationTests):
    """Test case for ChatWriter that inherits from standard LangChain tests."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return ChatWriter model class."""
        return ChatWriter

    @property
    def chat_model_params(self) -> Dict:
        """Return any additional parameters needed."""
        return {
            "model_name": "palmyra-x5",
            "model_kwargs": {"timeout": 60},
        }

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice in tests."""
        return "auto"

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return True

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return {"WRITER_API_KEY": "key"}, {"api_key": "key"}, {"api_key": "key"}

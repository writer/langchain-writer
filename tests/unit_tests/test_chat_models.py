"""Unit tests ChatWriter wrapper

You need WRITER_API_KEY set in your environment to run these tests.
"""

import os

import pytest
from pydantic import SecretStr

from langchain_writer import ChatWriter


def keep_api_key(func):
    def keep_api_key_wrapper(*args, **kwargs):
        current_key = os.getenv("WRITER_API_KEY")
        func(*args, **kwargs)
        if current_key:
            os.environ["WRITER_API_KEY"] = current_key

    return keep_api_key_wrapper


@keep_api_key
def test_chat_writer_api_key_in_env():
    os.environ["WRITER_API_KEY"] = "API key"

    chat = ChatWriter()

    assert chat.api_key.get_secret_value() == "API key"


@keep_api_key
def test_chat_writer_api_key_not_in_env_error():
    os.environ.pop("WRITER_API_KEY", None)

    with pytest.raises(ValueError):
        ChatWriter()


@keep_api_key
def test_chat_writer_api_key_not_in_env_success():
    os.environ.pop("WRITER_API_KEY", None)

    chat = ChatWriter(api_key=SecretStr("API key"))

    assert chat.api_key.get_secret_value() == "API key"


def test_chat_writer_params_validation():
    params = {"temperature": 1.5, "max_tokens": -1}

    with pytest.raises(ValueError):
        ChatWriter(**params)

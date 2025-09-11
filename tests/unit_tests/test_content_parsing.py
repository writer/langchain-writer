import base64

import pytest
from langchain_core.messages import HumanMessage

from langchain_writer import ChatWriter


def test_non_vision_list_with_string_dict(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "How to sleep well?"},
        ]
    )

    formatted_messages, params = (
        chat_writer_non_vision_model._convert_messages_to_dicts([message])
    )

    assert "model" in params
    assert params["model"] == chat_writer_non_vision_model.model_name
    assert formatted_messages[0]["content"].strip() == "How to sleep well?"


def test_non_vision_list_with_string_dicts(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "How to sleep well?"},
            {"type": "text", "text": "How to get better marks at school?"},
        ]
    )

    formatted_messages, params = (
        chat_writer_non_vision_model._convert_messages_to_dicts([message])
    )

    assert "model" in params
    assert params["model"] == chat_writer_non_vision_model.model_name
    assert (
        formatted_messages[0]["content"].strip()
        == "How to sleep well? How to get better marks at school?"
    )


def test_non_vision_list_with_string_and_text_dict(
    chat_writer_non_vision_model: ChatWriter,
):
    message = HumanMessage(
        content=["Simple string", {"type": "text", "text": "and dict text"}]
    )
    formatted_messages, _ = chat_writer_non_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"].strip() == "Simple string and dict text"


def test_non_vision_single_string(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(content="Just a plain string")
    formatted_messages, _ = chat_writer_non_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == "Just a plain string"


def test_non_vision_list_with_single_string(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(content=["Just a plain string"])
    formatted_messages, _ = chat_writer_non_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"].strip() == "Just a plain string"


def test_non_vision_list_strings(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(content=["Just a plain string", "One more string"])
    formatted_messages, _ = chat_writer_non_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert (
        formatted_messages[0]["content"].strip()
        == "Just a plain string One more string"
    )


def test_non_vision_invalid_dict_type(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(content=[{"type": "image", "url": "http://example.com"}])
    with pytest.raises(TypeError):
        chat_writer_non_vision_model._convert_messages_to_dicts([message])


def test_vision_string(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=["This is vision text"])
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "This is vision text"}
    ]


def test_vision_strings(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=["This is vision text", "This is vision text"])
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "This is vision text"},
        {"type": "text", "text": "This is vision text"},
    ]


def test_vision_text_block(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=[{"type": "text", "text": "This is vision text"}])
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "This is vision text"}
    ]


def test_vision_text_blocks(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "This is vision text"},
            {"type": "text", "text": "This is vision text"},
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "This is vision text"},
        {"type": "text", "text": "This is vision text"},
    ]


def test_vision_image_url_block(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
    ]


def test_vision_image_base64_block(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {
                "type": "image",
                "base64": b"some bytes data",
                "mime_type": "application/octet-stream",
            }
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:application/octet-stream;base64,{base64.b64encode(b'some bytes data').decode('utf-8')}",
            },
        }
    ]


def test_vision_image_block_converted(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[{"type": "image", "url": "http://example.com/img.png"}]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}
    ]


def test_vision_string_and_dict_mixed(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=["some text", {"type": "text", "text": "dict text"}])
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "some text"},
        {"type": "text", "text": "dict text"},
    ]


def test_vision_string_and_image_dict_mixed(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=["some text", {"type": "image", "url": "http://example.com/img.png"}]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]


def test_vision_string_and_image_url_dict_mixed(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            "some text",
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]


def test_vision_text_dict_and_image_dict_mixed(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "dict text"},
            {"type": "image", "url": "http://example.com/img.png"},
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "dict text"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]


def test_vision_text_dict_and_image_url_dict_mixed(
    chat_writer_vision_model: ChatWriter,
):
    message = HumanMessage(
        content=[
            {"type": "text", "text": "dict text"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
    )
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == [
        {"type": "text", "text": "dict text"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]


def test_vision_single_string(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content="plain string")
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )

    assert formatted_messages[0]["content"] == "plain string"


def test_vision_invalid_block_type(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=[{"type": "unsupported", "foo": "bar"}])
    with pytest.raises(TypeError):
        chat_writer_vision_model._convert_messages_to_dicts([message])


def test_empty_content_non_vision(chat_writer_non_vision_model: ChatWriter):
    message = HumanMessage(content=[])
    formatted_messages, _ = chat_writer_non_vision_model._convert_messages_to_dicts(
        [message]
    )
    assert formatted_messages[0]["content"] == ""


def test_empty_content_vision(chat_writer_vision_model: ChatWriter):
    message = HumanMessage(content=[])
    formatted_messages, _ = chat_writer_vision_model._convert_messages_to_dicts(
        [message]
    )
    assert formatted_messages[0]["content"] == ""

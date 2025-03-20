"""Integration tests ChatWriter wrapper

You need WRITER_API_KEY set in your environment to run these tests.
"""

from typing import Optional, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from langchain_writer.chat_models import ChatWriter


def test_chat_model_invoke(chat_writer: ChatWriter):
    response = chat_writer.invoke("Hello")

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


def test_chat_model_response_metadata(chat_writer: ChatWriter):
    result = chat_writer.invoke(
        [HumanMessage(content="How to sleep well?")], logprobs=True
    )
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
        )
    )
    assert all(
        k in result.response_metadata["token_usage"]
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )


def test_chat_model_invoke_stop(chat_writer: ChatWriter):
    response = chat_writer.invoke("Hello", stop=[" "])

    assert response.content[-1] == " "


def test_chat_model_llm_output_contains_model_name(chat_writer: ChatWriter):
    message = HumanMessage(content="Hello")
    llm_result = chat_writer.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat_writer.model_name


def test_chat_model_system_message(chat_writer: ChatWriter):
    system_message = SystemMessage(content="Compose your responses in German language")
    user_message = HumanMessage(content="Hi! How can I can say 'Hi!' to other human?")

    response = chat_writer.invoke([system_message, user_message])

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


def test_chat_model_generation_with_n(chat_writer: ChatWriter):
    chat_writer.n = 2
    message = HumanMessage(content="Hello")
    response = chat_writer.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_chat_model_multiple_completions(chat_writer: ChatWriter):
    chat_writer.n = 5
    message = HumanMessage(content="Hello")
    response = chat_writer._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


def test_chat_model_batch(chat_writer: ChatWriter):
    response = chat_writer.batch(
        [
            "How to cook pancakes?",
            "How to compose poem?",
            "How to run faster?",
        ],
        config={"max_concurrency": 2},
    )

    assert len(response) == 3
    for batch in response:
        assert batch.content


@pytest.mark.asyncio
async def test_chat_model_ainvoke(chat_writer: ChatWriter):
    response = await chat_writer.ainvoke("Hello")

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_chat_model_ainvoke_stop(chat_writer: ChatWriter):
    response = await chat_writer.ainvoke("Hello", stop=[" "])

    assert response.content[-1] == " "


@pytest.mark.asyncio
async def test_chat_model_async_generation_with_n(chat_writer: ChatWriter):
    chat_writer.n = 2
    message = HumanMessage(content="Hello")
    response = await chat_writer.agenerate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_chat_model_async_response_metadata(chat_writer: ChatWriter):
    result = await chat_writer.ainvoke(
        [HumanMessage(content="How to sleep well?")], logprobs=True
    )
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
        )
    )
    assert all(
        k in result.response_metadata["token_usage"]
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
    )


@pytest.mark.asyncio
async def test_chat_model_abatch(chat_writer: ChatWriter):
    response = await chat_writer.abatch(
        [
            "How to cook pancakes?",
            "How to compose poem?",
            "How to run faster?",
        ],
        config={"max_concurrency": 3},
    )

    assert len(response) == 3
    for batch in response:
        assert batch.content


def test_chat_model_streaming(chat_writer: ChatWriter):
    for chunk in chat_writer.stream("Compose a tiny poem"):
        assert isinstance(chunk.content, str)


def test_chat_model_streaming_stop(chat_writer: ChatWriter):
    response = chat_writer.stream("Hello", stop=[" "])
    resulting_text = ""

    for chunk in response:
        resulting_text += chunk.content
        assert isinstance(chunk.content, str)

    assert resulting_text[-1] == " "


def test_chat_model_response_metadata_streaming(chat_writer: ChatWriter):
    full: Optional[BaseMessageChunk] = None
    for chunk in chat_writer.stream("How to sleep well?", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert "finish_reason" in cast(BaseMessageChunk, full).response_metadata
    assert "token_usage" in cast(BaseMessageChunk, full).response_metadata
    assert {"input_tokens", "output_tokens", "total_tokens"}.issubset(
        set(
            cast(BaseMessageChunk, full).response_metadata.get("token_usage", {}).keys()
        )
    )


@pytest.mark.asyncio
async def test_chat_model_response_metadata_astreaming(chat_writer: ChatWriter):
    full: Optional[BaseMessageChunk] = None
    async for chunk in chat_writer.astream("How to sleep well?", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert "finish_reason" in cast(BaseMessageChunk, full).response_metadata
    assert "token_usage" in cast(BaseMessageChunk, full).response_metadata
    assert {"input_tokens", "output_tokens", "total_tokens"}.issubset(
        set(
            cast(BaseMessageChunk, full).response_metadata.get("token_usage", {}).keys()
        )
    )


def test_chat_model_system_message_streaming(chat_writer: ChatWriter):
    system_message = SystemMessage(content="Compose your responses in German language")
    user_message = HumanMessage(content="Hi! How can I can say 'Hi!' to other human?")

    response = chat_writer.stream([system_message, user_message])

    for chunk in response:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.asyncio
async def test_chat_model_astreaming(chat_writer: ChatWriter):
    async for chunk in chat_writer.astream("Compose a tiny poem"):
        assert isinstance(chunk.content, str)


@pytest.mark.asyncio
async def test_chat_model_astreaming_stop(chat_writer: ChatWriter):
    response = chat_writer.astream("Hello", stop=[" "])
    resulting_text = ""

    async for chunk in response:
        resulting_text += chunk.content
        assert isinstance(chunk.content, str)

    assert resulting_text[-1] == " "

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
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_writer import GraphTool
from langchain_writer.chat_models import ChatWriter


@tool
def get_supercopa_trophies_count(club_name: str) -> Optional[int]:
    """Returns information about supercopa trophies count.

    Args:
        club_name: Club you want to investigate info of supercopa trophies about

    Returns:
        Number of supercopa trophies or None if there is no info about requested club
    """

    if club_name == "Barcelona":
        return 15
    elif club_name == "Real Madrid":
        return 13
    elif club_name == "Atletico Madrid":
        return 2
    else:
        return None


@tool
def get_laliga_points(club_name: str) -> Optional[int]:
    """Returns information about points scorred in LaLiga during this season.

    Args:
        club_name: Club you want to investigate info of LaLiga points about

    Returns:
        LaLiga table points or None if there is no info about requested club
    """

    if club_name == "Barcelona":
        return 38
    elif club_name == "Real Madrid":
        return 43
    elif club_name == "Atletico Madrid":
        return 44
    else:
        return None


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


get_product_info = {
    "type": "function",
    "function": {
        "name": "get_product_info",
        "description": "Get information about a product by its id",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "number",
                    "description": "The unique identifier of the product to retrieve information for",
                }
            },
            "required": ["product_id"],
        },
    },
}


def test_chat_model_tool_binding(chat_writer: ChatWriter):
    chat_writer.bind_tools([get_supercopa_trophies_count, get_laliga_points])

    assert len(chat_writer.tools) == 2
    for chat_tool in chat_writer.tools:
        assert chat_tool["function"]["name"] in [
            "get_supercopa_trophies_count",
            "get_laliga_points",
        ]


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


def test_chat_model_tool_calls(chat_writer: ChatWriter):
    chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points], tool_choice="auto"
    )

    response = chat_writer.invoke(
        "Does Barcelona have more supercopa trophies than Real Madrid?"
    )

    assert len(response.tool_calls) == 2
    for tool_call in response.tool_calls:
        assert tool_call["name"] in [
            "get_supercopa_trophies_count",
            "get_laliga_points",
        ]


def test_chat_model_tool_graph_call(chat_writer: ChatWriter, graph_tool: GraphTool):
    chat_writer.bind_tools([graph_tool])

    response = chat_writer.invoke(
        "Use knowledge graph tool to compose this answer. Tell me what th first line of documents stored in your KG"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0


def test_chat_model_tool_dict_definition_call(chat_writer: ChatWriter):
    chat_writer.bind_tools([get_product_info])

    response = chat_writer.invoke(
        "How many sugar does cookie with id: 1243 have per 100 gram?"
    )

    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "get_product_info"


def test_chat_model_tool_graph_and_function_call(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.invoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


def test_chat_model_tool_call_pydantic_definition(chat_writer: ChatWriter):
    chat_writer.bind_tools([GetWeather, GetPopulation], tool_choice="auto")

    response = chat_writer.invoke(
        "Which city is hotter today and which is bigger: LA or NY?"
    )

    assert len(response.tool_calls) == 4
    for tool_call in response.tool_calls:
        assert tool_call["name"] in ["GetWeather", "GetPopulation"]


def test_chat_model_tool_calls_choice(chat_writer: ChatWriter):
    chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points],
        tool_choice="get_laliga_points",
    )

    response = chat_writer.invoke(
        "Does Barcelona have more supercopa trophies than Real Madrid?"
    )

    assert len(response.tool_calls) == 2
    for call in response.tool_calls:
        assert call["name"] == "get_laliga_points"


def test_chat_model_tool_calls_with_tools_outputs(chat_writer: ChatWriter):
    chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points], tool_choice="auto"
    )
    messages = [
        HumanMessage("Does Barcelona have more supercopa trophies than Real Madrid?")
    ]
    response = chat_writer.invoke(messages)
    messages.append(response)

    for tool_call in response.tool_calls:
        selected_tool = {
            "get_laliga_points": get_laliga_points,
            "get_supercopa_trophies_count": get_supercopa_trophies_count,
        }[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    response = chat_writer.invoke(messages)

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


def test_chat_model_function_and_graph_calls_with_tools_outputs(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools(
        [
            get_supercopa_trophies_count,
            get_laliga_points,
            get_product_info,
            GetWeather,
            GetPopulation,
            graph_tool,
        ],
        tool_choice="auto",
    )
    messages = [
        HumanMessage(
            "Use knowledge graph tool to compose this answer. "
            "Tell me what th first line of documents stored in your KG. "
            "Also I want to know: how many SuperCopa trophies have Barcelona won?"
        )
    ]
    response = chat_writer.invoke(messages)
    messages.append(response)

    text_to_check_for_inclusion = response.content.lower()

    for tool_call in response.tool_calls:
        selected_tool = {
            "get_supercopa_trophies_count": get_supercopa_trophies_count,
        }[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    response = chat_writer.invoke(messages)

    text_to_check_for_inclusion += " " + response.content.lower()

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0
    assert any(
        [
            word in text_to_check_for_inclusion
            for word in ["supercopa", "barcelona", "trophies", "15"]
        ]
    )
    assert any(
        [
            word in text_to_check_for_inclusion
            for word in ["knowledge", "graph", "line", "document"]
        ]
    )


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
async def test_chat_model_tool_graph_acall(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool])

    response = await chat_writer.ainvoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_model_tool_graph_and_function_acall(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = await chat_writer.ainvoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


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


def test_chat_model_system_message_streaming(chat_writer: ChatWriter):
    system_message = SystemMessage(content="Compose your responses in German language")
    user_message = HumanMessage(content="Hi! How can I can say 'Hi!' to other human?")

    response = chat_writer.stream([system_message, user_message])

    for chunk in response:
        assert isinstance(chunk, AIMessageChunk)


def test_chat_model_tool_calls_streaming(chat_writer: ChatWriter):
    chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points],
        tool_choice="get_laliga_points",
    )

    response = chat_writer.stream(
        "Does Barcelona have more supercopa trophies than Real Madrid?"
    )

    for chunk in response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.tool_call_chunks:
            assert (
                chunk.tool_call_chunks[0]["args"] or chunk.tool_call_chunks[0]["name"]
            )


def test_chat_model_tool_calls_with_tools_outputs_stream(chat_writer: ChatWriter):
    chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points], tool_choice="auto"
    )
    messages = [
        HumanMessage("Does Barcelona have more supercopa trophies than Real Madrid?")
    ]
    response = chat_writer.invoke(messages)
    messages.append(response)

    for tool_call in response.tool_calls:
        selected_tool = {
            "get_laliga_points": get_laliga_points,
            "get_supercopa_trophies_count": get_supercopa_trophies_count,
        }[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    response = chat_writer.stream(messages)

    for chunk in response:
        assert isinstance(chunk.content, str)
        assert len(chunk.id) > 0


def test_chat_model_tool_graph_call_streaming(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool])

    response = chat_writer.stream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("graph_data")
    assert len(full.additional_kwargs["graph_data"]["sources"]) > 0


def test_chat_model_tool_function_graph_call_streaming(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.stream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("graph_data")
    assert len(full.additional_kwargs["graph_data"]["sources"]) > 0
    assert full.tool_calls
    assert len(full.tool_calls) == 1


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


@pytest.mark.asyncio
async def test_chat_model_tool_graph_call_streaming_async(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool])

    response = chat_writer.astream(
        "Use knowledge graph tool to compose this answer. Tell me what th first line of documents stored in your KG"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("graph_data")
    assert len(full.additional_kwargs["graph_data"]["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_model_tool_function_graph_call_astreaming(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.astream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what th first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("graph_data")
    assert full.additional_kwargs.get("graph_data")["sources"]
    assert len(full.additional_kwargs["graph_data"]["sources"]) > 0
    assert full.tool_calls
    assert len(full.tool_calls) == 1

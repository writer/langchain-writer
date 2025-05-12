import sys
from typing import Optional

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from writerai import BadRequestError
from writerai.types import ApplicationGenerateContentResponse

from langchain_writer import ChatWriter, GraphTool
from langchain_writer.tools import LLMTool, NoCodeAppTool, TranslationTool
from tests.integration_tests.conftest import (
    RESEARCH_APP_ID,
    TEXT_GENERATION_APP_ID,
    get_app_inputs,
)

if sys.version_info[0] < 3.10:

    async def anext(ait):
        return await ait.__anext__()


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


def test_translation_tool(chat_writer: ChatWriter, translation_tool: TranslationTool):
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.invoke(
        "Translate the following message to spanish: 'Hello, world!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "en"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "es"
    )


def test_translation_tool_source_lang_auto(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.invoke(
        "Translate the following message to Italian: 'Guten tag, queridos amigos!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "auto"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "it"
    )


def test_translation_tool_source_lang(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.invoke(
        "Translate the following message to Italian: 'Guten tag, liebe freunde!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "de"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "it"
    )


def test_translation_tool_target_lang(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.target_language_code = "fr"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.invoke(
        "Translate the following message: 'Guten tag, liebe freunde!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "de"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "en"
    )


def test_translation_tool_wrong_source_language(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "gg"
    chat_writer = chat_writer.bind_tools([translation_tool])

    with pytest.raises(BadRequestError):
        chat_writer.invoke(
            "Translate the following message: 'Guten tag, liebe freunde!'"
        )


def test_chat_model_tool_binding(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools(
        [get_supercopa_trophies_count, get_laliga_points]
    )

    assert len(chat_writer.kwargs["tools"]) == 2
    for chat_tool in chat_writer.kwargs["tools"]:
        assert chat_tool["function"]["name"] in [
            "get_supercopa_trophies_count",
            "get_laliga_points",
        ]


def test_chat_model_tool_calls(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools(
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
    chat_writer = chat_writer.bind_tools([graph_tool])

    response = chat_writer.invoke(
        "Use knowledge graph tool to compose this answer. Tell me what the first line of documents stored in your KG"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0


def test_chat_model_tool_llm_call(chat_writer: ChatWriter, llm_tool: LLMTool):
    chat_writer = chat_writer.bind_tools([llm_tool])

    response = chat_writer.invoke("Use LLM tool and tell me about Newton binom.")

    assert response.additional_kwargs.get("llm_data")
    assert any(
        [
            word in response.additional_kwargs["llm_data"]["prompt"]
            for word in ["Newton", "binom"]
        ]
    )


def test_chat_model_tool_dict_definition_call(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools([get_product_info])

    response = chat_writer.invoke(
        "How many sugar does cookie with id: 1243 have per 100 gram?"
    )

    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "get_product_info"


def test_chat_model_tool_graph_and_function_call(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer = chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.invoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


def test_chat_model_tool_llm_and_function_call(
    chat_writer: ChatWriter, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, get_supercopa_trophies_count])

    response = chat_writer.invoke(
        "Use LLM tool to compose this answer. "
        "Tell me how Newton binom works."
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("llm_data")
    assert len(response.additional_kwargs["llm_data"]["prompt"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


def test_chat_model_tool_call_pydantic_definition(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools(
        [GetWeather, GetPopulation], tool_choice="auto"
    )

    response = chat_writer.invoke(
        "Which city is hotter today and which is bigger: LA or NY?"
    )

    assert len(response.tool_calls) == 4
    for tool_call in response.tool_calls:
        assert tool_call["name"] in ["GetWeather", "GetPopulation"]


def test_chat_model_tool_calls_choice(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools(
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
    chat_writer = chat_writer.bind_tools(
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
    chat_writer = chat_writer.bind_tools(
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
            "Tell me what the first line of documents stored in your KG. "
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


@pytest.mark.asyncio
async def test_chat_model_tool_graph_acall(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer = chat_writer.bind_tools([graph_tool])

    response = await chat_writer.ainvoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_model_tool_llm_acall(chat_writer: ChatWriter, llm_tool: LLMTool):
    chat_writer = chat_writer.bind_tools([llm_tool])

    response = await chat_writer.ainvoke("Use LLM tool and tell me about Newton binom.")

    assert response.additional_kwargs.get("llm_data")
    assert any(
        [
            word in response.additional_kwargs["llm_data"]["prompt"]
            for word in ["Newton", "binom"]
        ]
    )


@pytest.mark.asyncio
async def test_chat_model_tool_graph_and_function_acall(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer = chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = await chat_writer.ainvoke(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG. "
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("graph_data")
    assert len(response.additional_kwargs["graph_data"]["sources"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


@pytest.mark.asyncio
async def test_chat_model_tool_llm_and_function_acall(
    chat_writer: ChatWriter, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, get_supercopa_trophies_count])

    response = await chat_writer.ainvoke(
        "Use LLM tool to compose this answer. "
        "Tell me how Newton binom works."
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    assert response.additional_kwargs.get("llm_data")
    assert len(response.additional_kwargs["llm_data"]["prompt"]) > 0
    assert response.tool_calls
    assert len(response.tool_calls) == 1


@pytest.mark.asyncio
async def test_translation_tool_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = await chat_writer.ainvoke(
        "Translate the following message to spanish: 'Hello, world!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "en"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "es"
    )


@pytest.mark.asyncio
async def test_translation_tool_source_lang_auto_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = await chat_writer.ainvoke(
        "Translate the following message to italian: 'Guten tag, queridos amigos!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "auto"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "it"
    )


@pytest.mark.asyncio
async def test_translation_tool_source_lang_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = await chat_writer.ainvoke(
        "Translate the following message to Italian: 'Guten tag, liebe freunde!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "de"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "it"
    )


@pytest.mark.asyncio
async def test_translation_tool_target_lang_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.target_language_code = "fr"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = await chat_writer.ainvoke(
        "Translate the following message: 'Guten tag, liebe freunde!'"
    )

    assert response.content
    assert response.additional_kwargs
    assert "translation_data" in response.additional_kwargs.keys()
    assert (
        response.additional_kwargs["translation_data"]["source_language_code"] == "de"
    )
    assert (
        response.additional_kwargs["translation_data"]["target_language_code"] == "en"
    )


@pytest.mark.asyncio
async def test_translation_tool_wrong_source_language_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "gg"
    chat_writer = chat_writer.bind_tools([translation_tool])

    with pytest.raises(BadRequestError):
        await chat_writer.ainvoke(
            "Translate the following message: 'Guten tag, liebe freunde!'"
        )


def test_chat_model_tool_calls_streaming(chat_writer: ChatWriter):
    chat_writer = chat_writer.bind_tools(
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
    chat_writer = chat_writer.bind_tools(
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
    chat_writer = chat_writer.bind_tools([graph_tool])

    response = chat_writer.stream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG"
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
    chat_writer = chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.stream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG. "
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


def test_translation_tool_streaming(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.stream(
        "Translate the following message to spanish: "
        "'Hello, world! I like programming on Python, "
        "and I want to share my knowledge with you."
        "Python is an interpreted programming language "
        "that can be used to perform a wide variety of tasks.'"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "en"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "es"


def test_translation_tool_source_lang_auto_streaming(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.stream(
        "Translate the following message to italian: 'Guten tag, queridos amigos!'"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "auto"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "it"


def test_translation_tool_source_lang_streaming(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.stream(
        "Translate the following message to Italian: 'Guten tag, liebe freunde!'"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "de"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "it"


def test_translation_tool_target_lang_streaming(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.target_language_code = "fr"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.stream(
        "Translate the following message: 'Guten tag, liebe freunde!'"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "de"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "en"


def test_translation_tool_wrong_source_language_streaming(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "gg"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.stream(
        "Translate the following message to spanish: "
        "'Hello, world! I like programming on Python, "
        "and I want to share my knowledge with you."
        "Python is an interpreted programming language "
        "that can be used to perform a wide variety of tasks.'"
    )

    with pytest.raises(BadRequestError):
        for _ in response:
            ...


@pytest.mark.asyncio
async def test_chat_model_tool_graph_call_streaming_async(
    chat_writer: ChatWriter, graph_tool: GraphTool
):
    chat_writer = chat_writer.bind_tools([graph_tool])

    response = chat_writer.astream(
        "Use knowledge graph tool to compose this answer. Tell me what the first line of documents stored in your KG"
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
    chat_writer = chat_writer.bind_tools([graph_tool, get_supercopa_trophies_count])

    response = chat_writer.astream(
        "Use knowledge graph tool to compose this answer. "
        "Tell me what the first line of documents stored in your KG. "
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


def test_chat_model_tool_llm_call_streaming(chat_writer: ChatWriter, llm_tool: LLMTool):
    chat_writer = chat_writer.bind_tools([llm_tool])

    response = chat_writer.stream("Use LLM tool and tell me about Newton binom.")

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("llm_data")
    assert len(full.additional_kwargs["llm_data"]["prompt"]) > 0


def test_chat_model_tool_function_llm_call_streaming(
    chat_writer: ChatWriter, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, get_supercopa_trophies_count])

    response = chat_writer.stream(
        "Use LLM tool to compose this answer. "
        "Tell me how Newton binom works."
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    full = next(response)
    for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("llm_data")
    assert len(full.additional_kwargs["llm_data"]["prompt"]) > 0
    assert full.tool_calls
    assert len(full.tool_calls) == 1


@pytest.mark.asyncio
async def test_chat_model_tool_llm_call_streaming_async(
    chat_writer: ChatWriter, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool])

    response = chat_writer.astream("Use LLM tool and tell me about Newton binom.")

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("llm_data")
    assert len(full.additional_kwargs["llm_data"]["prompt"]) > 0


@pytest.mark.asyncio
async def test_chat_model_tool_function_llm_call_astreaming(
    chat_writer: ChatWriter, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, get_supercopa_trophies_count])

    response = chat_writer.astream(
        "Use LLM tool to compose this answer. "
        "Tell me how Newton binom works."
        "Also I want to know: how many SuperCopa trophies have Barcelona won?"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.additional_kwargs.get("llm_data")
    assert full.additional_kwargs.get("llm_data")["prompt"]
    assert len(full.additional_kwargs["llm_data"]["prompt"]) > 0
    assert full.tool_calls
    assert len(full.tool_calls) == 1


def test_graph_llm_tool_call_error(
    chat_writer: ChatWriter, graph_tool: GraphTool, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, graph_tool])

    with pytest.raises(BadRequestError):
        chat_writer.invoke("Hello")


@pytest.mark.asyncio
async def test_graph_llm_tool_acall_error(
    chat_writer: ChatWriter, graph_tool: GraphTool, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, graph_tool])

    with pytest.raises(BadRequestError):
        await chat_writer.ainvoke("Hello")


def test_graph_llm_tool_streaming_error(
    chat_writer: ChatWriter, graph_tool: GraphTool, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, graph_tool])

    with pytest.raises(BadRequestError):
        for _ in chat_writer.stream("Hello"):
            ...


@pytest.mark.asyncio
async def test_graph_llm_tool_astreaming_error(
    chat_writer: ChatWriter, graph_tool: GraphTool, llm_tool: LLMTool
):
    chat_writer = chat_writer.bind_tools([llm_tool, graph_tool])

    with pytest.raises(BadRequestError):
        async for _ in chat_writer.astream("Hello"):
            ...


@pytest.mark.parametrize("app_id", [TEXT_GENERATION_APP_ID, RESEARCH_APP_ID])
def test_app_initialization(app_id):
    app_tool = NoCodeAppTool(app_id=app_id, name="Names library application")

    assert app_tool.app_id == app_id
    assert app_tool.name == "Names library application"
    assert len(app_tool.app_inputs) > 0


def test_no_code_app_run(no_code_app_tool: NoCodeAppTool):
    inputs = get_app_inputs(no_code_app_tool)

    result = no_code_app_tool.run(tool_input={"inputs": inputs})

    assert isinstance(result, ApplicationGenerateContentResponse)
    assert len(result.suggestion) > 0


@pytest.mark.asyncio
async def test_no_code_app_arun(no_code_app_tool: NoCodeAppTool):
    inputs = get_app_inputs(no_code_app_tool)

    result = await no_code_app_tool.arun(tool_input={"inputs": inputs})

    assert isinstance(result, ApplicationGenerateContentResponse)
    assert len(result.suggestion) > 0


def test_no_code_app_binded_call(
    chat_writer: ChatWriter, no_code_app_tool: NoCodeAppTool
):
    chat_writer = chat_writer.bind_tools([no_code_app_tool], tool_choice="auto")

    response = chat_writer.invoke(
        "Use no code app tool and create an answer, based on no-code app invocation result. "
        "Fill all app inputs by random strings"
    )

    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "No-code application"


@pytest.mark.asyncio
async def test_no_code_app_binded_acall(
    chat_writer: ChatWriter, no_code_app_tool: NoCodeAppTool
):
    chat_writer = chat_writer.bind_tools([no_code_app_tool], tool_choice="auto")

    response = await chat_writer.ainvoke(
        "Use no code app tool and create an answer, based on no-code app invocation result. "
        "Fill all app inputs by random strings"
    )

    assert response.tool_calls
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "No-code application"


def test_no_code_app_binded_streaming(
    chat_writer: ChatWriter, no_code_app_tool: NoCodeAppTool
):
    chat_writer = chat_writer.bind_tools([no_code_app_tool], tool_choice="auto")

    response = chat_writer.stream(
        "Use no code app tool and create an answer, based on no-code app invocation result. "
        "Fill all app inputs by random strings"
    )

    for chunk in response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.tool_call_chunks:
            assert (
                chunk.tool_call_chunks[0]["args"]
                or chunk.tool_call_chunks[0]["name"] == "No-code application"
            )


@pytest.mark.asyncio
async def test_no_code_app_binded_astreaming(
    chat_writer: ChatWriter, no_code_app_tool: NoCodeAppTool
):
    chat_writer = chat_writer.bind_tools([no_code_app_tool], tool_choice="auto")

    response = chat_writer.astream(
        "Use no code app tool and create an answer, based on no-code app invocation result. "
        "Fill all app inputs by random strings"
    )

    async for chunk in response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.tool_call_chunks:
            assert (
                chunk.tool_call_chunks[0]["args"]
                or chunk.tool_call_chunks[0]["name"] == "No-code application"
            )


@pytest.mark.asyncio
async def test_translation_tool_streaming_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.astream(
        "Translate the following message to spanish: "
        "'Hello, world! I like programming on Python, "
        "and I want to share my knowledge with you."
        "Python is an interpreted programming language "
        "that can be used to perform a wide variety of tasks.'"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "en"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "es"


@pytest.mark.asyncio
async def test_translation_tool_source_lang_auto_streaming_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.astream(
        "Translate the following message to italian: 'Guten tag, queridos amigos!'"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "auto"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "it"


@pytest.mark.asyncio
async def test_translation_tool_source_lang_streaming_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "de"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.astream(
        "Translate the following message to Italian: 'Guten tag, liebe freunde!'"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "de"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "it"


@pytest.mark.asyncio
async def test_translation_tool_target_lang_streaming_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.target_language_code = "fr"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.astream(
        "Translate the following message: 'Guten tag, liebe freunde!'"
    )

    full = await anext(response)
    async for chunk in response:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.content
    assert full.additional_kwargs
    assert "translation_data" in full.additional_kwargs.keys()
    assert full.additional_kwargs["translation_data"]["source_language_code"] == "de"
    assert full.additional_kwargs["translation_data"]["target_language_code"] == "en"


@pytest.mark.asyncio
async def test_translation_tool_wrong_source_language_streaming_async(
    chat_writer: ChatWriter, translation_tool: TranslationTool
):
    translation_tool.source_language_code = "gg"
    chat_writer = chat_writer.bind_tools([translation_tool])

    response = chat_writer.astream(
        "Translate the following message to spanish: "
        "'Hello, world! I like programming on Python, "
        "and I want to share my knowledge with you."
        "Python is an interpreted programming language "
        "that can be used to perform a wide variety of tasks.'"
    )

    with pytest.raises(BadRequestError):
        async for _ in response:
            ...

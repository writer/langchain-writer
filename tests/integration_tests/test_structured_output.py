from enum import Enum
from typing import Annotated, List

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict


class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"


class User(BaseModel):
    name: str = Field(..., description="The user's full name")
    age: int = Field(..., ge=0, description="The user's age in years")
    married: bool = Field(..., description="Marital status of the user")
    gender: Gender = Field(..., description="The user's gender")
    skills: List[str] = Field(
        default_factory=list, description="List of the user's skills (up to 2)"
    )


class UsersList(BaseModel):
    users: List[User] = Field(default_factory=list, description="List of users")


class Domain(str, Enum):
    tech = "tech"
    finance = "finance"
    healthcare = "healthcare"
    education = "education"
    manufacturing = "manufacturing"


class Company(TypedDict):
    name: Annotated[str, "The name of the company"]
    num_employees: Annotated[int, "Total number of employees in the company"]
    domain: Annotated[Domain, "The business domain or industry of the company"]
    is_resident: Annotated[
        bool, "Flag indicating whether the company is a local resident entity"
    ]
    contragents: Annotated[
        List[str], "List of names of contragent companies working with this company"
    ]


address_tool_schema = {
    "name": "AddressDefinition",
    "description": "Defines the details of a specific address including its location, type, and hosted companies.",
    "parameters": {
        "type": "object",
        "properties": {
            "street": {"type": "string", "description": "Name of the street"},
            "city": {"type": "string", "description": "Name of the city"},
            "number": {"type": "integer", "description": "House or building number"},
            "is_town": {
                "type": "boolean",
                "description": "Flag indicating if the address is in a town (True) or not (False)",
            },
            "companies": {
                "type": "array",
                "description": "List of company names hosted in this house (optional)",
                "items": {"type": "string"},
            },
        },
        "required": ["street", "city", "number", "is_town"],
    },
}

service_json_schema = {
    "json_schema": {
        "schema": {
            "type": "object",
            "properties": {
                "name": {"description": "The name of the service", "type": "string"},
                "price": {
                    "description": "The price of the service as a floating-point number",
                    "type": "number",
                },
                "guarantee": {
                    "description": "Indicates whether a guarantee is provided for the service",
                    "type": "boolean",
                },
                "required_materials": {
                    "description": "A list of materials required to provide the service (optional)",
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "price", "guarantee"],
        },
        "name": "CompanyService",
    },
    "type": "json_schema",
}

prompts = {
    "json_schema": """In recent years, home automation has seen a significant rise in both demand and innovation.
     Consumers are looking for reliable services that integrate seamlessly with their smart ecosystems.
     One standout offering on the market is the Smart Thermostat Installation, priced at 199.99 with a full
     satisfaction guarantee. Required materials include a compatible thermostat unit, mounting hardware, and wiring
     adapters. Meanwhile, eco-conscious homeowners often opt for the Solar Panel Cleaning service. It costs just 79.00
     and also includes a guarantee, ensuring that any re-soiling within a week is covered. Materials such as
     biodegradable detergent, water-fed poles, and microfiber pads are commonly used. However, not all services come
     with a guarantee. The Drone-Based Roof Inspection, for example, is priced at 129.50 but does not include one.
     Despite that, it's still popular due to its speed and non-invasive approach—requiring only the drone and a battery
     backup. Such service offerings continue to shape consumer expectations, combining technical efficiency with a
     growing emphasis on environmental and operational transparency.""",
    "pydantic_model": "Generate a list of 5 users. Provide generated data in required format.",
    "tool_schema": """Here is a text. Fetch from it addresses an related information, use provided desired output
     format. Return full address of building that hosts the largest amount of businesses.
     The wind carried the scent of smoke and machinery through the narrow streets of Kraków as Maja
     stepped off the tram at Elm Street. Number 24 loomed ahead, a pale stone building with high windows and rusted
     mailboxes. BrightForge occupied the second floor — its logo barely visible behind dust — and above it,
     MetricLoop's developers kept the lights on late into the night. In Zielona Dolina, far from the city’s iron pulse,
     there was only the rustle of trees and the hush of windmills. Lipowa Street curled gently through the hills, and
     at number 9, a wooden house stood quietly. The town kept its records tidy, and no business name had ever been
     scribbled beside that address. Further north, along the merchant veins of Gdańsk, Długa Street stretched in
     golden cobblestones. At number 102, a sleek modernist block reflected the gray sky. OceanNet had taken the lower
     floors. Polymark’s employees sipped espresso in the courtyard. On the top floor, AmberSys coded silently behind
     blackout blinds. Szeged felt older. Softer. Arany János Street bent like an elbow, and at number 55 stood a squat
     yellow building with worn steps. Locals claimed it held a bakery and a cloth repair shop, but the sign had fallen
     off years ago, and the register hadn’t been updated.""",
    "typed_dict": "Tell me about Tesla. Provide output in provided format.",
}

input_params = [
    (service_json_schema, prompts["json_schema"], "json_schema", True),
    (service_json_schema, prompts["json_schema"], "function_calling", True),
    (service_json_schema, prompts["json_schema"], "json_schema", False),
    (service_json_schema, prompts["json_schema"], "function_calling", False),
    (UsersList, prompts["pydantic_model"], "json_schema", True),
    (UsersList, prompts["pydantic_model"], "function_calling", True),
    (UsersList, prompts["pydantic_model"], "json_schema", False),
    (UsersList, prompts["pydantic_model"], "function_calling", False),
    (Company, prompts["typed_dict"], "json_schema", True),
    (Company, prompts["typed_dict"], "function_calling", True),
    (Company, prompts["typed_dict"], "json_schema", False),
    (Company, prompts["typed_dict"], "function_calling", False),
    (address_tool_schema, prompts["tool_schema"], "json_schema", True),
    (address_tool_schema, prompts["tool_schema"], "function_calling", True),
    (address_tool_schema, prompts["tool_schema"], "json_schema", False),
    (address_tool_schema, prompts["tool_schema"], "function_calling", False),
]


def _validate_fields(response: dict | BaseModel, schema: dict | BaseModel) -> bool:
    if isinstance(response, BaseModel):
        response = response.model_dump()
        schema_keys = set(schema.model_fields.keys())
    elif isinstance(response, dict):
        try:
            schema_keys = set(schema.keys())
        except TypeError:
            schema_keys = set(schema.__annotations__.keys())

    response_keys = set(response.keys())

    if isinstance(schema, dict):
        if "json_schema" in schema:
            schema_keys = set(schema["json_schema"]["schema"]["properties"].keys())
        elif "schema" in schema:
            schema_keys = set(schema["schema"]["properties"].keys())
        elif "function" in schema:
            schema_keys = set(schema["function"]["parameters"]["properties"].keys())
        elif "parameters" in schema:
            schema_keys = set(schema["parameters"]["properties"].keys())

    return response_keys.issubset(schema_keys)


@pytest.mark.parametrize("schema, prompt, method, include_raw", input_params)
def test_structured_outputs_sync(schema, prompt, method, include_raw, chat_writer):
    structured_chat = chat_writer.with_structured_output(
        schema=schema, method=method, include_raw=include_raw
    )

    response = structured_chat.invoke(input=prompt)

    if include_raw:
        assert isinstance(response["raw"], AIMessage)
        assert response["parsing_error"] is None
        assert isinstance(response["parsed"], dict) or isinstance(
            response["parsed"], schema
        )

        if method == "tool_calling":
            assert len(response["raw"].tool_calls) > 0

        assert _validate_fields(response["parsed"], schema)
    else:
        assert isinstance(response, dict) or isinstance(response, schema)
        assert _validate_fields(response, schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema, prompt, method, include_raw", input_params)
async def test_structured_outputs_async(
    schema, prompt, method, include_raw, chat_writer
):
    structured_chat = chat_writer.with_structured_output(
        schema=schema, method=method, include_raw=include_raw
    )

    response = await structured_chat.ainvoke(input=prompt)

    if include_raw:
        assert isinstance(response["raw"], AIMessage)
        assert response["parsing_error"] is None
        assert isinstance(response["parsed"], dict) or isinstance(
            response["parsed"], schema
        )

        if method == "tool_calling":
            assert len(response["raw"].tool_calls) > 0
        assert _validate_fields(response["parsed"], schema)
    else:
        assert isinstance(response, dict) or isinstance(response, schema)
        assert _validate_fields(response, schema)


@pytest.mark.parametrize("schema, prompt, method, include_raw", input_params)
def test_structured_outputs_sync_streaming(
    schema, prompt, method, include_raw, chat_writer
):
    structured_chat = chat_writer.with_structured_output(
        schema=schema, method=method, include_raw=include_raw
    )

    stream = structured_chat.stream(input=prompt)

    response = next(stream)
    for chunk in stream:
        assert isinstance(response, BaseModel) or isinstance(response, dict)
        if include_raw:
            response.update(chunk)
        else:
            response = chunk

    if include_raw:
        assert isinstance(response["raw"], AIMessage)
        assert response["parsing_error"] is None
        assert isinstance(response["parsed"], dict) or isinstance(
            response["parsed"], schema
        )

        if method == "tool_calling":
            assert len(response["raw"].tool_calls) > 0
        assert _validate_fields(response["parsed"], schema)
    else:
        assert isinstance(response, dict) or isinstance(response, schema)
        assert _validate_fields(response, schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema, prompt, method, include_raw", input_params)
async def test_structured_outputs_async_streaming(
    schema, prompt, method, include_raw, chat_writer
):
    structured_chat = chat_writer.with_structured_output(
        schema=schema, method=method, include_raw=include_raw
    )

    stream = structured_chat.astream(input=prompt)

    response = await anext(stream)
    async for chunk in stream:
        assert isinstance(response, BaseModel) or isinstance(response, dict)
        if include_raw:
            response.update(chunk)
        else:
            response = chunk

    if include_raw:
        assert isinstance(response["raw"], AIMessage)
        assert response["parsing_error"] is None
        assert isinstance(response["parsed"], dict) or isinstance(
            response["parsed"], schema
        )

        if method == "tool_calling":
            assert len(response["raw"].tool_calls) > 0
        assert _validate_fields(response["parsed"], schema)
    else:
        assert isinstance(response, dict) or isinstance(response, schema)
        assert _validate_fields(response, schema)


def test_schema_parsing_error_with_raw(chat_writer):
    structured_chat = chat_writer.with_structured_output(
        schema=User, method="function_calling", include_raw=True
    )

    response = structured_chat.invoke(input="Parse this user. Name: John. Age: -19")

    assert response["raw"]
    assert response["parsing_error"]
    assert len(response["parsing_error"].errors()) > 0
    assert response["parsed"] is None


def test_schema_parsing_error(chat_writer):
    structured_chat = chat_writer.with_structured_output(
        schema=User, method="function_calling", include_raw=False
    )

    with pytest.raises(ValidationError):
        structured_chat.invoke(input="Parse this user. Name: John. Age: -19")

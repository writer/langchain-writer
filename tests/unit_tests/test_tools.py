from datetime import datetime
from unittest.mock import patch

import pytest
from writerai.resources import ApplicationsResource
from writerai.types.application_retrieve_response import (
    ApplicationRetrieveResponse,
    Input,
)

from langchain_writer.tools import NoCodeAppTool


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "Name": ["Joe", "Tadej"],
            "File input": "file input id",
            "Image input": "image input id",
        },
        {"Name": "Mark"},
    ],
)
def test_test_inputs_success(inputs):
    with patch.object(
        ApplicationsResource,
        "retrieve",
        lambda x, y: ApplicationRetrieveResponse(
            id="id",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            name="name",
            status="deployed",
            type="generation",
            inputs=[
                Input(input_type="text", name="Name", required=True),
                Input(input_type="dropdown", name="T-Shirt size", required=False),
                Input(input_type="file", name="File input", required=False),
                Input(input_type="media", name="Image input", required=False),
            ],
        ),
    ):

        no_code_app_tool = NoCodeAppTool(api_key="api_key", app_id="app_id")

        converted_inputs = no_code_app_tool.convert_inputs(inputs)

        assert len(converted_inputs) == len(inputs.keys())
        assert all(
            [inputs.get(converted_input["id"]) for converted_input in converted_inputs]
        )


@pytest.mark.parametrize(
    "inputs, error_message_snippet",
    [
        [
            {
                "Name": "",
                "File input": "file input id",
                "Image input": "image input id",
            },
            "required to run this no-code app tool.",
        ],
        [
            {
                "File input": "file input id",
                "Image input": "image input id",
            },
            "required to run this no-code app tool.",
        ],
        [{"Name": 14}, "Input should be a string or list of strings."],
        [{"Name": ["a", 1]}, "Input should be a string or list of strings."],
        [{}, "To run no-code app tool you must pass non-empty inputs dict."],
    ],
)
def test_test_inputs_error(inputs, error_message_snippet):
    with patch.object(
        ApplicationsResource,
        "retrieve",
        lambda x, y: ApplicationRetrieveResponse(
            id="id",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            name="name",
            status="deployed",
            type="generation",
            inputs=[
                Input(input_type="text", name="Name", required=True),
                Input(input_type="dropdown", name="T-Shirt size", required=False),
                Input(input_type="file", name="File input", required=False),
                Input(input_type="media", name="Image input", required=False),
            ],
        ),
    ):

        no_code_app_tool = NoCodeAppTool(api_key="api_key", app_id="app_id")

        with pytest.raises(ValueError) as info:
            no_code_app_tool.convert_inputs(inputs)

            assert error_message_snippet in str(info.value)

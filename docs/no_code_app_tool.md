# NoCodeAppTool

`NoCodeAppTool` is a specialized tool in the Writer LangChain integration that enables access to Writer's no-code applications as LLM tools. This tool allows language models to interact with pre-built applications to enhance their responses.

**Example notebook**: [tools.ipynb](./tools.ipynb)

## Overview

The `NoCodeAppTool` is designed to wrap Writer's no-code applications as tools for language models. Unlike the `GraphTool` and `LLMTool`, the `NoCodeAppTool` is based on the standard "function" tool type and requires manual execution and passing tool call results into the message history.

## Installation

```bash
pip install -U langchain-writer
```

You'll need to set up your Writer API key:

```bash
export WRITER_API_KEY="your-api-key"
```

## Initialization

```python
from langchain_writer.tools import NoCodeAppTool

# Create a NoCodeAppTool with a specific app ID
no_code_app_tool = NoCodeAppTool(
    app_id="your-app-id",        # ID of the no-code application to use
    name="Custom App Name",      # Optional: custom name for the tool
    description="Custom description"  # Optional: custom description
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["function"]` | `"function"` | The tool type, always "function" for NoCodeAppTool |
| `name` | `str` | `"No-code application"` | The name of the tool |
| `description` | `str` | `"No-code application powered by Palmyra"` | Description passed to the model |
| `app_id` | `str` | Required | ID of the no-code application to use |
| `app_inputs` | `list[ResponseInput]` | Auto-initialized | List of input parameters for the app |
| `api_key` | `SecretStr` | From env var | Writer API key |

## Usage with ChatWriter

The `NoCodeAppTool` is designed to be used with the `ChatWriter` class:

```python
from langchain_writer import ChatWriter
from langchain_writer.tools import NoCodeAppTool
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

chat = ChatWriter()

# Create a NoCodeAppTool
app_tool = NoCodeAppTool(
    app_id=os.getenv("APP_ID"),
    name="Social post generator",
    description="No-code app that generates social posts from product descriptions"
)

# Bind the tool ChatWriter
chat_with_tools = chat.bind_tools([app_tool])

product_description = "The Terra running shoe is a high-performance, lightweight shoe that offers a comfortable and durable experience. The shoe is also lightweight and easy to carry, making it an ideal option for runners looking for a new pair of shoes."

# Create a conversation
messages = [
    HumanMessage("Can you help me generate a social post about a new running shoe? Here is the product description:\n\n" + product_description)
]

# Get the model's response
response = chat_with_tools.invoke(messages)
messages.append(response)

# If the model requests to use the tool
if response.tool_calls:
    for tool_call in response.tool_calls:
        print("\nTool call:", tool_call)

        # Dynamically build the inputs dictionary from the args
        inputs = {}
        for arg in tool_call['args']:
            inputs[arg] = tool_call['args'][arg]

        # Execute the tool call
        try:
            tool_response = app_tool.run(tool_input={"inputs": inputs})
            messages.append(tool_response.suggestion)
        except Exception as e:
            print(f"Error running tool: {e}")

    # Get the final response
    try:
        final_response = chat_with_tools.invoke(messages)
        print("\nFinal response:", final_response.content)
    except Exception as e:
        print(f"Error getting final response: {e}")
else:
    print("No tool calls were requested.")
```

## Input Handling

The `NoCodeAppTool` automatically retrieves the input parameters for the app during initialization. When invoking the tool, you need to provide inputs in a specific format:

```python
# Example of providing inputs to the tool
inputs = {
    "Input name": "Input value",
    "Another input name": ["List", "of", "string", "values"]
}

# Invoke the tool with the inputs
result = app_tool.run(tool_input={"inputs": inputs})
```

Input values can be:
- Strings
- Lists of strings

## Important notes

1. The `NoCodeAppTool` requires manual execution, unlike the `GraphTool` and `LLMTool` which are executed remotely by Writer's servers.

2. The `app_id` parameter is crucial as it specifies which no-code application to use. This ID should correspond to an application that has been created in your Writer account.

3. The tool automatically retrieves the input parameters for the app during initialization, so you don't need to specify them manually.

4. Required inputs must be provided when invoking the tool, or a `ValueError` will be raised.

## Use cases

- Generating creative content (poems, stories, etc.)
- Performing specialized data transformations
- Creating custom outputs based on user inputs
- Integrating with domain-specific applications
- Enhancing responses with formatted or structured content
- Leveraging pre-built applications for common tasks

# LLMTool

`LLMTool` is a specialized tool in the Writer LangChain integration that enables delegation to specific Writer models. This tool allows language models to delegate calls to different Palmyra model types to enhance their responses.

**Example notebook**: [tools.ipynb](./tools.ipynb)

## Overview

The `LLMTool` is designed specifically for use within the Writer environment and has a type of "llm" rather than the standard "function" type used by most LangChain tools. It's important to note that this tool does not support direct invocations and is meant to be used only within the Writer chat environment.

## Installation

```bash
pip install -U langchain-writer
```

## Initialization

```python
from langchain_writer.tools import LLMTool

# Create an LLMTool with a specific model
llm_tool = LLMTool(
    model_name="palmyra-med"  # Specify the Palmyra model to use
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["llm"]` | `"llm"` | The tool type, always "llm" for LLMTool |
| `name` | `str` | `"Large Language Model"` | The name of the tool |
| `description` | `str` | `"LLM tool provides a way to perform sub calls to another type of model to compose response on user request."` | Description passed to the model |
| `model_name` | `Literal["palmyra-x-004", "palmyra-x-003-instruct", "palmyra-med", "palmyra-fin", "palmyra-creative"]` | `"palmyra-x-004"` | Name of the LLM to invoke |

## Usage with ChatWriter

The `LLMTool` is designed to be used with the `ChatWriter` class:

```python
from langchain_writer import ChatWriter
from langchain_writer.tools import LLMTool

# Initialize the ChatWriter
llm = ChatWriter(
    model="palmyra-x-004",
    temperature=0.7,
    api_key="your-api-key"
)

# Create an LLMTool
llm_tool = LLMTool(model_name="palmyra-med")

# Bind the tool to the ChatWriter
llm_with_tools = llm.bind_tools([llm_tool])

# Now the model can delegate to the specialized LLM in its responses
response = llm_with_tools.invoke([
    ("system", "You are a helpful assistant with access to medical knowledge."),
    ("human", "Can you explain the symptoms of hypertension?")
])

# The LLM data is available in the response
print(response.additional_kwargs["llm_data"])
```

## Important notes

1. Due to its remote execution, the `LLMTool` does not support direct invocation through the `_run` method. Attempting to call this method will raise a `NotImplementedError`.

2. The `model_name` parameter is crucial as it specifies which Palmyra model the primary model can delegate to. Choose the appropriate model based on the specific domain or task requirements.

3. The `description` parameter is very important, as it provides context to the model using tool calling (e.g. Palmyra X 004 and later) which helps it understand the purpose of the tool.

4. When the model uses the LLM tool, the execution happens remotely on Writer's servers, and the response includes additional data in the `additional_kwargs["llm_data"]` field.

## Use cases

- Delegating medical questions to a specialized medical model (`palmyra-med`)
- Using a creative model for generating creative content (`palmyra-creative`)
- Leveraging a financial model for financial analysis (`palmyra-fin`)
# ChatWriter

`ChatWriter` is a LangChain integration for Writer's chat model API. It provides a seamless way to interact with Writer's powerful language models for chat-based applications.

**Example notebook**: [chat.ipynb](./chat.ipynb)

## Installation

```bash
pip install -U langchain-writer
```

You'll need to set up your Writer API key:

```bash
export WRITER_API_KEY="your-api-key"
```

## Basic usage

```python
from langchain_writer import ChatWriter

llm = ChatWriter(
    model="palmyra-x-004",  # default model
    temperature=0.7,        # controls randomness (0-1)
    max_tokens=None,        # maximum number of tokens to generate
    timeout=None,           # request timeout
    max_retries=2,          # number of retries on failure
    api_key="your-api-key"  # optional, will use env var if not provided
)
```

### Invoke (Synchronous)

```python
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]

response = llm.invoke(messages)
print(response.content)  # "J'aime la programmation."
```

### Stream (Synchronous)

```python
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Async support

```python
# Single invocation
response = await llm.ainvoke(messages)

# Streaming
async for chunk in await llm.astream(messages):
    print(chunk.content, end="", flush=True)

# Batch processing
responses = await llm.abatch([messages1, messages2, messages3])
```

## Tool calling

The `ChatWriter` supports function calling through tools:

```python
from langchain_writer.tools import GraphTool
from pydantic import BaseModel, Field

# Define a tool using Pydantic
class GetWeather(BaseModel):
    '''Get the current weather in a given location'''
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

# Create a GraphTool for knowledge graph access
graph_tool = GraphTool(graph_ids=['id1', 'id2'])

# Bind tools to the model
llm.bind_tools([graph_tool, GetWeather])

# Now the model can use these tools in its responses
response = llm.invoke([
    ("system", "You are a helpful assistant."),
    ("human", "What's the weather like in New York?")
])
```

> **Note**: Writer tools binding modifies the initial object instead of creating a new one with bound tools. Besides 'function' type, WriterChat supports 'graph' tool type via the `GraphTool` class.

## Response metadata

You can access metadata about the response:

```python
ai_msg = llm.invoke(messages)
metadata = ai_msg.response_metadata
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"palmyra-x-004"` | The Writer model to use |
| `temperature` | `float` | `0.7` | Controls randomness (0-1) |
| `model_kwargs` | `Dict[str, Any]` | `{}` | Additional model parameters |
| `n` | `int` | `1` | Number of completions to generate |
| `max_tokens` | `Optional[int]` | `None` | Maximum number of tokens to generate |
| `stop` | `Optional[Union[str, List[str]]]` | `None` | Sequences where the model should stop generating |
| `logprobs` | `bool` | `True` | Whether to return log probabilities |
| `tools` | `list[dict[str, Any]]` | `[]` | List of tools available to the model |
| `tool_choice` | `Union[Literal["none", "auto"], dict[str, Any]]` | `"auto"` | Which tool to use |

## Advanced features

### Custom message handling

The `ChatWriter` class provides utilities for converting between LangChain message formats and Writer's internal formats:

- `_convert_messages_to_dicts`: Converts LangChain messages to Writer dictionaries
- `_create_chat_result`: Creates a ChatResult from a Writer response

### Tool binding

The `bind_tools` method allows you to provide tools to the model that it can use to perform actions:

```python
llm.bind_tools(
    tools=[tool1, tool2],
    tool_choice="auto"  # or "none", or a specific tool name
)

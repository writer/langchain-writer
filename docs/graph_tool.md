# GraphTool

`GraphTool` is a specialized tool in the Writer LangChain integration that enables access to Writer's Knowledge Graph functionality. This tool allows language models to retrieve information from structured knowledge graphs to enhance their responses.

**Example notebook**: [tools.ipynb](./tools.ipynb)

## Overview

The `GraphTool` is designed specifically for use within the Writer environment and has a type of "graph" rather than the standard "function" type used by most LangChain tools. It's important to note that this tool does not support direct invocations and is meant to be used only within the Writer chat environment.

## Installation

```bash
pip install -U langchain-writer
```

## Initialization

```python
from langchain_writer.tools import GraphTool

# Create a GraphTool with specific graph IDs
graph_tool = GraphTool(
    graph_ids=["id1", "id2"],  # IDs of the knowledge graphs to query
    subqueries=False           # Whether to include subqueries in the response
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `Literal["graph"]` | `"graph"` | The tool type, always "graph" for GraphTool |
| `name` | `str` | `"Knowledge Graph"` | The name of the tool |
| `description` | `str` | `"Graph of files from which model can fetch data to compose response on user request."` | Description passed to the model |
| `graph_ids` | `list[str]` | Required | List of graph IDs to handle requests |
| `subqueries` | `bool` | `False` | Whether to include the subqueries used by Palmyra in the response |

## Usage with ChatWriter

The `GraphTool` is designed to be used with the `ChatWriter` class:

```python
from langchain_writer import ChatWriter
from langchain_writer.tools import GraphTool

# Initialize the ChatWriter
llm = ChatWriter(
    model="palmyra-x-004",
    temperature=0.7,
    api_key="your-api-key"
)

# Create a GraphTool
graph_tool = GraphTool(graph_ids=["id1", "id2"])

# Bind the tool to the ChatWriter
llm.bind_tools([graph_tool])

# Now the model can access the knowledge graph in its responses
response = llm.invoke([
    ("system", "You are a helpful assistant with access to a knowledge graph."),
    ("human", "Can you tell me about our company's Q2 financial results?")
])
```

## Important notes

1. The `GraphTool` does not support direct invocation through the `_run` method. Attempting to call this method will raise a `NotImplementedError`.

2. When binding a `GraphTool` to a `ChatWriter`, the original `ChatWriter` object is modified in place, rather than creating a new instance with the tools bound.

3. The `graph_ids` parameter is crucial as it specifies which knowledge graphs the model can access. These IDs should correspond to knowledge graphs that have been created in your Writer account.

4. The `subqueries` parameter controls whether the response includes the intermediate queries that Palmyra (Writer's model) used to retrieve information from the knowledge graph.

## Use cases

- Retrieving company-specific information from internal documents
- Accessing structured knowledge bases for domain-specific questions
- Enhancing responses with factual information from verified sources
- Creating chatbots that can reference internal knowledge bases

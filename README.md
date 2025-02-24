# langchain-writer

This package contains the official LangChain integrations for Writer through their `writer-sdk`.

## Installation and Setup

- Install the LangChain partner package:

```bash
pip install -U langchain-writer
```

- Sign up for [Writer AI Studio](https://app.writer.com/aistudio/signup?utm_campaign=devrel) and follow this [Quickstart](https://dev.writer.com/api-guides/quickstart) to obtain an API key.
- Set your Writer API key as an environment variable (`WRITER_API_KEY`).

## Chat capabilities

The `ChatWriter` class provides support of streaming and non-streaming chat completions, tool calls, batching, and asynchronous usage.

### Streaming (sync/async):
```python
from langchain_writer import ChatWriter

llm = ChatWriter()

# Sync chat call
generator = llm.stream("Sing a ballad of LangChain.")

for chunk in generator:
    print(chunk)

# Async chat call
generator = await llm.astream("Sing a ballad of LangChain.")

async for chunk in generator:
    print(chunk)
```

### Non-streaming (sync/async):

```python
from langchain_writer import ChatWriter

llm = ChatWriter()

# Sync chat call
llm.invoke("Sing a ballad of LangChain.")

# Async chat call
await llm.ainvoke("Sing a ballad of LangChain.")
```

### Batching (sync/async)

```python
from langchain_writer import ChatWriter

llm = ChatWriter()

llm.batch(
        [
            "How to cook pancakes?",
            "How to compose poem?",
            "How to run faster?",
        ],
        config={"max_concurrency": 2},
    )
```

### Tool binding

```python
from langchain_writer import ChatWriter
from langchain_core.tools import tool
from typing import Optional
from pydantic import BaseModel, Field


@tool
def get_supercopa_trophies_count(club_name: str) -> Optional[int]:
    """Returns information about supercopa trophies count.

    Args:
        club_name: Club you want to investigate info of supercopa trophies about

    Returns:
        Number of supercopa trophies or None if there is no info about requested club
    """
    # Tool implementation


class GetWeather(BaseModel):
    '''Get the current weather in a given location'''

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm = ChatWriter()

llm.bind_tools([get_supercopa_trophies_count, GetWeather])
```

## Additional resources
To learn more about LangChain, see the [official LangChain documentation](https://python.langchain.com/docs/introduction/). To learn more about Writer, see the [Writer developer documentation](https://dev.writer.com/home/introduction).

## About Writer
Writer is the full-stack generative AI platform for enterprises. Quickly and easily build and deploy generative AI apps with a suite of developer tools fully integrated with our platform of LLMs, graph-based RAG tools, AI guardrails, and more. Learn more at [writer.com](https://www.writer.com?utm_source=github&utm_medium=readme&utm_campaign=devrel).

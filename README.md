# langchain-writer

This package contains the LangChain integrations for Writer through their `writer-sdk`.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-writer
```

- Get a Writer api key and set it as an environment variable (`WRITER_API_KEY`)

## Chat capabilities

`Chat Writer` class provide support of streaming/nonstreaming chat completions, tool calls, batching and asynchronous usage:

### Streaming (sync/async):
```python
from langchain_writer import ChatWriter

llm = ChatWriter()

#Sync chat call
generator = llm.stream("Sing a ballad of LangChain.")

for chunk in generator:
    print(chunk)

#Async chat call
generator = await llm.astream("Sing a ballad of LangChain.")

async for chunk in generator:
    print(chunk)
```

### Non streaming (sync/async):

```python
from langchain_writer import ChatWriter

llm = ChatWriter()

#Sync chat call
llm.invoke("Sing a ballad of LangChain.")

#Async chat call
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

### Tools binding
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
    #Tool implementation


class GetWeather(BaseModel):
    '''Get the current weather in a given location'''

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

llm = ChatWriter()

llm_with_tools = llm.bind_tools(
    [get_supercopa_trophies_count, GetWeather]
)
```

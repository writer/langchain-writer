# langchain-writer

This package contains the LangChain integrations for Writer through their `writer-sdk`.

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-writer
```

- Get a Writer api key and set it as an environment variable (`WRITER_API_KEY`)

## Chat Models

`ChatWriter` class exposes chat models from Writer.

```python
from langchain_writer import ChatWriter

llm = ChatWriter()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`WriterEmbeddings` class exposes embeddings from Writer.

```python
from langchain_writer import WriterEmbeddings

embeddings = WriterEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`WriterLLM` class exposes LLMs from Writer.

```python
from langchain_writer import WriterLLM

llm = WriterLLM()
llm.invoke("The meaning of life is")
```
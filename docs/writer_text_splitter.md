# WriterTextSplitter

`WriterTextSplitter` is a LangChain integration for Writer's context-aware text splitting capabilities. It provides intelligent ways to split long texts into semantically meaningful chunks, which is essential for effective processing of large documents in LLM applications.

**Example notebook**: [text_splitter.ipynb](./text_splitter.ipynb)

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
from langchain_writer import WriterTextSplitter

splitter = WriterTextSplitter(
    strategy="llm_split",     # Splitting strategy to use
    api_key="your-api-key"    # Optional, will use env var if not provided
)

# Text to split
long_text = """
This is a very long document with multiple paragraphs and sections.
It contains information across various topics and needs to be split
into smaller, semantically meaningful chunks for processing.
...
"""

# Split the text into chunks
chunks = splitter.split_text(long_text)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("-" * 50)
```

## Async support

```python
# Asynchronously split text
chunks = await splitter.asplit_text(long_text)
```

## Splitting strategies

The `WriterTextSplitter` offers three different strategies for text splitting:

1. **`llm_split`** (default): Uses Writer's language model to split the text in a context-aware manner, preserving semantic meaning across chunks. This provides the highest quality splits but may be slower.

2. **`fast_split`**: Uses a fast heuristic-based approach for splitting. This is quicker but may not preserve semantic context as well as the LLM-based approach.

3. **`hybrid_split`**: Combines both strategies, offering a balance between speed and quality.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `Literal["llm_split", "fast_split", "hybrid_split"]` | `"llm_split"` | The strategy to use for text splitting |
| `api_key` | `str` | `None` | Writer API key (optional if set as environment variable) |

## Integration with LangChain

As a `TextSplitter` implementation, `WriterTextSplitter` can be used anywhere in the LangChain ecosystem that requires text splitting:

```python
from langchain_core.document_loaders import TextLoader
from langchain_core.document_transformers import RecursiveCharacterTextSplitter

# Load a document
loader = TextLoader("path/to/document.txt")
document = loader.load()[0]

# Use WriterTextSplitter instead of the default splitter
splitter = WriterTextSplitter(strategy="hybrid_split")
chunks = splitter.split_documents([document])

# Use the chunks with a vector store
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)
```

## How it works

The `WriterTextSplitter` works by:

1. Sending the text to Writer's context-aware splitting API
2. Applying the selected strategy (LLM-based, fast, or hybrid)
3. Receiving semantically meaningful chunks in response
4. Returning these chunks as a list of strings

## Use cases

- Preparing long documents for retrieval-augmented generation (RAG)
- Creating meaningful chunks for embedding and vector search
- Processing large texts that exceed context windows of language models
- Maintaining semantic coherence across document fragments

## Advantages over traditional splitters

Unlike traditional character or token-based splitters, `WriterTextSplitter` considers the semantic meaning of the text, resulting in:

- More coherent chunks that preserve context
- Better performance in downstream tasks like question answering
- Reduced redundancy across chunks
- Improved retrieval accuracy when used with vector stores

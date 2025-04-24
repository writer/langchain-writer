# LangChain Writer Integration Documentation

This documentation provides detailed information about the components available in the `langchain-writer` package, which integrates Writer's AI capabilities with the LangChain framework.

## Components

### [ChatWriter](./chat_writer.md)

The `ChatWriter` class provides integration with Writer's chat model API, allowing you to create conversational AI applications with LangChain. It supports both synchronous and asynchronous operations, streaming responses, and tool calling.

### [GraphTool](./graph_tool.md)

The `GraphTool` class enables access to Writer's Knowledge Graph functionality, allowing language models to retrieve information from structured knowledge graphs to enhance their responses.

### [LLMTool](./llm_tool.md)

The `LLMTool` class enables access to Writer's LLM tool type functionality, allowing sub invocation of other Palmyra models to enhance their responses.

### [NoCodeAppTool](./no_code_app_tool.md)

The `NoCodeAppTool` class enables access to Writer no-code applications as LLM tools.

### [PDFParser](./pdf_parser.md)

The `PDFParser` class provides integration with Writer's PDF parsing capabilities, allowing you to extract and process text content from PDF documents using Writer's advanced parsing technology.

### [WriterTextSplitter](./writer_text_splitter.md)

The `WriterTextSplitter` class provides integration with Writer's context-aware text splitting capabilities, allowing you to split long texts into semantically meaningful chunks for effective processing in LLM applications.

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
from langchain_writer import ChatWriter, PDFParser, WriterTextSplitter
from langchain_writer.tools import GraphTool, LLMTool, NoCodeAppTool
from langchain_core.documents.base import Blob

# Initialize components
llm = ChatWriter(model_name="palmyra-x5")
parser = PDFParser(format="markdown")
splitter = WriterTextSplitter(strategy="llm_split")
graph_tool = GraphTool(graph_ids=["id1", "id2"])
llm_tool = LLMTool(model_name="palmyra-creative")
no_code_app_tool = NoCodeAppTool(app_id="id")

# Use the chat model
response = llm.invoke([
    ("system", "You are a helpful assistant."),
    ("human", "Tell me about LangChain.")
])

# Parse a PDF
blob = Blob.from_path("path/to/document.pdf")
documents = parser.parse(blob)

# Split text
chunks = splitter.split_text("Long text to split...")

# Use the graph tool with the chat model
llm_with_tool = llm.bind_tools([graph_tool, llm_tool, no_code_app_tool])
response = llm_with_tool.invoke([
    ("system", "You are a helpful assistant with access to a knowledge graph."),
    ("human", "What information can you find about our company?")
])
```

## Advanced usage

For more detailed information about each component, please refer to their respective documentation pages.

## API reference

- [ChatWriter API](./chat_writer.md)
- [GraphTool API](./graph_tool.md)
- [LLMTool API](./llm_tool.md)
- [NoCodeAppTool API](./no_code_app_tool.md)
- [PDFParser API](./pdf_parser.md)
- [WriterTextSplitter API](./writer_text_splitter.md)

## Example notebooks

The following Jupyter notebooks provide interactive examples of how to use each component:

- [chat.ipynb](./chat.ipynb) - Examples of using the ChatWriter component
- [tools.ipynb](./tools.ipynb) - Examples of tools usage
- [pdf_parser.ipynb](./pdf_parser.ipynb) - Examples of using the PDFParser component
- [text_splitter.ipynb](./text_splitter.ipynb) - Examples of using the WriterTextSplitter component

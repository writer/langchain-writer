# PDFParser

`PDFParser` is a LangChain integration for Writer's PDF parsing capabilities. It provides a way to extract and process text content from PDF documents using Writer's advanced parsing technology.

**Example notebook**: [pdf_parser.ipynb](./pdf_parser.ipynb)

## Installation

```bash
pip install -U langchain-writer
```

You'll need to set up your Writer API key:

```bash
export WRITER_API_KEY="your-api-key"
```

## Basic usage

The `PDFParser` implements LangChain's `BaseBlobParser` interface, allowing it to be used with LangChain's document loading system:

```python
from langchain_writer import PDFParser
from langchain_core.documents.base import Blob

# Initialize the parser
parser = PDFParser(
    format="markdown",        # Output format: "text" or "markdown"
    api_key="your-api-key"    # Optional, will use env var if not provided
)

# Create a blob from a PDF file
blob = Blob.from_path("path/to/your/document.pdf")

# Parse the PDF into LangChain Document objects
documents = parser.parse(blob)

# Access the extracted content
for doc in documents:
    print(doc.page_content)
```

## Async support

`PDFParser` also supports asynchronous parsing:

```python
# Asynchronously parse a PDF
documents = await parser.aparse(blob)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `Literal["text", "markdown"]` | `"text"` | The format into which the PDF content should be converted |
| `api_key` | `str` | `None` | Writer API key (optional if set as environment variable) |

## How it works

The `PDFParser` works by:

1. Uploading the PDF file to Writer's servers
2. Using Writer's PDF parsing API to extract and format the content
3. Deleting the uploaded file after parsing is complete
4. Returning the parsed content as LangChain Document objects

## Implementation details

### File handling

The parser includes utility functions for managing PDF files:

- `upload_pdf_file`: Uploads a PDF file to Writer's servers
- `delete_file`: Deletes a file from Writer's servers after processing
- `generate_file_name`: Generates a unique filename based on the current timestamp

### Limitations

- Only supports files with MIME type `application/pdf`
- Does not support lazy parsing (the `lazy_parse` method raises `NotImplementedError`)

## Integration with LangChain

As a `BaseBlobParser` implementation, `PDFParser` can be used with LangChain's document loading system:

```python
from langchain_core.document_loaders import DirectoryLoader

# Create a loader that uses the PDFParser
loader = DirectoryLoader(
    "path/to/pdf/directory",
    glob="**/*.pdf",
    loader_cls=lambda file_path: PDFParser().parse(Blob.from_path(file_path))
)

# Load all PDFs in the directory
documents = loader.load()
```

## Error handling

The parser includes error handling for common issues:

- Raises `NotImplementedError` if the file is not a PDF
- Raises `ValueError` with a descriptive message if a file cannot be found for deletion
- Propagates Writer API errors with appropriate context

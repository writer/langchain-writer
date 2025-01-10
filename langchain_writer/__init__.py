from importlib import metadata

from langchain_writer.chat_models import ChatWriter
from langchain_writer.document_loaders import WriterLoader
from langchain_writer.embeddings import WriterEmbeddings
from langchain_writer.retrievers import WriterRetriever
from langchain_writer.toolkits import WriterToolkit
from langchain_writer.tools import WriterTool
from langchain_writer.vectorstores import WriterVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatWriter",
    "WriterVectorStore",
    "WriterEmbeddings",
    "WriterLoader",
    "WriterRetriever",
    "WriterToolkit",
    "WriterTool",
    "__version__",
]

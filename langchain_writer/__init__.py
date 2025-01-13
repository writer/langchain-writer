from importlib import metadata

from langchain_writer.chat_models import ChatWriter
from langchain_writer.document_loaders import WriterLoader
from langchain_writer.toolkits import WriterToolkit
from langchain_writer.tools import WriterTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatWriter",
    "WriterLoader",
    "WriterToolkit",
    "WriterTool",
    "__version__",
]

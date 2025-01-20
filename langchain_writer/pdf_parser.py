"""Writer pdf parser."""

from datetime import datetime
from typing import Iterator, Literal

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from pydantic import Field
from writerai import AsyncWriter, NotFoundError, Writer

from langchain_writer.base import BaseWriter


def upload_pdf_file(writer_client: Writer, file_data: bytes, file_name: str) -> str:
    uploaded_file = writer_client.files.upload(
        content=file_data,
        content_disposition=f"attachment; filename='{file_name}'",
        content_type="application/pdf",
    )
    return uploaded_file.id


def delete_file(writer_client: Writer, file_id: str) -> str:
    try:
        writer_client.files.delete(file_id)
        return file_id
    except NotFoundError as e:
        raise ValueError(f"Can't find file: {file_id}") from e


async def upload_pdf_file_async(
    writer_client: AsyncWriter, file_data: bytes, file_name: str
) -> str:
    uploaded_file = await writer_client.files.upload(
        content=file_data,
        content_disposition=f"attachment; filename='{file_name}'",
        content_type="application/pdf",
    )
    return uploaded_file.id


async def delete_file_async(writer_client: AsyncWriter, file_id: str) -> str:
    try:
        await writer_client.files.delete(file_id)
        return file_id
    except NotFoundError as e:
        raise ValueError(f"Can't find file: {file_id}") from e


def generate_file_name():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class PDFParser(BaseWriter, BaseBlobParser):
    """`Writer` PDF parser integration.

    Setup:
        Install ``langchain-writer`` and set environment variable ``WRITER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-writer

            export WRITER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_writer import PDFParser

            parser = PDFParser(
                format="markdown",
                api_key="..."
            )

    Invoke:
        .. code-block:: python

            blob = Blob.from_path("path/to/file.pdf")

            parsed_doc = parser.parse(file)

        .. code-block:: python

    """

    """The format into which the PDF content should be converted."""
    format: Literal["text", "markdown"] = Field(default="text")

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        raise NotImplementedError(
            "Writer does not support lazy parsing. Use 'parse' instead."
        )

    def parse(self, blob: Blob) -> list[Document]:
        if blob.mimetype != "application/pdf":
            raise NotImplementedError("PDF parser only supports 'application/pdf'")

        file_id = upload_pdf_file(self.client, blob.as_bytes(), generate_file_name())
        parsed_pdf = self.client.tools.parse_pdf(file_id=file_id, format=self.format)
        delete_file(self.client, file_id)
        return [Document(page_content=parsed_pdf.content)]

    async def aparse(self, blob: Blob) -> list[Document]:
        if blob.mimetype != "application/pdf":
            raise NotImplementedError("PDF parser only supports 'application/pdf'")

        file_id = await upload_pdf_file_async(
            self.async_client, blob.as_bytes(), generate_file_name()
        )
        parsed_pdf = await self.async_client.tools.parse_pdf(
            file_id=file_id, format=self.format
        )
        await delete_file_async(self.async_client, file_id)
        return [Document(page_content=parsed_pdf.content)]

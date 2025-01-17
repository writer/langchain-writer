"""Writer text splitter."""

from typing import List, Literal

from langchain_text_splitters import TextSplitter
from pydantic import Field

from langchain_writer.base import BaseWriter


class WriterTextSplitter(BaseWriter, TextSplitter):
    """`Writer` text splitter integration.

    Setup:
        Install ``langchain-writer`` and set environment variable ``WRITER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-writer

            export WRITER_API_KEY="your-api-key"

    Instantiate:
        .. code-block:: python

            from langchain_writer import ChatWriter

            splitter = WriterTextSplitter(
                strategy="llm_split",
                api_key="..."
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]

            llm.invoke(messages)

        .. code-block:: python

    """

    """The strategy to be used for splitting the text into chunks.
     llm_split uses the language model to split the text,
     fast_split uses a fast heuristic-based approach,
     hybrid_split combines both strategies."""
    strategy: Literal["llm_split", "fast_split", "hybrid_split"] = Field(
        default="llm_split"
    )

    def split_text(self, text: str) -> List[str]:
        response = self.client.tools.context_aware_splitting(
            strategy=self.strategy, text=text
        )
        return response.chunks

    async def asplit_text(self, text: str) -> List[str]:
        response = await self.async_client.tools.context_aware_splitting(
            strategy=self.strategy, text=text
        )
        return response.chunks

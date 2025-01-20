"""Integration tests Writer PDF parser wrapper

You need WRITER_API_KEY set in your environment
"""

import pytest


def test_pdf_parser_invocation(pdf_parser, pdf_file):
    parsed_docs = pdf_parser.parse(pdf_file)

    assert len(parsed_docs) == 1
    for doc in parsed_docs:
        assert len(doc.page_content) > 0


def test_pdf_parser_invocation_with_format(pdf_parser, pdf_file):
    pdf_parser.format = "markdown"
    parsed_docs = pdf_parser.parse(pdf_file)

    assert len(parsed_docs) == 1
    for doc in parsed_docs:
        assert len(doc.page_content) > 0


def test_pdf_parser_type_error(pdf_parser, text_file):
    with pytest.raises(NotImplementedError):
        pdf_parser.parse(text_file)


@pytest.mark.asyncio
async def test_pdf_parser_invocation_async(pdf_parser, pdf_file):
    parsed_docs = await pdf_parser.aparse(pdf_file)

    assert len(parsed_docs) == 1
    for doc in parsed_docs:
        assert len(doc.page_content) > 0


@pytest.mark.asyncio
async def test_pdf_parser_invocation_with_format_async(
    pdf_parser,
    pdf_file,
):
    pdf_parser.format = "markdown"
    parsed_docs = await pdf_parser.aparse(pdf_file)

    assert len(parsed_docs) == 1
    for doc in parsed_docs:
        assert len(doc.page_content) > 0


@pytest.mark.asyncio
async def test_pdf_parser_type_error_async(pdf_parser, text_file):
    with pytest.raises(NotImplementedError):
        await pdf_parser.aparse(text_file)

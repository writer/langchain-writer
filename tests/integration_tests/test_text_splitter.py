import pytest
from writerai import BadRequestError


def test_text_splitter_invocation(text_splitter, text_to_split):
    chunks = text_splitter.split_text(text_to_split)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk in text_to_split


def test_text_splitter_invocation_with_strategy(text_splitter, text_to_split):
    text_splitter.strategy = "hybrid_split"
    chunks = text_splitter.split_text(text_to_split)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.strip() in text_to_split


def test_text_splitter_invocation_oversize(text_splitter):
    oversize_text = "wrd " * 4001

    with pytest.raises(BadRequestError):
        text_splitter.split_text(oversize_text)


@pytest.mark.asyncio
async def test_text_splitter_invocation_async(text_splitter, text_to_split):
    chunks = await text_splitter.asplit_text(text_to_split)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.strip() in text_to_split


@pytest.mark.asyncio
async def test_text_splitter_invocation_with_strategy_async(
    text_splitter, text_to_split
):
    text_splitter.strategy = "hybrid_split"
    chunks = await text_splitter.asplit_text(text_to_split)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert chunk.strip() in text_to_split


@pytest.mark.asyncio
async def test_text_splitter_invocation_oversize_async(text_splitter):
    oversize_text = "wrd " * 4001

    with pytest.raises(BadRequestError):
        await text_splitter.asplit_text(oversize_text)

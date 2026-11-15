import io

import pytest

from pdf_processor import PDFProcessor
from tests.helpers import build_report_pdf


@pytest.fixture
def processor() -> PDFProcessor:
    return PDFProcessor()


@pytest.fixture
def report_pdf() -> io.BytesIO:
    return io.BytesIO(build_report_pdf(pages=2))


def test_extract_text_roundtrip(processor: PDFProcessor, report_pdf: io.BytesIO) -> None:
    text = processor.extract_text_from_pdf(report_pdf)
    assert "controlled experiments" in text
    assert len(text) > 500


def test_extract_text_from_corrupt_file(processor: PDFProcessor) -> None:
    with pytest.raises(RuntimeError, match="Error reading PDF"):
        processor.extract_text_from_pdf(io.BytesIO(b"not a pdf"))


def test_metadata(processor: PDFProcessor, report_pdf: io.BytesIO) -> None:
    metadata = processor.get_pdf_metadata(report_pdf)
    assert metadata["num_pages"] == 2
    assert metadata["title"] == "Unknown"


def test_clean_text(processor: PDFProcessor) -> None:
    raw = "alpha\t\tbeta  \n\n\n\n gamma\x00 delta "
    assert processor.clean_text(raw) == "alpha beta\n\ngamma delta"


def test_count_tokens(processor: PDFProcessor) -> None:
    assert processor.count_tokens("") == 0
    assert processor.count_tokens("hello world") > 0


def test_chunk_respects_token_limit(processor: PDFProcessor) -> None:
    text = "This is one sentence. " * 60
    chunks = processor.chunk_text(text, max_tokens=30)
    assert len(chunks) > 1
    assert all(processor.count_tokens(chunk) <= 30 for chunk in chunks)


def test_chunk_hard_splits_oversized_sentence(processor: PDFProcessor) -> None:
    text = "word " * 500
    chunks = processor.chunk_text(text, max_tokens=50)
    assert len(chunks) > 1
    assert all(processor.count_tokens(chunk) <= 50 for chunk in chunks)


def test_chunk_keeps_short_text_whole(processor: PDFProcessor) -> None:
    assert processor.chunk_text("A short document.", max_tokens=100) == ["A short document."]


def test_chunk_rejects_empty_text(processor: PDFProcessor) -> None:
    with pytest.raises(ValueError, match="No extractable text"):
        processor.chunk_text("   \n  ")


def test_chunk_rejects_bad_limit(processor: PDFProcessor) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        processor.chunk_text("hello", max_tokens=0)

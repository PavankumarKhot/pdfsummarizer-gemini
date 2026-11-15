"""Opt-in tests that hit the real Gemini API. Run with RUN_LIVE_TESTS=1."""

import os

import pytest

from summarizer import PDFSummarizer

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call the real Gemini API",
)


@pytest.mark.live
def test_live_summary() -> None:
    summarizer = PDFSummarizer(max_concurrency=1)
    summary = summarizer.summarize_text(
        "The cat sat on the mat. It was a sunny day in the small village.",
        "brief",
    )
    assert summary.strip()

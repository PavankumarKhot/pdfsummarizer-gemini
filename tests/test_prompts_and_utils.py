import pytest

from prompts import CHUNK_SUMMARY, COMBINE, STYLES, SUMMARY_TYPES
from utils import (
    api_key_error,
    create_summary_dataframe,
    describe_api_error,
    estimate_processing_time,
    export_summary_to_text,
)


def test_summary_types_match_styles() -> None:
    assert SUMMARY_TYPES == list(STYLES)
    assert set(SUMMARY_TYPES) == {"brief", "detailed", "bullet_points", "executive"}


def test_chunk_prompt_renders_style_and_text() -> None:
    messages = CHUNK_SUMMARY.format_messages(style=STYLES["brief"], text="hello world")
    assert messages[0].content.startswith("You are a precise document summarizer")
    assert "hello world" in messages[1].content
    assert STYLES["brief"] in messages[1].content


def test_combine_prompt_renders_summaries() -> None:
    messages = COMBINE.format_messages(style="short", summaries="Section 1: a\nSection 2: b")
    assert "Section 1: a" in messages[1].content


def test_api_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not found" in api_key_error()

    monkeypatch.setenv("GEMINI_API_KEY", "short")
    assert "incomplete" in api_key_error()

    monkeypatch.setenv("GEMINI_API_KEY", "a" * 40)
    assert api_key_error() is None


def test_describe_api_error_variants() -> None:
    assert "rate limit" in describe_api_error(RuntimeError("429 RESOURCE_EXHAUSTED")).lower()
    assert "403" in describe_api_error(RuntimeError("403 PERMISSION_DENIED"))
    assert "not available" in describe_api_error(RuntimeError("404 NOT_FOUND"))
    assert "rejected the API key" in describe_api_error(RuntimeError("API key not valid"))
    assert "boom" in describe_api_error(RuntimeError("boom"))


def _summary_data() -> dict:
    return {
        "summary_type": "detailed",
        "total_chunks": 2,
        "combined_summary": "Final text.",
        "individual_summaries": [
            {
                "chunk_number": 1,
                "summary": "One.",
                "ok": True,
                "original_length": 100,
                "summary_length": 25,
            },
            {
                "chunk_number": 2,
                "summary": "Two.",
                "ok": False,
                "original_length": 0,
                "summary_length": 0,
            },
        ],
    }


def test_dataframe_statistics() -> None:
    dataframe = create_summary_dataframe(_summary_data())
    assert list(dataframe.columns) == [
        "Chunk",
        "Original Length",
        "Summary Length",
        "Compression Ratio",
        "Status",
    ]
    assert dataframe.loc[0, "Compression Ratio"] == 0.25
    assert dataframe.loc[1, "Compression Ratio"] == 0
    assert dataframe.loc[1, "Status"] == "Error"


def test_export_text_report() -> None:
    report = export_summary_to_text(_summary_data(), "doc.pdf")
    assert "File: doc.pdf" in report
    assert "FINAL SUMMARY" in report
    assert "Final text." in report
    assert "Chunk 2 (FAILED):" in report


def test_estimate_processing_time_increases_with_chunks() -> None:
    assert estimate_processing_time(1) == "~5 seconds"
    assert estimate_processing_time(100).endswith("s")
    assert "min" in estimate_processing_time(1000)

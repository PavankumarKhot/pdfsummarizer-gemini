import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from summarizer import PDFSummarizer, message_text


def make_summarizer(fn, **kwargs) -> PDFSummarizer:
    return PDFSummarizer(llm=RunnableLambda(fn), **kwargs)


def fake_llm(content: str = "summary"):
    return lambda _value: AIMessage(content=content)


def test_summarize_text() -> None:
    summarizer = make_summarizer(fake_llm("  short summary  "))
    assert summarizer.summarize_text("some text", "brief") == "short summary"


def test_summarize_chunks_map_reduce() -> None:
    summarizer = make_summarizer(fake_llm("part"), max_concurrency=2)
    result = summarizer.summarize_chunks(["first", "second"], "brief")

    assert result["total_chunks"] == 2
    assert [s["ok"] for s in result["individual_summaries"]] == [True, True]
    assert result["combined_summary"] == "part"
    assert result["summary_type"] == "brief"


def test_single_chunk_skips_combine() -> None:
    summarizer = make_summarizer(fake_llm("only"), max_concurrency=1)
    result = summarizer.summarize_chunks(["one chunk"], "detailed")
    assert result["combined_summary"] == "only"


def test_chunk_failures_are_captured() -> None:
    def flaky(value) -> AIMessage:
        if "BAD" in value.to_string():
            raise RuntimeError("boom")
        return AIMessage(content="ok")

    summarizer = make_summarizer(flaky, max_concurrency=1)
    result = summarizer.summarize_chunks(["good text", "BAD text"], "brief")

    statuses = [s["ok"] for s in result["individual_summaries"]]
    assert statuses == [True, False]
    assert result["combined_summary"] == "ok"


def test_all_chunks_failing_returns_message() -> None:
    def always_fails(_value) -> AIMessage:
        raise RuntimeError("down")

    summarizer = make_summarizer(always_fails, max_concurrency=1)
    result = summarizer.summarize_chunks(["a", "b"], "brief")
    assert all(not s["ok"] for s in result["individual_summaries"])
    assert result["combined_summary"] == "No valid summaries were generated."


def test_analysis_failure_is_reported() -> None:
    def always_fails(_value) -> AIMessage:
        raise RuntimeError("no access")

    result = make_summarizer(always_fails).analyze_document_structure("text")
    assert result["status"] == "error"
    assert "no access" in result["analysis"]


def test_quotes_are_cleaned() -> None:
    content = '1. "Alpha quote"\n- Beta quote\n\n\u2022 Gamma quote'
    summarizer = make_summarizer(fake_llm(content))
    assert summarizer.extract_key_quotes("text") == ["Alpha quote", "Beta quote", "Gamma quote"]


def test_quotes_are_capped_at_ten() -> None:
    summarizer = make_summarizer(fake_llm("\n".join(f"quote {i}" for i in range(15))))
    assert len(summarizer.extract_key_quotes("text")) == 10


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        PDFSummarizer()


def test_message_text_handles_block_content() -> None:
    class BlockMessage:
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
            {"type": "image", "data": "ignored"},
        ]

    assert message_text(BlockMessage()) == "first\nsecond"


def test_message_text_handles_plain_string() -> None:
    assert message_text("  plain  ") == "plain"

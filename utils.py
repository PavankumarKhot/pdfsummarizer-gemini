"""Display, export, statistics and API-key helpers (no UI imports at module level)."""

import os
import time

import pandas as pd


def get_api_key() -> str | None:
    """Return the configured Gemini API key, if any."""
    return os.getenv("GEMINI_API_KEY") or None


def api_key_error() -> str | None:
    """Return a user-facing problem description, or None when the key looks usable."""
    api_key = get_api_key()
    if not api_key:
        return (
            "GEMINI_API_KEY not found. Copy .env.example to .env, add your key, "
            "and restart. Get a free key at https://aistudio.google.com/apikey"
        )
    if len(api_key) < 20:
        return (
            "GEMINI_API_KEY seems incomplete. Get a new one at "
            "https://aistudio.google.com/apikey"
        )
    return None


def describe_api_error(exc: Exception) -> str:
    """Turn a Gemini/LangChain exception into an actionable message."""
    text = str(exc)
    lowered = text.lower()
    if "429" in text or "resource_exhausted" in lowered or "rate limit" in lowered:
        return (
            "Gemini rate limit reached (429). Wait a minute, lower MAX_CONCURRENCY, "
            "use a smaller PDF, or switch to a lighter model."
        )
    if "403" in text or "permission_denied" in lowered:
        return (
            "Gemini rejected the request (403). Check that the API key is enabled for the "
            "Generative Language API."
        )
    if "404" in text or "not_found" in lowered:
        return (
            "The selected model is not available for this key (404). Pick another model "
            "in the sidebar or set GEMINI_MODEL in .env."
        )
    if "api key not valid" in lowered or "api_key_invalid" in lowered:
        return "Gemini rejected the API key. Generate a new key and update .env."
    return f"Gemini API error: {text}"


def format_summary_display(summary_data: dict) -> None:
    """Render summary statistics, final summary and per-chunk summaries in Streamlit."""
    import streamlit as st

    st.subheader("\U0001f4ca Summary Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data["total_chunks"])
    col2.metric(
        "Successful Summaries", sum(s["ok"] for s in summary_data["individual_summaries"])
    )
    col3.metric("Summary Type", summary_data["summary_type"].replace("_", " ").title())

    st.subheader("\U0001f4dd Final Summary")
    st.write(summary_data["combined_summary"])

    with st.expander("View Individual Chunk Summaries"):
        for s in summary_data["individual_summaries"]:
            st.write(f"**Chunk {s['chunk_number']}**" + ("" if s["ok"] else " \u26a0 failed"))
            st.write(s["summary"])
            st.write(
                f"*Original length: {s['original_length']} chars, "
                f"Summary length: {s['summary_length']} chars*"
            )
            st.divider()


def export_summary_to_text(summary_data: dict, filename: str = "") -> str:
    """Build the plain-text report; filename (if given) appears in the header."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data["combined_summary"],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for s in summary_data["individual_summaries"]:
        lines += [f"Chunk {s['chunk_number']}{'' if s['ok'] else ' (FAILED)'}:", s["summary"], ""]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    return pd.DataFrame(
        [
            {
                "Chunk": s["chunk_number"],
                "Original Length": s["original_length"],
                "Summary Length": s["summary_length"],
                "Compression Ratio": (
                    round(s["summary_length"] / s["original_length"], 2)
                    if s["original_length"]
                    else 0
                ),
                "Status": "Success" if s["ok"] else "Error",
            }
            for s in summary_data["individual_summaries"]
        ]
    )


def estimate_processing_time(num_chunks: int, concurrency: int = 4) -> str:
    """Rough estimate: ~3s per chunk with `concurrency` running at once, plus 5s to combine."""
    seconds = int(num_chunks * 3 / max(concurrency, 1) + 5)
    return f"~{seconds} seconds" if seconds < 60 else f"~{seconds // 60} min {seconds % 60}s"

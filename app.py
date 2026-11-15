"""Streamlit UI for the PDF AI Summarizer."""

import streamlit as st

from config import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MODEL,
    MAX_CHUNK_TOKENS,
    MAX_CONCURRENCY,
    MIN_CHUNK_TOKENS,
    MODELS,
)
from pdf_processor import PDFProcessor
from prompts import SUMMARY_TYPES
from summarizer import PDFSummarizer
from utils import (
    api_key_error,
    create_summary_dataframe,
    describe_api_error,
    estimate_processing_time,
    export_summary_to_text,
    format_summary_display,
)

st.set_page_config(page_title="PDF AI Summarizer", page_icon="\U0001f4c4", layout="wide")


@st.cache_resource
def get_processor() -> PDFProcessor:
    return PDFProcessor()


@st.cache_resource
def get_summarizer(model: str) -> PDFSummarizer:
    return PDFSummarizer(model=model, max_concurrency=MAX_CONCURRENCY)


def main() -> None:
    st.title("\U0001f4c4 PDF Reader + AI Summarizer")
    st.markdown(
        "Upload a PDF document and get intelligent summaries powered by Google Gemini."
    )

    error = api_key_error()
    if error:
        st.error(f"\u274c {error}")
        st.stop()
    st.sidebar.caption("\U0001f511 Gemini API key loaded")

    st.sidebar.header("\u2699 Settings")
    models = MODELS if DEFAULT_MODEL in MODELS else [DEFAULT_MODEL, *MODELS]
    model = st.sidebar.selectbox(
        "Model",
        models,
        index=models.index(DEFAULT_MODEL),
        help="Gemini model used for all LLM calls",
    )

    summary_type = st.sidebar.selectbox(
        "Summary Type",
        SUMMARY_TYPES,
        index=SUMMARY_TYPES.index("detailed"),
        help="Choose the type of summary you want",
    )

    max_tokens = st.sidebar.slider(
        "Max Tokens per Chunk",
        MIN_CHUNK_TOKENS,
        MAX_CHUNK_TOKENS,
        DEFAULT_CHUNK_TOKENS,
        step=500,
        help="Maximum tokens per text chunk (affects processing speed and cost)",
    )

    show_analysis = st.sidebar.checkbox(
        "Show Document Analysis", value=True, help="Analyze document structure and content"
    )
    show_quotes = st.sidebar.checkbox(
        "Extract Key Quotes", value=False, help="Extract important quotes from the document"
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", help="Upload a PDF document to summarize"
    )

    # Drop stale results when the file or any processing setting changes.
    file_id = uploaded_file.file_id if uploaded_file else None
    signature = (file_id, summary_type, max_tokens, show_analysis, show_quotes, model)
    if st.session_state.get("signature") != signature:
        st.session_state.pop("results", None)
        st.session_state.signature = signature

    if uploaded_file is None:
        st.info("Upload a text-based PDF to get started. Scanned image PDFs need OCR first.")
        return

    st.success(f"\u2705 File uploaded: {uploaded_file.name}")
    processor = get_processor()

    with st.spinner("Analyzing PDF metadata..."):
        metadata = processor.get_pdf_metadata(uploaded_file)

    if "error" in metadata:
        st.warning(f"Could not read PDF metadata: {metadata['error']}")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"\U0001f4c4 Pages: {metadata.get('num_pages', 'Unknown')}")
        st.info(f"\U0001f4dd Title: {metadata.get('title', 'Unknown')}")
    with col2:
        st.info(f"\U0001f464 Author: {metadata.get('author', 'Unknown')}")
        st.info(f"\U0001f4cb Subject: {metadata.get('subject', 'Unknown')}")

    if st.button("\U0001f680 Process PDF", type="primary"):
        process_pdf(
            uploaded_file,
            processor,
            model,
            summary_type,
            max_tokens,
            show_analysis,
            show_quotes,
        )

    if "results" in st.session_state:
        display_results(st.session_state.results)


def process_pdf(
    uploaded_file,
    processor: PDFProcessor,
    model: str,
    summary_type: str,
    max_tokens: int,
    show_analysis: bool,
    show_quotes: bool,
) -> None:
    """Run the pipeline and store the results in st.session_state['results']."""
    st.session_state.pop("results", None)

    with st.spinner("\U0001f4d6 Extracting text from PDF..."):
        try:
            raw_text = processor.extract_text_from_pdf(uploaded_file)
            st.success(f"\u2705 Extracted {len(raw_text)} characters")
        except Exception as e:  # noqa: BLE001
            st.error(f"\u274c Error extracting text: {e}")
            return

    with st.spinner("\U0001f9f9 Cleaning and processing text..."):
        clean_text = processor.clean_text(raw_text)
        st.success(
            f"\u2705 Cleaned text: {len(clean_text)} characters, "
            f"~{processor.count_tokens(clean_text)} tokens"
        )

    with st.spinner("\u2702 Splitting text into chunks..."):
        try:
            chunks = processor.chunk_text(clean_text, max_tokens)
        except ValueError as e:
            st.error(f"\u274c {e}")
            return
        st.success(f"\u2705 Created {len(chunks)} chunks")
        st.info(f"\u23f1 Estimated processing time: {estimate_processing_time(len(chunks))}")

    try:
        summarizer = get_summarizer(model)
    except Exception as e:  # noqa: BLE001
        st.error(f"\u274c Could not initialise the model: {describe_api_error(e)}")
        return

    analysis = None
    if show_analysis:
        with st.spinner("\U0001f50d Analyzing document structure..."):
            analysis = summarizer.analyze_document_structure(clean_text)

    with st.spinner(f"\U0001f916 Generating {summary_type} summaries..."):
        try:
            summary_data = summarizer.summarize_chunks(chunks, summary_type)
        except Exception as e:  # noqa: BLE001
            st.error(f"\u274c Error generating summaries: {describe_api_error(e)}")
            return

    failed = [s for s in summary_data["individual_summaries"] if not s["ok"]]
    if failed:
        st.warning(
            f"\u26a0 {len(failed)} of {summary_data['total_chunks']} chunk(s) failed. "
            + describe_api_error(Exception(failed[0]["summary"]))
        )

    quotes = None
    if show_quotes:
        with st.spinner("\U0001f4ac Extracting key quotes..."):
            try:
                quotes = summarizer.extract_key_quotes(clean_text[:5000])
            except Exception as e:  # noqa: BLE001
                st.warning(f"\u26a0 Could not extract key quotes: {describe_api_error(e)}")

    st.session_state.results = {
        "file_name": uploaded_file.name,
        "summary_data": summary_data,
        "analysis": analysis,
        "quotes": quotes,
    }


def display_results(results: dict) -> None:
    """Render stored results: analysis, quotes, summary, downloads and stats."""
    summary_data = results["summary_data"]

    analysis = results.get("analysis")
    if analysis:
        if analysis["status"] == "success":
            st.subheader("\U0001f4ca Document Analysis")
            st.write(analysis["analysis"])
        else:
            st.warning("\u26a0 Could not analyze document structure")

    if results.get("quotes"):
        st.subheader("\U0001f4ac Key Quotes")
        for i, quote in enumerate(results["quotes"], 1):
            st.write(f'{i}. *"{quote}"*')

    st.header("\U0001f4cb Summary Results")
    format_summary_display(summary_data)

    st.subheader("\U0001f4e4 Export Options")
    summary_df = create_summary_dataframe(summary_data)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="\U0001f4c4 Download as Text",
            data=export_summary_to_text(summary_data, results["file_name"]),
            file_name=f"{results['file_name']}_summary.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            label="\U0001f4ca Download Statistics as CSV",
            data=summary_df.to_csv(index=False),
            file_name=f"{results['file_name']}_stats.csv",
            mime="text/csv",
        )

    with st.expander("\U0001f4c8 Processing Statistics"):
        st.dataframe(summary_df)
        if not summary_df.empty:
            st.metric(
                "Average Compression Ratio",
                f"{summary_df['Compression Ratio'].mean():.2f}",
            )
            st.metric(
                "Success Rate",
                f"{(summary_df['Status'] == 'Success').mean() * 100:.1f}%",
            )


if __name__ == "__main__":
    main()

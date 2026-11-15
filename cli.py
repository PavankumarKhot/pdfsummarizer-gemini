"""Command-line entry point: summarize a PDF without the Streamlit UI.

Usage:
    python cli.py path/to/document.pdf --style detailed
    python cli.py document.pdf --analysis --quotes --out summary.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import DEFAULT_CHUNK_TOKENS, DEFAULT_MODEL, MAX_CONCURRENCY, MODELS
from pdf_processor import PDFProcessor
from prompts import SUMMARY_TYPES
from summarizer import PDFSummarizer
from utils import api_key_error, describe_api_error, export_summary_to_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a PDF with Google Gemini.")
    parser.add_argument("pdf", type=Path, help="Path to a text-based PDF file")
    parser.add_argument("--style", default="detailed", choices=SUMMARY_TYPES)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--analysis", action="store_true", help="Also analyze document structure")
    parser.add_argument("--quotes", action="store_true", help="Also extract key quotes")
    parser.add_argument("--out", type=Path, help="Write the text report to this path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    error = api_key_error()
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not args.pdf.is_file():
        print(f"error: PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    processor = PDFProcessor()

    with args.pdf.open("rb") as handle:
        metadata = processor.get_pdf_metadata(handle)
        try:
            raw_text = processor.extract_text_from_pdf(handle)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    if not raw_text.strip():
        print("error: no extractable text - the PDF may be scanned images", file=sys.stderr)
        return 1

    clean_text = processor.clean_text(raw_text)
    try:
        chunks = processor.chunk_text(clean_text, args.max_tokens)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"File: {args.pdf.name}")
    print(f"Pages: {metadata.get('num_pages', 'Unknown')}")
    print(f"Tokens: ~{processor.count_tokens(clean_text)}")
    print(f"Chunks: {len(chunks)} (max {args.max_tokens} tokens each)")
    print(f"Model: {args.model} | Style: {args.style}")

    summarizer = PDFSummarizer(model=args.model, max_concurrency=args.concurrency)

    analysis = None
    if args.analysis:
        print("Analyzing document structure...", file=sys.stderr)
        analysis = summarizer.analyze_document_structure(clean_text)

    print(f"Summarizing {len(chunks)} chunk(s)...", file=sys.stderr)
    try:
        summary_data = summarizer.summarize_chunks(chunks, args.style)
    except Exception as e:  # noqa: BLE001
        print(f"error: {describe_api_error(e)}", file=sys.stderr)
        return 1

    quotes = None
    if args.quotes:
        print("Extracting key quotes...", file=sys.stderr)
        try:
            quotes = summarizer.extract_key_quotes(clean_text[:5000])
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not extract quotes: {describe_api_error(e)}", file=sys.stderr)

    failed = [s for s in summary_data["individual_summaries"] if not s["ok"]]
    if failed and len(failed) == summary_data["total_chunks"]:
        print(
            "error: every chunk failed. " + describe_api_error(Exception(failed[0]["summary"])),
            file=sys.stderr,
        )
        return 1

    if args.json:
        payload = {
            "file": args.pdf.name,
            "metadata": metadata,
            "model": args.model,
            "analysis": analysis,
            "quotes": quotes,
            "summary": summary_data,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if analysis and analysis["status"] == "success":
        print("\n=== Document Analysis ===")
        print(analysis["analysis"])

    if quotes:
        print("\n=== Key Quotes ===")
        for i, quote in enumerate(quotes, 1):
            print(f'{i}. "{quote}"')

    print("\n=== Summary ===")
    print(summary_data["combined_summary"])
    print("\n=== Chunk Statistics ===")
    for s in summary_data["individual_summaries"]:
        status = "ok" if s["ok"] else "FAILED"
        print(
            f"chunk {s['chunk_number']}: {status}, "
            f"{s['original_length']} -> {s['summary_length']} chars"
        )

    if args.out:
        report = export_summary_to_text(summary_data, args.pdf.name)
        args.out.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

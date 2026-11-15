"""Gemini calls via LangChain: chunk summaries, combine, analysis and quotes."""

import logging
import os
import re

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from config import DEFAULT_MODEL, MAX_CONCURRENCY
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

logger = logging.getLogger(__name__)


def message_text(message) -> str:
    """Extract plain text from a Gemini response across langchain-google-genai versions."""
    if isinstance(message, str):
        return message.strip()

    text = getattr(message, "text", None)
    if text is None:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            blocks: list[str] = []
            for block in content:
                if isinstance(block, str):
                    blocks.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    blocks.append(str(block.get("text", "")))
            text = "\n".join(blocks)
        else:
            text = content
    return str(text).strip()


class PDFSummarizer:
    def __init__(self, model: str | None = None, llm=None, max_concurrency: int | None = None):
        self.model = model or DEFAULT_MODEL
        self.max_concurrency = max_concurrency or MAX_CONCURRENCY
        if llm is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0.3,
                google_api_key=api_key,
                max_retries=4,
            )
        self.llm = llm

        parser = RunnableLambda(message_text)
        self.chunk_chain = CHUNK_SUMMARY | self.llm | parser
        self.combine_chain = COMBINE | self.llm | parser
        self.analysis_chain = ANALYSIS | self.llm | parser
        self.quotes_chain = QUOTES | self.llm | parser

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        """Summarize a single block of text (one chunk)."""
        style = STYLES.get(summary_type, STYLES["detailed"])
        return self.chunk_chain.invoke({"style": style, "text": text})

    def summarize_chunks(self, chunks: list[str], summary_type: str = "detailed") -> dict:
        """Map-summarize every chunk, then reduce the results into one summary."""
        style = STYLES.get(summary_type, STYLES["detailed"])
        inputs = [{"style": style, "text": chunk} for chunk in chunks]
        results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        summaries = []
        for i, (chunk, result) in enumerate(zip(chunks, results, strict=True), start=1):
            ok = not isinstance(result, Exception)
            if not ok:
                logger.warning("Chunk %d failed: %s", i, result)
            summary = result if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": i,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        """Reduce per-chunk summaries into a single summary."""
        valid = [cs["summary"] for cs in chunk_summaries if cs.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(f"Section {i}: {s}" for i, s in enumerate(valid, start=1))
        style = STYLES.get(summary_type, STYLES["detailed"])
        try:
            return self.combine_chain.invoke({"style": style, "summaries": sections})
        except Exception as e:  # noqa: BLE001
            logger.warning("Combine step failed, returning section summaries: %s", e)
            return f"Combined Summary:\n\n{sections}"

    def analyze_document_structure(self, text: str) -> dict:
        """Return a short structural analysis of the first part of the document."""
        try:
            analysis = self.analysis_chain.invoke({"text": text[:3000]})
            return {"analysis": analysis, "status": "success"}
        except Exception as e:  # noqa: BLE001
            logger.warning("Document analysis failed: %s", e)
            return {"analysis": f"Error analyzing document: {e}", "status": "error"}

    def extract_key_quotes(self, text: str) -> list[str]:
        """Extract up to 10 verbatim quotes from the first part of the document."""
        response = self.quotes_chain.invoke({"text": text})
        lines = (
            re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s*", "", line).strip(' "\u201c\u201d')
            for line in response.splitlines()
        )
        return [line for line in lines if line][:10]

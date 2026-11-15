"""Test helpers: build small, valid PDFs without external dependencies."""

import io


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]], font_size: int = 11, leading: int = 14) -> bytes:
    """Return PDF bytes for the given pages of text lines."""
    if not pages:
        raise ValueError("build_pdf needs at least one page")

    n_pages = len(pages)
    font_obj = 3 + 2 * n_pages
    objects: dict[int, str] = {}
    kids: list[str] = []

    for index, lines in enumerate(pages):
        page_obj = 3 + 2 * index
        content_obj = page_obj + 1
        kids.append(f"{page_obj} 0 R")

        parts = ["BT", f"/F1 {font_size} Tf", f"{leading} TL", "72 720 Td"]
        for line_index, line in enumerate(lines):
            if line_index:
                parts.append("T*")
            parts.append(f"({_escape(line)}) Tj")
        parts.append("ET")
        stream = "\n".join(parts)

        objects[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        )
        objects[content_obj] = (
            f"<< /Length {len(stream.encode('latin-1'))} >>\n"
            f"stream\n{stream}\nendstream"
        )

    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {n_pages} >>"
    objects[font_obj] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = out.tell()
        out.write(f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1"))

    xref_offset = out.tell()
    size = max(objects) + 1
    out.write(f"xref\n0 {size}\n".encode("latin-1"))
    out.write(b"0000000000 65535 f \n")
    for number in range(1, size):
        out.write(f"{offsets.get(number, 0):010d} 00000 n \n".encode("latin-1"))
    out.write(
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return out.getvalue()


def wrap_lines(text: str, width: int = 88) -> list[str]:
    """Greedy word wrap so lines fit a page."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_report_pdf(pages: int = 2, paragraphs_per_page: int = 4) -> bytes:
    """A deterministic multi-page report used by tests and the sample generator."""
    sentence_templates = [
        "The analysis evaluates how document structure affects comprehension and recall.",
        "Each section presents evidence collected from controlled experiments and surveys.",
        "Results show a consistent improvement when summaries preserve the original ordering.",
        "The authors note that longer inputs require careful chunk boundaries and overlap.",
        "Limitations include a small sample, a single language, and a narrow domain.",
        "Future work should test additional models and measure factual consistency directly.",
        "Table {n} reports the compression ratio for every configuration in the study.",
        "The discussion connects these findings to prior work on automatic summarization.",
    ]
    page_texts: list[list[str]] = []
    counter = 0
    for _ in range(pages):
        lines: list[str] = []
        for _ in range(paragraphs_per_page):
            sentences = []
            for template in sentence_templates:
                counter += 1
                sentences.append(template.format(n=counter))
            lines.extend(wrap_lines(" ".join(sentences)))
            lines.append("")
        page_texts.append(lines)
    return build_pdf(page_texts)

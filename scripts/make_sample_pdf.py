"""Generate samples/sample.pdf for a headless end-to-end run."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.helpers import build_report_pdf  # noqa: E402


def main() -> None:
    samples = ROOT / "samples"
    samples.mkdir(exist_ok=True)
    target = samples / "sample.pdf"
    target.write_bytes(build_report_pdf(pages=12, paragraphs_per_page=6))
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

"""Shared configuration: models, chunking bounds and concurrency.

Values can be overridden with environment variables (loaded from `.env`).
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Models verified against the Google AI Studio model list for this project.
MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

DEFAULT_CHUNK_TOKENS = 8000
MIN_CHUNK_TOKENS = 2000
MAX_CHUNK_TOKENS = 12000

# How many chunk summaries run in parallel. Free-tier keys may need a lower value.
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))

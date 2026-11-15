"""UI smoke tests using Streamlit's AppTest harness (no browser needed)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_boots_without_exception() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value.startswith("\U0001f4c4")
    assert len(app.sidebar.selectbox) == 2
    assert app.sidebar.selectbox[0].label == "Model"
    assert app.sidebar.selectbox[1].label == "Summary Type"
    assert app.sidebar.slider[0].label == "Max Tokens per Chunk"


def test_api_key_error_stops_app(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(str(APP), default_timeout=30).run()

    assert not app.exception
    assert app.error[0].icon == "\u274c"
    assert "GEMINI_API_KEY" in app.error[0].value

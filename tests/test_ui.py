"""Unit tests for Component 7: Streamlit Inspection UI."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "src" / "ui" / "app.py")


def test_streamlit_app_renders_tabs_and_titles():
    """Verify Streamlit app renders successfully without exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    titles = [t.value for t in at.title]
    assert "Cross-Document Fact Knowledge Layer" in titles

    tabs = [t.label for t in at.tabs]
    assert any("Overview" in t for t in tabs)
    assert any("All Facts" in t for t in tabs)
    assert any("Relationships" in t for t in tabs)
    assert any("Failures" in t for t in tabs)
    assert any("Ingested Documents" in t for t in tabs)


def test_streamlit_app_sidebar_elements():
    """Verify sidebar controls exist (file uploader, quick load buttons)."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Sidebar contains file uploader and buttons
    assert len(at.sidebar.button) >= 3
    button_labels = [b.label for b in at.sidebar.button]
    assert any("Delhivery" in lbl for lbl in button_labels)
    assert any("Macro" in lbl for lbl in button_labels)
    assert any("Clear Knowledge Store" in lbl for lbl in button_labels)

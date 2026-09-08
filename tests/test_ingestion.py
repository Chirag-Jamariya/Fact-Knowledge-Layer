"""Unit and integration tests for PDF Ingestion Engine (Component 2)."""

import os
import pytest
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import format_markdown_table, clean_cell
from src.models.corpus import DocumentRecord, PageContent

SAMPLE_PDF_PATH = "/home/hot-coffee/Downloads/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"


def test_clean_cell_and_format_markdown_table():
    """Verify clean_cell cleans spaces and escapes pipes, and format_markdown_table produces valid markdown."""
    assert clean_cell("  Revenue \n (INR Cr)  ") == "Revenue (INR Cr)"
    assert clean_cell("A | B") == "A \\| B"

    headers = ["Metric", "FY23", "FY24"]
    rows = [["Revenue", "7,225", "8,142"], ["EBITDA", "-68", "127"]]
    md = format_markdown_table(headers, rows)

    assert "| Metric | FY23 | FY24 |" in md
    assert "| --- | --- | --- |" in md
    assert "| Revenue | 7,225 | 8,142 |" in md


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF_PATH), reason="Starter dataset PDF not found")
def test_parse_real_starter_pdf_first_pages():
    """Verify that PDFParser parses real starter PDF and produces 1-indexed pages and tables."""
    parser = PDFParser()
    doc_record, pages = parser.parse_document(SAMPLE_PDF_PATH, max_pages=5)

    assert isinstance(doc_record, DocumentRecord)
    assert doc_record.filename == "03-delhivery-q4-fy24-earnings-presentation.pdf"
    assert doc_record.total_pages == 27
    assert len(pages) == 5

    # Check 1-indexed numbering
    for idx, page in enumerate(pages):
        assert isinstance(page, PageContent)
        assert page.page_number == idx + 1
        assert page.doc_id == doc_record.doc_id
        assert isinstance(page.raw_text, str)

    # Check enriched text generation
    enriched = parser.get_enriched_page_text(pages[0])
    assert isinstance(enriched, str)
    assert len(enriched) > 0


def test_file_not_found_raises():
    """Verify parser raises FileNotFoundError for missing path."""
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_document("/non/existent/path.pdf")

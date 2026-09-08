"""Table extraction module for PDF documents."""

import logging
from typing import List, Optional
import pdfplumber
from src.models.corpus import ExtractedTable

logger = logging.getLogger(__name__)


def clean_cell(cell: Optional[str]) -> str:
    """Normalize whitespace and newlines in table cell text."""
    if cell is None:
        return ""
    # Replace internal newlines with space, strip outer whitespace
    cleaned = " ".join(str(cell).split())
    # Escape pipe characters to prevent markdown formatting issues
    return cleaned.replace("|", "\\|")


def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format structured table rows into a standard GitHub-flavored Markdown table."""
    if not headers and not rows:
        return ""
    
    col_count = len(headers) if headers else max(len(row) for row in rows)
    if col_count == 0:
        return ""

    # Ensure header has proper length
    norm_headers = [clean_cell(h) for h in headers] if headers else [f"Col {i+1}" for i in range(col_count)]
    while len(norm_headers) < col_count:
        norm_headers.append(f"Col {len(norm_headers)+1}")

    header_line = "| " + " | ".join(norm_headers) + " |"
    separator_line = "| " + " | ".join(["---"] * col_count) + " |"

    formatted_rows = []
    for row in rows:
        norm_row = [clean_cell(c) for c in row]
        while len(norm_row) < col_count:
            norm_row.append("")
        formatted_rows.append("| " + " | ".join(norm_row[:col_count]) + " |")

    return "\n".join([header_line, separator_line] + formatted_rows)


def extract_tables_from_plumber_page(page: pdfplumber.page.Page, page_number: int) -> List[ExtractedTable]:
    """Extract all tabular regions from a pdfplumber page as structured ExtractedTable objects."""
    extracted_tables: List[ExtractedTable] = []

    try:
        raw_tables = page.extract_tables()
        if not raw_tables:
            return extracted_tables

        for idx, table in enumerate(raw_tables):
            if not table or len(table) < 2:
                # Need at least a header row and one data row
                continue

            # First non-empty row serves as header
            raw_headers = table[0]
            raw_rows = table[1:]

            # Filter out completely empty columns or rows
            cleaned_headers = [clean_cell(c) for c in raw_headers]
            cleaned_rows = [[clean_cell(c) for c in row] for row in raw_rows if any(row)]

            if not cleaned_rows:
                continue

            md_repr = format_markdown_table(cleaned_headers, cleaned_rows)

            extracted_tables.append(
                ExtractedTable(
                    page_number=page_number,
                    headers=cleaned_headers,
                    rows=cleaned_rows,
                    caption=f"Table {idx+1} on Page {page_number}",
                    markdown_repr=md_repr,
                )
            )
    except Exception as e:
        logger.warning(f"Failed to extract tables on page {page_number}: {e}")

    return extracted_tables

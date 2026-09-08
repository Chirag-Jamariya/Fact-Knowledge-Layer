"""High-performance dual-engine PDF parser using PyMuPDF and pdfplumber."""

import os
import hashlib
import logging
from typing import List, Tuple, Optional
import pymupdf as fitz
import pdfplumber

from src.models.corpus import DocumentRecord, PageContent
from src.ingestion.table_extractor import extract_tables_from_plumber_page
from src.ingestion.image_extractor import extract_images_from_fitz_page

logger = logging.getLogger(__name__)


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file for deterministic cataloging."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PDFParser:
    """
    Dual-engine parser:
    - PyMuPDF (fitz) provides high-speed layout-preserving text & visual extraction.
    - pdfplumber provides high-precision tabular boundary extraction.
    """

    def __init__(self, min_image_dim: int = 120):
        self.min_image_dim = min_image_dim

    def parse_document(
        self,
        pdf_path: str,
        max_pages: Optional[int] = None,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> Tuple[DocumentRecord, List[PageContent]]:
        """
        Parse a PDF into a DocumentRecord and a list of 1-indexed PageContent objects.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        checksum = compute_sha256(pdf_path)
        doc_id = hashlib.sha256(f"{filename}:{checksum}".encode("utf-8")).hexdigest()[:12]

        pages_content: List[PageContent] = []

        # Open both fitz and pdfplumber handles
        doc_fitz = fitz.open(pdf_path)
        total_pages = len(doc_fitz)
        num_pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        logger.info(f"Ingesting '{filename}' ({num_pages_to_process}/{total_pages} pages)")

        doc_plumber = None
        if extract_tables:
            try:
                doc_plumber = pdfplumber.open(pdf_path)
            except Exception as e:
                logger.warning(f"Could not open pdfplumber for '{filename}': {e}. Proceeding without tables.")

        try:
            for page_idx in range(num_pages_to_process):
                page_num = page_idx + 1  # 1-indexed
                fitz_page = doc_fitz[page_idx]

                # Extract layout-preserving text blocks
                raw_text = fitz_page.get_text("text") or ""
                # Normalize linebreaks and whitespace
                clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                normalized_text = "\n".join(clean_lines)

                # Extract tables using pdfplumber if available
                tables = []
                if doc_plumber and page_idx < len(doc_plumber.pages):
                    try:
                        plumber_page = doc_plumber.pages[page_idx]
                        tables = extract_tables_from_plumber_page(plumber_page, page_num)
                    except Exception as te:
                        logger.debug(f"Table extraction error on page {page_num}: {te}")

                # Extract images/charts using fitz
                images = []
                if extract_images:
                    try:
                        images = extract_images_from_fitz_page(
                            fitz_page,
                            page_num,
                            min_width=self.min_image_dim,
                            min_height=self.min_image_dim,
                        )
                    except Exception as ie:
                        logger.debug(f"Image extraction error on page {page_num}: {ie}")

                page_obj = PageContent(
                    doc_id=doc_id,
                    page_number=page_num,
                    raw_text=normalized_text,
                    tables=tables,
                    images=images,
                    char_count=len(normalized_text),
                )
                pages_content.append(page_obj)

        finally:
            doc_fitz.close()
            if doc_plumber:
                doc_plumber.close()

        doc_record = DocumentRecord(
            doc_id=doc_id,
            filename=filename,
            file_path=os.path.abspath(pdf_path),
            total_pages=total_pages,
            checksum_sha256=checksum,
            metadata={
                "parsed_pages": num_pages_to_process,
                "total_tables_extracted": sum(len(p.tables) for p in pages_content),
                "total_images_extracted": sum(len(p.images) for p in pages_content),
            },
        )

        return doc_record, pages_content

    @staticmethod
    def get_enriched_page_text(page: PageContent) -> str:
        """
        Produce an enriched text representation of a page that injects
        Markdown-formatted tables directly into the page stream for downstream LLM extraction.
        """
        parts = []
        if page.raw_text:
            parts.append(page.raw_text)

        if page.tables:
            parts.append("\n--- EXTRACTED TABLES ---")
            for t in page.tables:
                if t.caption:
                    parts.append(f"\n### {t.caption}")
                parts.append(t.markdown_repr)

        if page.images:
            parts.append("\n--- EXTRACTED FIGURES ---")
            for img in page.images:
                if img.caption:
                    parts.append(f"- [Figure on Page {page.page_number}]: {img.caption}")

        return "\n\n".join(parts)

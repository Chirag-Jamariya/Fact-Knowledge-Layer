"""Corpus and document ingestion data models."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class ExtractedTable(BaseModel):
    """Structured representation of a table extracted from a PDF page."""
    page_number: int = Field(ge=1, description="1-indexed PDF page number")
    headers: List[str] = Field(default_factory=list, description="Column header strings")
    rows: List[List[str]] = Field(default_factory=list, description="Matrix of row cells as strings")
    caption: Optional[str] = Field(default=None, description="Table title or adjacent caption")
    markdown_repr: str = Field(description="Markdown format representation of the table")


class ExtractedImage(BaseModel):
    """Metadata for an image, chart, or figure extracted from a PDF page."""
    page_number: int = Field(ge=1, description="1-indexed PDF page number")
    image_index: int = Field(ge=0, description="Index of image on the page")
    caption: Optional[str] = Field(default=None, description="Nearby text or chart label")
    width: int
    height: int
    image_format: str = Field(default="png")


class PageContent(BaseModel):
    """Complete extracted content of a single PDF page."""
    doc_id: str
    page_number: int = Field(ge=1, description="1-indexed PDF page number")
    raw_text: str = Field(description="Normalized textual content of the page")
    tables: List[ExtractedTable] = Field(default_factory=list)
    images: List[ExtractedImage] = Field(default_factory=list)
    char_count: int = Field(default=0)


class DocumentRecord(BaseModel):
    """Catalog record for an ingested PDF document."""
    model_config = ConfigDict(extra="allow")

    doc_id: str = Field(description="Unique identifier for the document")
    filename: str = Field(description="Original filename (e.g., annual_report.pdf)")
    file_path: str = Field(description="Absolute path on disk")
    total_pages: int = Field(ge=1, description="Total number of pages in document")
    checksum_sha256: str = Field(description="SHA256 checksum of the file content")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    corpus: Optional[str] = Field(default=None, description="Logical corpus identifier")

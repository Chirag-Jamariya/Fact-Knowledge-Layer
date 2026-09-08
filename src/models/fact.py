"""Fact and Source Evidence data models."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib


class FactCategory(str, Enum):
    NUMERICAL = "numerical"
    SEMANTIC = "semantic"


class SourceEvidence(BaseModel):
    """Verifiable link grounding a fact directly in its source document."""
    document_id: str = Field(description="Unique identifier or filename of the source document")
    page_number: int = Field(ge=1, description="1-indexed page number in the original PDF")
    section_title: Optional[str] = Field(default=None, description="Nearby section header or chapter name")
    content_type: str = Field(default="text", description="'text' | 'table' | 'chart_caption'")
    verbatim_quote: str = Field(description="Exact snippet, sentence, or table row containing the factual assertion")
    bounding_box: Optional[Dict[str, float]] = Field(default=None, description="Optional bounding box coordinates")


class ContextDimensions(BaseModel):
    """Contextual metadata explaining conditions under which the fact is true."""
    temporal_period: Optional[str] = Field(default=None, description="e.g., 'FY 2023-24', 'Q4 FY24', '2022', 'Calendar Year 2024'")
    reporting_scope: Optional[str] = Field(default=None, description="e.g., 'Consolidated', 'Standalone', 'India Operations', 'Global'")
    unit: Optional[str] = Field(default=None, description="e.g., 'INR Crores', 'Millions', 'Percentage', 'Count', 'USD'")
    accounting_standard: Optional[str] = Field(default=None, description="e.g., 'Ind AS', 'IFRS', 'US GAAP'")


class Fact(BaseModel):
    """Atomic factual statement with grounding and contextual attributes."""
    fact_id: str = Field(default="", description="Deterministic unique identifier")
    category: FactCategory = Field(default=FactCategory.NUMERICAL)
    entity: str = Field(description="Subject entity (e.g., 'Delhivery Limited', 'Indian Economy')")
    metric: str = Field(description="Canonical metric or property (e.g., 'Revenue from Operations', 'EBITDA Margin', 'Headline CPI')")
    value: str = Field(description="Normalized value string (e.g., '8,142 Cr', '5.4%', '740 Million')")
    raw_value: str = Field(description="Exact literal value as extracted from the text/table")
    context: ContextDimensions = Field(default_factory=ContextDimensions)
    evidence: SourceEvidence
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    corpus: str = Field(default="", description="Logical corpus name (e.g. 'India Macroeconomics', 'Delhivery')")
    metric_type: str = Field(default="", description="Canonical definition type e.g., 'Adjusted EBITDA', 'Revenue from Operations'")
    value_nature: str = Field(default="absolute_currency", description="'percentage' | 'margin' | 'absolute_currency' | 'count' | 'growth_rate' | 'ratio'")
    temporal_nature: str = Field(default="period_duration", description="'cumulative' | 'point_in_time' | 'period_duration'")
    is_header_metadata: bool = Field(default=False, description="Whether extracted value is an erroneous table header or column year artifact")

    def model_post_init(self, __context: Any) -> None:
        import re

        # Disambiguate similar metrics based on quote and metric text
        quote_lower = (self.evidence.verbatim_quote or "").lower()
        metric_lower = (self.metric or "").lower()
        val_lower = (self.value or "").lower()

        # 1. EBITDA / Margin Disambiguation
        if "ebitda" in metric_lower or "ebitda" in quote_lower:
            if "adjusted ebitda margin" in quote_lower or ("adj" in quote_lower and "margin" in quote_lower and "ebitda" in quote_lower):
                self.metric = "Adjusted EBITDA Margin"
                self.metric_type = "Adjusted EBITDA Margin"
                self.value_nature = "margin"
            elif "ebitda margin" in quote_lower or "ebitda margin" in metric_lower:
                self.metric = "EBITDA Margin"
                self.metric_type = "EBITDA Margin"
                self.value_nature = "margin"
            elif "service ebitda" in quote_lower or "service ebitda" in metric_lower:
                self.metric = "Service EBITDA"
                self.metric_type = "Service EBITDA"
                self.value_nature = "absolute_currency"
            elif "reported ebitda" in quote_lower or "reported ebitda" in metric_lower:
                self.metric = "Reported EBITDA"
                self.metric_type = "Reported EBITDA"
                self.value_nature = "absolute_currency"
            elif "adjusted" in metric_lower or "adjusted" in quote_lower or "adj." in quote_lower or "adj " in quote_lower:
                self.metric = "Adjusted EBITDA"
                self.metric_type = "Adjusted EBITDA"
                self.value_nature = "absolute_currency"
            elif "ebitda" in metric_lower:
                self.metric = "EBITDA"
                self.metric_type = "EBITDA"
                self.value_nature = "absolute_currency"

        # 2. Revenue Disambiguation
        elif "revenue" in metric_lower or "revenue" in quote_lower:
            if "revenue from services" in quote_lower:
                self.metric = "Revenue from Services"
                self.metric_type = "Revenue from Services"
            elif "contracts with customers" in quote_lower or "revenue from customers" in quote_lower:
                self.metric = "Revenue from Customers"
                self.metric_type = "Revenue from Customers"
            elif "revenue from operations" in metric_lower or "revenue from operations" in quote_lower or "operating revenue" in quote_lower:
                self.metric = "Revenue from Operations"
                self.metric_type = "Revenue from Operations"

        # Default metric_type
        if not self.metric_type:
            self.metric_type = self.metric

        # 3. Value Nature Determination
        if "margin" in self.metric.lower() or "margin" in quote_lower:
            self.value_nature = "margin"
        elif any(k in self.metric.lower() for k in ["growth", "inflation", "cpi", "deficit", "rate"]):
            self.value_nature = "growth_rate"
        elif "%" in self.value or "%" in self.raw_value or (self.context and self.context.unit and "%" in self.context.unit):
            if "margin" in self.metric.lower():
                self.value_nature = "margin"
            elif any(k in self.metric.lower() for k in ["growth", "inflation", "cpi", "rate"]):
                self.value_nature = "growth_rate"
            else:
                self.value_nature = "percentage"
        elif any(k in self.metric.lower() for k in ["pin codes", "delivery centres", "sortation centres", "centres", "gateways", "headcount", "workforce", "shipments", "volume"]):
            self.value_nature = "count"
        elif any(c in self.value for c in ["₹", "$", "€", "£"]) or any(u in (self.context.unit or "").lower() for u in ["cr", "crore", "mn", "million", "bn", "billion", "inr", "usd", "rs"]):
            self.value_nature = "absolute_currency"
        elif not self.value_nature:
            self.value_nature = "numerical"

        # 4. Temporal Nature Determination
        period_lower = (self.context.temporal_period or "").lower()
        if any(k in period_lower for k in ["since inception", "since incorporation", "cumulative", "lifetime"]) or any(k in quote_lower for k in ["since inception", "since incorporation", "cumulative shipments", "over lifetime", "cumulative"]):
            self.temporal_nature = "cumulative"
        elif any(k in period_lower for k in ["as of", "as at", "at march", "at end"]):
            self.temporal_nature = "point_in_time"
        else:
            self.temporal_nature = "period_duration"

        # 5. Table Header / Column Year Extraction Anomaly Detection
        clean_val = self.raw_value.strip(" ,.:;()[]/\\")
        if re.fullmatch(r"20\d{2}", clean_val):
            # Check if this 4-digit number was extracted from a table header row or date column
            if any(h in quote_lower for h in ["table header", "column header", "fiscal year ended", "year ended", "ended march 31", "ended december 31", "for the year ended"]):
                self.is_header_metadata = True
            elif any(k in self.metric.lower() for k in ["revenue", "ebitda", "growth", "deficit"]):
                self.is_header_metadata = True

        if not self.fact_id:
            # Generate deterministic hash if not provided
            key = f"{self.evidence.document_id}:{self.evidence.page_number}:{self.entity}:{self.metric}:{self.value}"
            self.fact_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

        if self.corpus and self.corpus.strip():
            self.corpus = self.corpus.strip()
        else:
            ent = (self.entity or "").lower()
            doc = (self.evidence.document_id or "").lower()
            if "delhivery" in ent or "delhivery" in doc:
                self.corpus = "Delhivery"
            elif any(k in ent or k in doc for k in ["india", "rbi", "economic survey", "imf india", "indian economy"]):
                self.corpus = "India Macroeconomics"
            elif self.entity and self.entity.strip():
                self.corpus = self.entity.strip()
            else:
                self.corpus = "General Knowledge"

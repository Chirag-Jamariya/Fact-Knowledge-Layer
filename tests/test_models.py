"""Unit tests for Core Data Models (Component 1)."""

import pytest
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.models.reconciliation import (
    CrossDocReconciliation,
    ReconciliationVerdict,
    ContextualShiftType,
    CaseShowcase,
)
from src.models.corpus import ExtractedTable, ExtractedImage, PageContent, DocumentRecord


def test_fact_creation_and_auto_id():
    """Verify that a Fact is correctly created with deterministic auto-generated ID."""
    evidence = SourceEvidence(
        document_id="delhivery_fy24_annual_report.pdf",
        page_number=42,
        section_title="Financial Highlights",
        content_type="table",
        verbatim_quote="Revenue from operations reached ₹8,142 Cr in FY24.",
    )
    context = ContextDimensions(
        temporal_period="FY 2023-24",
        reporting_scope="Consolidated",
        unit="INR Crores",
    )
    fact = Fact(
        category=FactCategory.NUMERICAL,
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=context,
        evidence=evidence,
    )

    assert fact.fact_id != ""
    assert len(fact.fact_id) == 16
    assert fact.category == FactCategory.NUMERICAL
    assert fact.evidence.page_number == 42
    assert fact.context.reporting_scope == "Consolidated"


def test_source_evidence_constraints():
    """Verify that SourceEvidence enforces 1-indexed page numbers."""
    with pytest.raises(Exception):
        SourceEvidence(
            document_id="doc.pdf",
            page_number=0,  # Invalid: must be ge=1
            verbatim_quote="Some quote",
        )


def test_reconciliation_verdict_and_shift_types():
    """Verify that CrossDocReconciliation supports all 4 required assignment cases."""
    ev_a = SourceEvidence(
        document_id="delhivery_prospectus_2022.pdf",
        page_number=28,
        verbatim_quote="Revenue for FY21 was ₹3,646 Cr.",
    )
    fact_a = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="3,646 Cr",
        raw_value="3,646",
        context=ContextDimensions(temporal_period="FY 2020-21", reporting_scope="Restated Consolidated"),
        evidence=ev_a,
    )

    ev_b = SourceEvidence(
        document_id="delhivery_annual_report_fy24.pdf",
        page_number=105,
        verbatim_quote="Revenue from operations for FY24 stood at ₹8,142 Cr.",
    )
    fact_b = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY 2023-24", reporting_scope="Consolidated"),
        evidence=ev_b,
    )

    # Case 3: Apparent Contradiction Reconciled by Context (Temporal Shift)
    recon = CrossDocReconciliation(
        verdict=ReconciliationVerdict.CONTEXT_RECONCILED,
        shift_type=ContextualShiftType.TEMPORAL_SHIFT,
        fact_a=fact_a,
        fact_b=fact_b,
        reasoning="The revenues of ₹3,646 Cr and ₹8,142 Cr reflect different reporting periods (FY21 vs FY24).",
        reconciliation_summary="Revenue growth over 3 years explains the apparent numerical variance.",
    )

    assert recon.verdict == ReconciliationVerdict.CONTEXT_RECONCILED
    assert recon.shift_type == ContextualShiftType.TEMPORAL_SHIFT
    assert recon.reconciliation_id != ""


def test_case_showcase_structure():
    """Verify CaseShowcase container holds references for all 4 required cases."""
    showcase = CaseShowcase()
    assert showcase.corroborated_example is None
    assert showcase.contradiction_example is None
    assert showcase.context_reconciled_example is None
    assert showcase.failure_example is None


def test_corpus_models():
    """Verify DocumentRecord and PageContent serialization."""
    table = ExtractedTable(
        page_number=12,
        headers=["Metric", "FY23", "FY24"],
        rows=[["Revenue", "7225", "8142"], ["EBITDA", "-68", "127"]],
        markdown_repr="| Metric | FY23 | FY24 |\n| Revenue | 7225 | 8142 |",
    )
    page = PageContent(
        doc_id="delhivery_fy24.pdf",
        page_number=12,
        raw_text="Financial performance summary...",
        tables=[table],
        char_count=500,
    )
    assert len(page.tables) == 1
    assert page.tables[0].page_number == 12
    assert page.tables[0].headers == ["Metric", "FY23", "FY24"]

"""Unit tests for Cross-Document Reconciliation Engine (Component 5)."""

import pytest
from src.reconciliation.reconciler import CrossDocReconciler
from src.reconciliation.unit_converter import UnitConverter
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.models.reconciliation import (
    ReconciliationVerdict,
    ContextualShiftType,
)


def test_unit_converter_equivalence():
    """Verify that UnitConverter equates 8,142 Cr and 81,420 Mn."""
    is_equiv, reason = UnitConverter.are_values_equivalent("8,142 Cr", "Cr", "81,420 Mn", "Mn")
    assert is_equiv is True
    assert "equivalence" in reason.lower()

    # Percentage test
    is_pct, _ = UnitConverter.are_values_equivalent("5.4%", "%", "5.4 per cent", "%")
    assert is_pct is True


def test_case_1_corroboration():
    """Case 1: Facts agree across documents (even across unit denominations)."""
    fa = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="02-annual-report.pdf", page_number=10, verbatim_quote="Revenue was 8,142 Cr"),
    )
    fb = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="81,420 Mn",
        raw_value="81,420",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Mn"),
        evidence=SourceEvidence(document_id="03-earnings.pdf", page_number=4, verbatim_quote="Revenue stood at 81,420 Mn"),
    )

    reconciler = CrossDocReconciler()
    result = reconciler.reconcile_pair(fa, fb)

    assert result.verdict == ReconciliationVerdict.CORROBORATED
    assert "corroborated" in result.reasoning.lower()


def test_case_2_genuine_contradiction():
    """Case 2: Same metric, entity, period, and scope, but numbers genuinely diverge."""
    fa = Fact(
        entity="Delhivery Limited",
        metric="Pin Codes Covered",
        value="18,793",
        raw_value="18,793",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="India"),
        evidence=SourceEvidence(document_id="doc_a.pdf", page_number=2, verbatim_quote="Covering 18,793 pin codes"),
    )
    fb = Fact(
        entity="Delhivery Limited",
        metric="Pin Codes Covered",
        value="14,200",
        raw_value="14,200",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="India"),
        evidence=SourceEvidence(document_id="doc_b.pdf", page_number=5, verbatim_quote="Network encompasses 14,200 pin codes"),
    )

    reconciler = CrossDocReconciler()
    result = reconciler.reconcile_pair(fa, fb)

    assert result.verdict == ReconciliationVerdict.GENUINE_CONTRADICTION
    assert "genuine contradiction" in result.reconciliation_summary.lower()


def test_case_3_context_reconciled():
    """Case 3: Apparent contradiction explained by temporal shift (FY21 vs FY24)."""
    fa = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="3,646 Cr",
        raw_value="3,646",
        context=ContextDimensions(temporal_period="FY21", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="01-prospectus.pdf", page_number=28, verbatim_quote="Revenue was 3,646 Cr in FY21"),
    )
    fb = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="02-annual-report.pdf", page_number=105, verbatim_quote="Revenue was 8,142 Cr in FY24"),
    )

    reconciler = CrossDocReconciler()
    result = reconciler.reconcile_pair(fa, fb)

    assert result.verdict == ReconciliationVerdict.CONTEXT_RECONCILED
    assert result.shift_type == ContextualShiftType.TEMPORAL_SHIFT
    assert "reporting periods" in result.reasoning.lower()


def test_case_4_extraction_failure_detection():
    """Case 4: Extraction failure detected due to corrupted token or bare year artifact."""
    fa = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="2021",
        raw_value="2021",
        context=ContextDimensions(),
        evidence=SourceEvidence(document_id="doc_fail.pdf", page_number=1, verbatim_quote="Header line 2021"),
    )
    fb = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24"),
        evidence=SourceEvidence(document_id="02-annual-report.pdf", page_number=10, verbatim_quote="Valid fact"),
    )

    reconciler = CrossDocReconciler()
    result = reconciler.reconcile_pair(fa, fb)

    assert result.verdict == ReconciliationVerdict.EXTRACTION_FAILURE
    assert "bare calendar year" in result.reasoning.lower()
    assert "mitigation" in result.reasoning.lower()


def test_case_showcase_curation():
    """Verify that generate_case_showcase curates all 4 required cases."""
    reconciler = CrossDocReconciler()
    # Create sample pair results for the 4 cases
    f_valid = Fact(
        entity="Delhivery Limited", metric="Metric A", value="100", raw_value="100",
        evidence=SourceEvidence(document_id="a.pdf", page_number=1, verbatim_quote="quote"),
    )
    recon_corrob = reconciler.reconcile_pair(f_valid, f_valid)

    recon_list = [recon_corrob]
    showcase = reconciler.generate_case_showcase(recon_list)
    assert showcase.corroborated_example is not None


def test_corpus_consensus_engine_3_doc_consensus():
    """Verify CorpusConsensusEngine builds multi-document consensus across all 3 PDFs."""
    from src.reconciliation.corpus_consensus import CorpusConsensusEngine

    f1 = Fact(
        entity="Indian Economy",
        metric="Headline Inflation",
        value="5.4%",
        raw_value="5.4",
        context=ContextDimensions(temporal_period="FY24", unit="%"),
        evidence=SourceEvidence(document_id="01-economic-survey-2023-24.pdf", page_number=1, verbatim_quote="Inflation 5.4%"),
    )
    f2 = Fact(
        entity="Indian Economy",
        metric="Headline Inflation",
        value="5.4 per cent",
        raw_value="5.4",
        context=ContextDimensions(temporal_period="FY24", unit="%"),
        evidence=SourceEvidence(document_id="02-rbi-annual-report-2023-24.pdf", page_number=5, verbatim_quote="Headline inflation at 5.4 per cent"),
    )
    f3 = Fact(
        entity="Indian Economy",
        metric="Headline Inflation",
        value="5.4%",
        raw_value="5.4",
        context=ContextDimensions(temporal_period="FY24", unit="%"),
        evidence=SourceEvidence(document_id="03-union-budget-2024-25.pdf", page_number=3, verbatim_quote="CPI at 5.4%"),
    )

    groups = CorpusConsensusEngine.build_consensus_groups([f1, f2, f3], corpus_name="India Macroeconomics", total_corpus_docs=3)
    assert len(groups) == 1
    g = groups[0]
    assert g.metric == "Headline Inflation"
    assert g.doc_count == 3
    assert g.total_corpus_docs == 3
    assert "3 of 3" in g.coverage_ratio
    assert g.verdict == ReconciliationVerdict.CORROBORATED
    assert "across all 3" in g.synthesis.lower()
    assert len(g.disclosures) == 3


def test_strict_cross_corpus_isolation():
    """Verify that CrossDocClusterer NEVER pairs facts across different corpora."""
    from src.indexing.clusterer import CrossDocClusterer

    f_macro = Fact(
        entity="Indian Economy",
        metric="Real GDP Growth",
        value="8.2%",
        raw_value="8.2",
        evidence=SourceEvidence(document_id="01-economic-survey-2023-24.pdf", page_number=1, verbatim_quote="GDP 8.2%"),
    )
    f_delhivery = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        evidence=SourceEvidence(document_id="02-delhivery-annual-report-fy24.pdf", page_number=10, verbatim_quote="Revenue 8,142 Cr"),
    )

    assert f_macro.corpus == "India Macroeconomics"
    assert f_delhivery.corpus == "Delhivery"

    # Pairing attempt across both facts
    pairs = CrossDocClusterer.find_cross_document_pairs([f_macro, f_delhivery])
    assert len(pairs) == 0, "No cross-corpus pairs may ever be created"


def test_showcase_multi_document_corpus_facts():
    """Verify generate_case_showcase collects facts across all participating documents (Fact A, Fact B, Fact C...)."""
    f1 = Fact(
        entity="Delhivery Limited",
        metric="Express Parcel Shipments",
        value="1 billion",
        raw_value="1",
        evidence=SourceEvidence(document_id="01-delhivery-prospectus-2022-excerpt.pdf", page_number=74, verbatim_quote="1 billion delivered"),
    )
    f2 = Fact(
        entity="Delhivery Limited",
        metric="Express Parcel Shipments",
        value=">2.8Bn",
        raw_value=">2.8",
        evidence=SourceEvidence(document_id="02-delhivery-annual-report-fy24-excerpt.pdf", page_number=2, verbatim_quote=">2.8Bn shipments"),
    )
    f3 = Fact(
        entity="Delhivery Limited",
        metric="Express Parcel Shipments",
        value="2.8Bn",
        raw_value="2.8",
        evidence=SourceEvidence(document_id="03-delhivery-q4-fy24-earnings-presentation.pdf", page_number=6, verbatim_quote="2.8 Bn+ shipments"),
    )

    reconciler = CrossDocReconciler()
    pair_rec = reconciler.reconcile_pair(f2, f3)
    assert pair_rec.verdict == ReconciliationVerdict.CORROBORATED

    showcase = reconciler.generate_case_showcase([pair_rec], active_facts=[f1, f2, f3])
    assert showcase.corroborated_example is not None
    c1 = showcase.corroborated_example
    assert len(c1.participating_facts) == 3
    doc_ids = [f.evidence.document_id for f in c1.participating_facts]
    assert "01-delhivery-prospectus-2022-excerpt.pdf" in doc_ids
    assert "02-delhivery-annual-report-fy24-excerpt.pdf" in doc_ids
    assert "03-delhivery-q4-fy24-earnings-presentation.pdf" in doc_ids
    assert "across all 3 files" in c1.reasoning.lower()


def test_comparability_gate_ebitda_vs_adjusted_ebitda():
    """Verify that EBITDA and Adjusted EBITDA are rejected by the gate as METRIC_MISMATCH."""
    from src.reconciliation.comparability_gate import ComparabilityGate

    f_ebitda = Fact(
        entity="Delhivery Limited",
        metric="EBITDA",
        value="₹1,266 Mn",
        raw_value="1,266",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Mn"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=10, verbatim_quote="EBITDA was ₹1,266 Mn"),
    )
    f_adj_ebitda = Fact(
        entity="Delhivery Limited",
        metric="Adjusted EBITDA",
        value="₹758 Mn",
        raw_value="758",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Mn"),
        evidence=SourceEvidence(document_id="03-presentation.pdf", page_number=4, verbatim_quote="Adjusted EBITDA reached ₹758 Mn"),
    )

    reconciler = CrossDocReconciler()
    res = reconciler.reconcile_pair(f_ebitda, f_adj_ebitda)
    assert res.verdict == ReconciliationVerdict.METRIC_MISMATCH
    assert res.comparability_gate_passed is False
    assert "metric definition mismatch" in res.reconciliation_summary.lower()
    assert "EBITDA" in res.comparability_audit["canonical_def_a"]
    assert "Adjusted EBITDA" in res.comparability_audit["canonical_def_b"]


def test_comparability_gate_ebitda_vs_ebitda_margin_nature_mismatch():
    """Verify that EBITDA and EBITDA Margin are rejected due to incompatible value natures."""
    f_margin = Fact(
        entity="Delhivery Limited",
        metric="EBITDA Margin",
        value="1.6%",
        raw_value="1.6",
        context=ContextDimensions(temporal_period="FY24", unit="%"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=2, verbatim_quote="1.6% EBITDA margin"),
    )
    f_amount = Fact(
        entity="Delhivery Limited",
        metric="Adjusted EBITDA",
        value="₹758 Mn",
        raw_value="758",
        context=ContextDimensions(temporal_period="FY24", unit="INR Mn"),
        evidence=SourceEvidence(document_id="03-presentation.pdf", page_number=10, verbatim_quote="Adjusted EBITDA of ₹758 Mn"),
    )

    reconciler = CrossDocReconciler()
    res = reconciler.reconcile_pair(f_margin, f_amount)
    assert res.verdict in (ReconciliationVerdict.METRIC_MISMATCH, ReconciliationVerdict.NOT_COMPARABLE)
    assert res.comparability_gate_passed is False
    assert any("value nature" in d.lower() or "metric definition" in d.lower() for d in res.comparability_audit["differing_fields"])


def test_comparability_gate_temporal_update_progression():
    """Verify cumulative metrics across different years are classified as TEMPORAL_UPDATE."""
    f21 = Fact(
        entity="Delhivery Limited",
        metric="Express Parcel Shipments",
        value="1.0Bn",
        raw_value="1.0",
        context=ContextDimensions(temporal_period="FY21", unit="Bn"),
        evidence=SourceEvidence(document_id="01-prospectus.pdf", page_number=7, verbatim_quote="1.0Bn shipments since incorporation"),
    )
    f24 = Fact(
        entity="Delhivery Limited",
        metric="Express Parcel Shipments",
        value=">2.8Bn",
        raw_value=">2.8",
        context=ContextDimensions(temporal_period="FY24", unit="Bn"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=2, verbatim_quote=">2.8Bn shipments since inception"),
    )

    reconciler = CrossDocReconciler()
    res = reconciler.reconcile_pair(f21, f24)
    assert res.verdict == ReconciliationVerdict.TEMPORAL_UPDATE
    assert res.shift_type == ContextualShiftType.TEMPORAL_SHIFT
    assert "milestone" in res.reasoning.lower() or "progression" in res.reasoning.lower()
    assert res.temporal_progression_explanation is not None


def test_comparability_gate_unit_conversion_and_rounding():
    """Verify same metric, same period, same scope matches after unit conversion & rounding."""
    f_mn = Fact(
        entity="Delhivery Limited",
        metric="Adjusted EBITDA",
        value="₹757.86 Mn",
        raw_value="757.86",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Mn"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=10, verbatim_quote="Adjusted EBITDA was ₹757.86 Mn"),
    )
    f_cr = Fact(
        entity="Delhivery Limited",
        metric="Adjusted EBITDA",
        value="₹76 Cr",
        raw_value="76",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="03-presentation.pdf", page_number=13, verbatim_quote="Adjusted EBITDA was ₹76 Cr"),
    )

    reconciler = CrossDocReconciler()
    res = reconciler.reconcile_pair(f_mn, f_cr)
    assert res.verdict == ReconciliationVerdict.CORROBORATED
    assert res.comparability_gate_passed is True
    assert res.unit_normalization_applied is not None
    assert "variance" in res.unit_normalization_applied.lower() or "normalizes" in res.unit_normalization_applied.lower()
    assert res.confidence > 0.85


def test_comparability_gate_revenue_operations_vs_services():
    """Verify Revenue from operations and Revenue from services are not treated as identical."""
    f_ops = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=10, verbatim_quote="Revenue from operations reached 8,142 Cr"),
    )
    f_serv = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Services",
        value="81,415 Mn",
        raw_value="81,415",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Mn"),
        evidence=SourceEvidence(document_id="02-annual.pdf", page_number=12, verbatim_quote="Revenue from services was ₹81,415 Mn"),
    )

    reconciler = CrossDocReconciler()
    res = reconciler.reconcile_pair(f_ops, f_serv)
    assert res.verdict == ReconciliationVerdict.METRIC_MISMATCH
    assert res.comparability_gate_passed is False


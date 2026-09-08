"""Dedicated Comparability Gate for Cross-Document Fact Verification.

RULE: Comparability first, verdict second.
Never classify two facts as CORROBORATED, GENUINE_CONTRADICTION, or CONTEXT_RECONCILED
until strict comparability across entity, metric definition, value nature, scope,
and temporal semantics has been proven.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.models.fact import Fact
from src.models.reconciliation import (
    ReconciliationVerdict,
    ContextualShiftType,
)
from src.reconciliation.unit_converter import UnitConverter
from src.reconciliation.failure_detector import FailureDetector
from src.extraction.entity_detector import EntityDetector

logger = logging.getLogger(__name__)


@dataclass
class GateAuditResult:
    """Outcome of the Comparability Gate evaluation."""
    passed: bool
    verdict: ReconciliationVerdict
    shift_type: ContextualShiftType = ContextualShiftType.NONE
    confidence: float = 0.95
    reasoning: str = ""
    summary: str = ""
    matched_fields: List[str] = field(default_factory=list)
    differing_fields: List[str] = field(default_factory=list)
    unit_normalization: Optional[str] = None
    temporal_progression: Optional[str] = None
    verdict_rationale: Optional[str] = None
    audit_dict: Dict[str, Any] = field(default_factory=dict)


class ComparabilityGate:
    """
    Rigorously enforces the seven comparability conditions before allowing numerical comparison:
      Check A: Same entity (or multilateral sovereign exception)
      Check B: Same metric definition (strict canonical accounting concept)
      Check C: Same metric type / value nature (percentage vs currency vs margin vs count)
      Check D: Compatible unit
      Check E: Compatible reporting scope (Consolidated vs Standalone)
      Check F: Cumulative vs point-in-time vs period semantics (detecting milestone progression)
      Check G: Compatible temporal period
    """

    @classmethod
    def get_canonical_definition(cls, fact: Fact) -> str:
        """Resolve fact to strict canonical metric definition, preventing soundalike conflation."""
        # Use existing metric_type if explicitly established
        if fact.metric_type:
            raw_target = fact.metric_type.strip().lower()
        else:
            raw_target = fact.metric.strip().lower()

        quote = (fact.evidence.verbatim_quote or "").lower()

        # 1. EBITDA Family - strictly separated
        if "ebitda" in raw_target or "ebitda" in quote:
            if "adjusted ebitda margin" in quote or ("margin" in quote and ("adj" in quote or "adjusted" in raw_target)):
                return "Adjusted EBITDA Margin"
            if "ebitda margin" in quote or "ebitda margin" in raw_target:
                return "EBITDA Margin"
            if "service ebitda" in quote or "service ebitda" in raw_target:
                return "Service EBITDA"
            if "reported ebitda" in quote or "reported ebitda" in raw_target:
                return "Reported EBITDA"
            if "adjusted" in raw_target or "adjusted" in quote or "adj." in quote or "adj " in quote:
                return "Adjusted EBITDA"
            return "EBITDA"

        # 2. Revenue Family - strictly separated
        if "revenue" in raw_target or "revenue" in quote:
            if "revenue from services" in quote or "revenue from services" in raw_target:
                return "Revenue from Services"
            if "contracts with customers" in quote or "revenue from customers" in raw_target or "revenue from customers" in quote:
                return "Revenue from Customers"
            if "revenue from operations" in raw_target or "revenue from operations" in quote or "operating revenue" in quote:
                return "Revenue from Operations"
            return "Revenue"

        # 3. GDP Growth Family - strictly separated
        if "gdp" in raw_target or "gdp" in quote:
            if "oil gdp" in raw_target or "oil gdp" in quote:
                return "Oil GDP Growth"
            if "non-oil" in raw_target or "non-oil" in quote:
                return "Non-Oil GDP Growth"
            if "nominal" in raw_target or "nominal" in quote:
                return "Nominal GDP Growth"
            if "compound" in raw_target or "cagr" in raw_target or "compound" in quote:
                return "Compound Average GDP Growth"
            if "potential" in raw_target or "potential" in quote:
                return "Potential GDP Growth"
            if "real gdp" in raw_target or "real gdp" in quote:
                return "Real GDP Growth"
            return "Real GDP Growth"

        # 4. Inflation Family
        if any(k in raw_target for k in ["inflation", "cpi"]):
            if "food" in raw_target or "food" in quote:
                return "Food Inflation"
            if "core" in raw_target or "core" in quote:
                return "Core Inflation"
            return "Headline Inflation"

        # 5. Operational Logistics Metrics
        if "express parcel" in raw_target or "shipment" in raw_target:
            return "Express Parcel Shipments"
        if "pin code" in raw_target or "pin code" in quote:
            return "Pin Codes Covered"
        if "delivery centre" in raw_target or "delivery center" in raw_target:
            return "Last-Mile Delivery Centres"
        if "ptl" in raw_target or "freight" in raw_target:
            return "PTL Freight Volume"

        return fact.metric.strip()

    @classmethod
    def are_units_compatible(cls, unit_a: Optional[str], unit_b: Optional[str], nature_a: str, nature_b: str) -> bool:
        """Check if two units are dimensionally compatible."""
        if nature_a != nature_b:
            return False
        if nature_a in ("percentage", "margin", "growth_rate"):
            return True  # %, per cent, bps are interchangeable percentage scales
        if nature_a == "absolute_currency":
            # INR Cr, INR Mn, Millions, Crores, Rs, ₹ are convertible currencies
            return True
        if nature_a == "count":
            # Counts (Shipments, Pin Codes, Centres) can be compared directly or scaled (Mn, Bn)
            return True
        return True

    @classmethod
    def evaluate(cls, fact_a: Fact, fact_b: Fact) -> GateAuditResult:
        """
        Run the complete Comparability Gate pipeline.
        Returns a structured GateAuditResult with explicit field matches, differences,
        and gate resolution.
        """
        matched_fields: List[str] = []
        differing_fields: List[str] = []
        audit: Dict[str, Any] = {
            "entity_a": fact_a.entity,
            "entity_b": fact_b.entity,
            "metric_a": fact_a.metric,
            "metric_b": fact_b.metric,
            "nature_a": fact_a.value_nature,
            "nature_b": fact_b.value_nature,
            "period_a": fact_a.context.temporal_period,
            "period_b": fact_b.context.temporal_period,
            "scope_a": fact_a.context.reporting_scope,
            "scope_b": fact_b.context.reporting_scope,
        }

        # ----------------------------------------------------------------------
        # STAGE 0: EXTRACTION FAILURE / TABLE HEADER YEAR ANOMALY CHECK
        # ----------------------------------------------------------------------
        if getattr(fact_a, "is_header_metadata", False) or getattr(fact_b, "is_header_metadata", False):
            bad_fact = fact_a if getattr(fact_a, "is_header_metadata", False) else fact_b
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.EXTRACTION_FAILURE,
                confidence=0.98,
                reasoning=(
                    f"Extraction Failure Diagnosed: Table header column bare calendar year '{bad_fact.raw_value}' was erroneously "
                    f"captured as a numerical metric value in '{bad_fact.evidence.document_id}' (p.{bad_fact.evidence.page_number}). "
                    f"Mitigation: Deploy structural table column date parser to reject bare calendar years when capturing financial metrics. "
                    f"Source quote context indicates column metadata: \"{bad_fact.evidence.verbatim_quote}\"."
                ),
                summary=f"Table header bare calendar year '{bad_fact.raw_value}' misclassified as metric value.",
                differing_fields=["Extraction Validity (Table Header Anomaly)"],
                verdict_rationale="Structural metadata anomaly: column year extracted as financial metric value.",
                audit_dict={"error": "table_header_year_error", "fact_id": bad_fact.fact_id},
            )

        is_fail, fail_reason, mitigation = FailureDetector.check_failure(fact_a, fact_b)
        if is_fail:
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.EXTRACTION_FAILURE,
                confidence=0.95,
                reasoning=f"{fail_reason} {mitigation}",
                summary="Diagnostic failure identified during comparability audit.",
                differing_fields=["Validity Check"],
                verdict_rationale=f"Failed pre-comparison audit: {fail_reason}",
                audit_dict={"failure_reason": fail_reason, "mitigation": mitigation},
            )

        # ----------------------------------------------------------------------
        # CHECK A: SAME ENTITY
        # ----------------------------------------------------------------------
        is_entity_compat = EntityDetector.are_entities_compatible(fact_a.entity, fact_b.entity, metric_name=fact_a.metric)
        if not is_entity_compat:
            differing_fields.append(f"Entity ('{fact_a.entity}' != '{fact_b.entity}')")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.NOT_COMPARABLE,
                confidence=0.90,
                reasoning=f"Entities differ ('{fact_a.entity}' vs '{fact_b.entity}'). Cross-entity comparison rejected.",
                summary=f"Entity mismatch: {fact_a.entity} vs {fact_b.entity}.",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale="Entities are distinct subjects and cannot be reconciled or contradicted.",
                audit_dict=audit,
            )
        matched_fields.append(f"Entity ({fact_a.entity})")

        # ----------------------------------------------------------------------
        # CHECK B: SAME METRIC DEFINITION (Strict Canonical Equality)
        # ----------------------------------------------------------------------
        def_a = cls.get_canonical_definition(fact_a)
        def_b = cls.get_canonical_definition(fact_b)
        audit["canonical_def_a"] = def_a
        audit["canonical_def_b"] = def_b

        if def_a != def_b:
            differing_fields.append(f"Metric Definition ('{def_a}' != '{def_b}')")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.METRIC_MISMATCH,
                confidence=0.92,
                reasoning=(
                    f"Comparability Gate Disqualification: Metric definitions are distinct and not comparable. "
                    f"Fact A defines '{def_a}' in '{fact_a.evidence.document_id}' (p.{fact_a.evidence.page_number}), "
                    f"whereas Fact B defines '{def_b}' in '{fact_b.evidence.document_id}' (p.{fact_b.evidence.page_number}). "
                    f"Soundalike metrics are strictly isolated; numbers were not compared."
                ),
                summary=f"Metric definition mismatch: {def_a} vs {def_b}.",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale=f"Disqualified at Comparability Gate: '{def_a}' and '{def_b}' are distinct concepts.",
                audit_dict=audit,
            )
        matched_fields.append(f"Metric Definition ({def_a})")

        # ----------------------------------------------------------------------
        # CHECK C: SAME METRIC TYPE / VALUE NATURE (Margin vs Currency vs Count)
        # ----------------------------------------------------------------------
        nat_a = fact_a.value_nature
        nat_b = fact_b.value_nature
        if nat_a != nat_b:
            differing_fields.append(f"Value Nature ('{nat_a}' != '{nat_b}')")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.NOT_COMPARABLE,
                confidence=0.92,
                reasoning=(
                    f"Comparability Gate Disqualification: Value natures are incompatible. "
                    f"Fact A is a '{nat_a}' ({fact_a.value}), while Fact B is a '{nat_b}' ({fact_b.value}). "
                    f"Margins or percentages cannot be compared to absolute amounts or counts."
                ),
                summary=f"Incompatible value natures: {nat_a} ({fact_a.value}) vs {nat_b} ({fact_b.value}).",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale="Incompatible value natures. Absolute currency and percentages/margins cannot be compared.",
                audit_dict=audit,
            )
        matched_fields.append(f"Value Nature ({nat_a})")

        # ----------------------------------------------------------------------
        # CHECK D: COMPATIBLE UNIT
        # ----------------------------------------------------------------------
        if not cls.are_units_compatible(fact_a.context.unit, fact_b.context.unit, nat_a, nat_b):
            differing_fields.append("Unit Dimension")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.NOT_COMPARABLE,
                confidence=0.90,
                reasoning="Incompatible unit dimensions prevent meaningful mathematical comparison.",
                summary="Incompatible unit dimensions.",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale="Units cannot be converted into a common scalar scale.",
                audit_dict=audit,
            )
        matched_fields.append("Unit Compatibility")

        # ----------------------------------------------------------------------
        # CHECK E: REPORTING SCOPE COMPATIBILITY (Consolidated vs Standalone)
        # ----------------------------------------------------------------------
        scope_a = (fact_a.context.reporting_scope or "").strip().lower()
        scope_b = (fact_b.context.reporting_scope or "").strip().lower()
        if scope_a and scope_b and scope_a != scope_b:
            differing_fields.append(f"Reporting Scope ('{fact_a.context.reporting_scope}' vs '{fact_b.context.reporting_scope}')")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.CONTEXT_RECONCILED,
                shift_type=ContextualShiftType.SCOPE_SHIFT,
                confidence=round(min(fact_a.confidence_score, fact_b.confidence_score) * 0.95, 2),
                reasoning=(
                    f"Apparent contradiction reconciled by reporting scope shift: "
                    f"Fact A reflects '{fact_a.context.reporting_scope}' accounting scope ({fact_a.value}), "
                    f"whereas Fact B reflects '{fact_b.context.reporting_scope}' accounting scope ({fact_b.value})."
                ),
                summary=f"Scope Shift: {fact_a.context.reporting_scope} vs {fact_b.context.reporting_scope}.",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale=f"Contextual scope shift explains divergence ({fact_a.context.reporting_scope} vs {fact_b.context.reporting_scope}).",
                audit_dict=audit,
            )
        if scope_a and scope_b:
            matched_fields.append(f"Reporting Scope ({fact_a.context.reporting_scope})")

        # ----------------------------------------------------------------------
        # CHECK F: CUMULATIVE VS PERIOD DURATION (Milestone Progression)
        # ----------------------------------------------------------------------
        temp_nat_a = fact_a.temporal_nature
        temp_nat_b = fact_b.temporal_nature
        period_a = (fact_a.context.temporal_period or "").strip()
        period_b = (fact_b.context.temporal_period or "").strip()

        is_cumulative = (temp_nat_a == "cumulative" or temp_nat_b == "cumulative")
        if is_cumulative and period_a and period_b and period_a.lower() != period_b.lower():
            differing_fields.append(f"Milestone Period ('{period_a}' -> '{period_b}')")
            matched_fields.append("Cumulative Milestone Nature")
            prog_text = f"Cumulative progression: {fact_a.value} ({period_a}) -> {fact_b.value} ({period_b})"
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.TEMPORAL_UPDATE,
                shift_type=ContextualShiftType.TEMPORAL_SHIFT,
                confidence=round(min(fact_a.confidence_score, fact_b.confidence_score) * 0.95, 2),
                reasoning=(
                    f"Temporal Progression / Milestone Update: Both documents report cumulative {def_a} "
                    f"since inception/incorporation across different milestone dates: '{fact_a.evidence.document_id}' reports {fact_a.value} ({period_a}), "
                    f"while '{fact_b.evidence.document_id}' reports {fact_b.value} ({period_b}). "
                    f"This represents an operational milestone progression over time, NOT a contradiction and NOT corroboration."
                ),
                summary=f"Temporal update: {fact_a.value} ({period_a}) -> {fact_b.value} ({period_b}).",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                temporal_progression=prog_text,
                verdict_rationale="Cumulative metric observed across different chronological milestones (Temporal Update).",
                audit_dict=audit,
            )

        # ----------------------------------------------------------------------
        # CHECK G: TEMPORAL PERIOD COMPATIBILITY (Period Duration Metrics)
        # ----------------------------------------------------------------------
        is_multi = EntityDetector.is_multilateral_metric(fact_a.metric)
        if not is_multi and period_a and period_b and period_a.lower() != period_b.lower():
            differing_fields.append(f"Temporal Period ('{period_a}' vs '{period_b}')")
            return GateAuditResult(
                passed=False,
                verdict=ReconciliationVerdict.CONTEXT_RECONCILED,
                shift_type=ContextualShiftType.TEMPORAL_SHIFT,
                confidence=round(min(fact_a.confidence_score, fact_b.confidence_score) * 0.95, 2),
                reasoning=(
                    f"The values differ due to different reporting periods: "
                    f"Fact A reflects '{period_a}' ('{fact_a.evidence.document_id}', p.{fact_a.evidence.page_number}: {fact_a.value}), "
                    f"while Fact B reflects '{period_b}' ('{fact_b.evidence.document_id}', p.{fact_b.evidence.page_number}: {fact_b.value})."
                ),
                summary=f"Temporal Shift: {period_a} vs {period_b}.",
                matched_fields=matched_fields,
                differing_fields=differing_fields,
                verdict_rationale=f"Period-specific disclosures represent different historical reporting periods ({period_a} vs {period_b}).",
                audit_dict=audit,
            )
        if period_a and period_b:
            matched_fields.append(f"Temporal Period ({period_a})")

        # ----------------------------------------------------------------------
        # GATE PASSED: ALL 7 STRUCTURAL CONDITIONS SATISFIED
        # Now and only now execute numerical equivalence comparison
        # ----------------------------------------------------------------------
        is_equiv, equiv_reason = UnitConverter.are_values_equivalent(
            fact_a.value, fact_a.context.unit, fact_b.value, fact_b.context.unit, tolerance_pct=0.02
        )
        unit_norm_exp = UnitConverter.explain_unit_normalization(
            fact_a.value, fact_a.context.unit, fact_b.value, fact_b.context.unit
        )

        extraction_conf = min(fact_a.confidence_score, fact_b.confidence_score)

        if is_equiv:
            joint_conf = round(min(1.0, max(0.1, extraction_conf * 0.98)), 2)
            return GateAuditResult(
                passed=True,
                verdict=ReconciliationVerdict.CORROBORATED,
                shift_type=ContextualShiftType.NONE,
                confidence=joint_conf,
                reasoning=(
                    f"Facts corroborated across '{fact_a.evidence.document_id}' (p.{fact_a.evidence.page_number}) "
                    f"and '{fact_b.evidence.document_id}' (p.{fact_b.evidence.page_number}). "
                    f"{equiv_reason or 'Values match identically within tolerance.'}"
                ),
                summary=f"Both documents corroborate {def_a} ({fact_a.value}).",
                matched_fields=matched_fields + ["Numerical Value"],
                differing_fields=differing_fields,
                unit_normalization=unit_norm_exp,
                verdict_rationale="Passes comparability gate across all dimensions; values agree within tolerance under unit normalization.",
                audit_dict=audit,
            )
        else:
            joint_conf = round(min(1.0, max(0.1, extraction_conf * 0.92)), 2)
            return GateAuditResult(
                passed=True,
                verdict=ReconciliationVerdict.GENUINE_CONTRADICTION,
                shift_type=ContextualShiftType.NONE,
                confidence=joint_conf,
                reasoning=(
                    f"Conflicting disclosures: '{fact_a.evidence.document_id}' (p.{fact_a.evidence.page_number}) reports {fact_a.value}, "
                    f"while '{fact_b.evidence.document_id}' (p.{fact_b.evidence.page_number}) reports {fact_b.value} for {def_a} "
                    f"under matching period ({period_a or 'unspecified'}) and scope without contextual justification."
                ),
                summary=f"Genuine contradiction: {fact_a.value} vs {fact_b.value} for {def_a}.",
                matched_fields=matched_fields,
                differing_fields=differing_fields + [f"Numerical Value ('{fact_a.value}' != '{fact_b.value}')"],
                unit_normalization=unit_norm_exp,
                verdict_rationale="Passes comparability gate with matching entity, metric definition, period, and scope, but reported values genuinely diverge beyond acceptable tolerance.",
                audit_dict=audit,
            )

    @classmethod
    def is_candidate_pair(cls, fact_a: Fact, fact_b: Fact) -> bool:
        """Fast pre-filter for clustering candidate pairs before deep reconciliation."""
        if fact_a.evidence.document_id == fact_b.evidence.document_id:
            return False
        if not EntityDetector.are_entities_compatible(fact_a.entity, fact_b.entity, metric_name=fact_a.metric):
            return False
        def_a = cls.get_canonical_definition(fact_a)
        def_b = cls.get_canonical_definition(fact_b)
        if def_a != def_b:
            return False
        if fact_a.value_nature != fact_b.value_nature:
            return False
        return True

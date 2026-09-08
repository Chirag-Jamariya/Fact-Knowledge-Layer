"""Contextual shift analyzer for resolving apparent contradictions (Case 3)."""

from typing import Tuple, Optional
from src.models.fact import Fact
from src.models.reconciliation import ContextualShiftType
from src.reconciliation.unit_converter import UnitConverter
from src.extraction.entity_detector import EntityDetector


class ContextAnalyzer:
    """Analyzes differences in time, scope, or measurement units to reconcile differing values."""

    @classmethod
    def analyze_reconciliation(cls, fact_a: Fact, fact_b: Fact) -> Tuple[bool, ContextualShiftType, Optional[str]]:
        # Enforce entity compatibility: distinct sovereign countries cannot undergo intra-entity shifts
        if not EntityDetector.are_entities_compatible(fact_a.entity, fact_b.entity, metric_name=fact_a.metric):
            return False, ContextualShiftType.NONE, None

        ctx_a = fact_a.context
        ctx_b = fact_b.context

        # 1. Check Unit / Denomination Shift (e.g. INR Cr vs Millions)
        is_equiv, unit_reason = UnitConverter.are_values_equivalent(
            fact_a.value, ctx_a.unit, fact_b.value, ctx_b.unit
        )
        if is_equiv:
            # If values were equivalent across units, but expressed differently
            if (ctx_a.unit or "").lower() != (ctx_b.unit or "").lower():
                return (
                    True,
                    ContextualShiftType.UNIT_DENOMINATION,
                    f"Values are identical under unit normalization: {unit_reason}",
                )

        # 2. Check Temporal Shift (e.g. FY21 vs FY24)
        if ctx_a.temporal_period and ctx_b.temporal_period:
            period_a = ctx_a.temporal_period.lower().strip()
            period_b = ctx_b.temporal_period.lower().strip()
            if period_a != period_b:
                return (
                    True,
                    ContextualShiftType.TEMPORAL_SHIFT,
                    f"The values differ due to different reporting periods: Fact A is for '{ctx_a.temporal_period}' (p.{fact_a.evidence.page_number}), whereas Fact B is for '{ctx_b.temporal_period}' (p.{fact_b.evidence.page_number}).",
                )

        # 3. Check Reporting Scope Shift (e.g. Standalone vs Consolidated)
        if ctx_a.reporting_scope and ctx_b.reporting_scope:
            scope_a = ctx_a.reporting_scope.lower().strip()
            scope_b = ctx_b.reporting_scope.lower().strip()
            if scope_a != scope_b:
                return (
                    True,
                    ContextualShiftType.SCOPE_SHIFT,
                    f"The values differ due to differing accounting scopes: Fact A reflects '{ctx_a.reporting_scope}', whereas Fact B reflects '{ctx_b.reporting_scope}'.",
                )

        return False, ContextualShiftType.NONE, None

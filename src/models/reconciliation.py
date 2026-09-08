"""Cross-Document Reconciliation data models representing the 4 assignment cases."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import hashlib
from src.models.fact import Fact


class ReconciliationVerdict(str, Enum):
    """The cross-document reconciliation taxonomy."""
    CORROBORATED = "corroborated"
    GENUINE_CONTRADICTION = "genuine_contradiction"
    TEMPORAL_UPDATE = "temporal_update"
    CONTEXT_RECONCILED = "context_reconciled"
    METRIC_MISMATCH = "metric_mismatch"
    NOT_COMPARABLE = "not_comparable"
    EXTRACTION_FAILURE = "extraction_failure"
    REASONING_FAILURE = "reasoning_failure"


class ContextualShiftType(str, Enum):
    """The contextual dimension explaining why an apparent contradiction exists."""
    TEMPORAL_SHIFT = "temporal_shift"          # e.g., FY22 vs FY24, or Q3 vs Q4
    SCOPE_SHIFT = "scope_shift"                # e.g., Standalone vs Consolidated
    UNIT_DENOMINATION = "unit_denomination"    # e.g., Crores vs Millions, USD vs INR
    METHODOLOGY_VINTAGE = "methodology_vintage"# e.g., Revised Estimate vs Actual
    NONE = "none"


class CrossDocReconciliation(BaseModel):
    """Pairwise cross-document comparison verdict with evidence and reasoning."""
    reconciliation_id: str = Field(default="", description="Deterministic unique identifier")
    verdict: ReconciliationVerdict
    shift_type: ContextualShiftType = Field(default=ContextualShiftType.NONE)
    fact_a: Fact = Field(description="First fact from Document A")
    fact_b: Fact = Field(description="Second fact from Document B")
    fact_a_id: str = Field(default="", description="Fact A unique ID reference")
    fact_b_id: str = Field(default="", description="Fact B unique ID reference")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    reasoning: str = Field(description="System's chain-of-thought explaining why facts agree, conflict, or reconcile")
    reconciliation_summary: str = Field(description="Concise 1-sentence executive summary")
    corpus: str = Field(default="", description="Logical corpus name ('India Macroeconomics', 'Delhivery')")
    participating_facts: List[Fact] = Field(
        default_factory=list,
        description="All participating facts across the corpus documents (Fact A from File 1, Fact B from File 2, Fact C from File 3...)"
    )
    comparability_gate_passed: bool = Field(default=True, description="Whether the pair passed all structural comparability checks")
    comparability_audit: Dict[str, Any] = Field(default_factory=dict, description="Detailed field-by-field comparability audit")
    unit_normalization_applied: Optional[str] = Field(default=None, description="Explanation of unit conversion or rounding applied")
    temporal_progression_explanation: Optional[str] = Field(default=None, description="Explanation of chronological or cumulative milestone progression")
    verdict_rationale: Optional[str] = Field(default=None, description="Explicit justification for why the final verdict was selected")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.participating_facts and self.fact_a and self.fact_b:
            self.participating_facts = [self.fact_a, self.fact_b]
        if not self.fact_a_id and self.fact_a:
            self.fact_a_id = self.fact_a.fact_id
        if not self.fact_b_id and self.fact_b:
            self.fact_b_id = self.fact_b.fact_id
        if not self.reconciliation_id:
            key = f"{self.fact_a_id}:{self.fact_b_id}:{self.verdict.value}"
            self.reconciliation_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        if not self.corpus and self.fact_a:
            self.corpus = getattr(self.fact_a, "corpus", "") or (
                "Delhivery" if "delhivery" in (self.fact_a.entity or "").lower() else "India Macroeconomics"
            )


class FailureType(str, Enum):
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    REASONING_ERROR = "REASONING_ERROR"
    QUALIFIER_DROPPED = "QUALIFIER_DROPPED"
    BOUNDARY_ERROR = "BOUNDARY_ERROR"
    LAYOUT_FLATTENING = "LAYOUT_FLATTENING"
    OCR_CORRUPTION = "OCR_CORRUPTION"
    EVIDENCE_MISSING = "EVIDENCE_MISSING"


class FailureRecord(BaseModel):
    """Detailed diagnostic record of an extraction or cross-document reasoning failure (Case 4)."""
    failure_id: str = Field(default="", description="Unique identifier for failure event")
    failure_type: FailureType
    source_document: str = Field(description="Source PDF filename")
    page_number: int = Field(ge=1, description="Page number where failure originated")
    expected_evidence: str = Field(description="Verbatim or expected text from original document")
    observed_result: str = Field(description="Flawed extraction or reasoning output")
    problem_description: str = Field(description="Detailed technical breakdown of why the error occurred")
    impact: str = Field(description="Downstream effect on fact store or cross-document reconciliation")
    detection_mechanism: str = Field(description="Method used to detect or diagnose this failure")
    mitigation_strategy: str = Field(description="Rule or system mitigation deployed to prevent or correct it")
    potential_improvement: str = Field(description="Recommended architectural or model enhancement")
    corpus: str = Field(default="", description="Logical corpus name ('India Macroeconomics', 'Delhivery')")
    fact_id: Optional[str] = None
    related_fact_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.failure_id:
            key = f"{self.source_document}:{self.page_number}:{self.failure_type.value}:{self.observed_result[:30]}"
            self.failure_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        if not self.corpus and self.source_document:
            self.corpus = "Delhivery" if "delhivery" in self.source_document.lower() else "India Macroeconomics"


class CorpusConsensusGroup(BaseModel):
    """Multi-document consensus analysis for a metric across an entire logical corpus."""
    group_id: str = Field(default="", description="Unique identifier for consensus group")
    corpus: str = Field(description="Logical corpus name (e.g. 'India Macroeconomics', 'Delhivery')")
    metric: str = Field(description="Metric being compared across the corpus documents")
    entity: str = Field(description="Primary entity name")
    total_corpus_docs: int = Field(ge=1, description="Total documents in this logical corpus")
    participating_docs: List[str] = Field(description="Filenames of documents disclosing this metric")
    doc_count: int = Field(ge=1, description="Number of participating documents in this consensus group")
    coverage_ratio: str = Field(description="e.g. '3 of 3 PDFs' or '2 of 3 PDFs'")
    verdict: ReconciliationVerdict = Field(description="Corpus-wide consensus verdict")
    status_summary: str = Field(description="High-level status (e.g., 'Supported across all 3 PDFs')")
    synthesis: str = Field(description="Narrative explanation synthesizing evidence across all participating documents")
    disclosures: List[Dict[str, Any]] = Field(default_factory=list, description="List of per-document disclosure items")
    facts: List[Fact] = Field(default_factory=list, description="All participating fact objects")

    def model_post_init(self, __context: Any) -> None:
        if not self.group_id:
            key = f"{self.corpus}:{self.metric}:{self.doc_count}"
            self.group_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class CaseShowcase(BaseModel):
    """Container explicitly grouping the 4 required cases for evaluation."""
    corroborated_example: Optional[CrossDocReconciliation] = None
    contradiction_example: Optional[CrossDocReconciliation] = None
    context_reconciled_example: Optional[CrossDocReconciliation] = None
    failure_example: Optional[CrossDocReconciliation] = None
    failure_record: Optional[FailureRecord] = None

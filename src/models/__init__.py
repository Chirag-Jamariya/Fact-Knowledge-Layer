"""Data models package for Fact Knowledge Layer."""

from src.models.corpus import (
    ExtractedTable,
    ExtractedImage,
    PageContent,
    DocumentRecord,
)
from src.models.fact import (
    FactCategory,
    SourceEvidence,
    ContextDimensions,
    Fact,
)
from src.models.reconciliation import (
    ReconciliationVerdict,
    ContextualShiftType,
    CrossDocReconciliation,
    CaseShowcase,
    CorpusConsensusGroup,
    FailureType,
    FailureRecord,
)

__all__ = [
    "ExtractedTable",
    "ExtractedImage",
    "PageContent",
    "DocumentRecord",
    "FactCategory",
    "SourceEvidence",
    "ContextDimensions",
    "Fact",
    "ReconciliationVerdict",
    "ContextualShiftType",
    "CrossDocReconciliation",
    "CaseShowcase",
    "CorpusConsensusGroup",
    "FailureType",
    "FailureRecord",
]

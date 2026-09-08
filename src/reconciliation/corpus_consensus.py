"""Corpus-Wide Multi-Document Verification and Consensus Engine."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.models.fact import Fact
from src.models.reconciliation import (
    CorpusConsensusGroup,
    ReconciliationVerdict,
    ContextualShiftType,
)
from src.reconciliation.unit_converter import UnitConverter
from src.reconciliation.context_analyzer import ContextAnalyzer
from src.reconciliation.comparability_gate import ComparabilityGate

logger = logging.getLogger(__name__)


class CorpusConsensusEngine:
    """Aggregates facts across all documents in a logical corpus into unified multi-document consensus groups."""

    @classmethod
    def build_consensus_groups(
        cls,
        facts: List[Fact],
        corpus_name: str,
        total_corpus_docs: Optional[int] = None,
    ) -> List[CorpusConsensusGroup]:
        """
        Group facts by canonical metric across distinct documents within the same corpus
        and determine corpus-wide agreement, contextual reconciliation, or discrepancy.
        """
        metric_buckets: Dict[str, Dict[str, List[Fact]]] = defaultdict(lambda: defaultdict(list))
        for f in facts:
            clean_metric = ComparabilityGate.get_canonical_definition(f)
            metric_buckets[clean_metric][f.evidence.document_id].append(f)

        all_docs_in_corpus = set(f.evidence.document_id for f in facts)
        total_docs = total_corpus_docs or max(len(all_docs_in_corpus), 1)

        consensus_groups: List[CorpusConsensusGroup] = []

        for metric_name, docs_dict in metric_buckets.items():
            # Only consider metrics appearing in >= 2 distinct documents
            if len(docs_dict) < 2:
                continue

            participating_docs = sorted(docs_dict.keys())
            doc_count = len(participating_docs)
            coverage_ratio = f"{doc_count} of {total_docs} PDFs"

            group_facts: List[Fact] = []
            rep_facts_per_doc: List[Fact] = []
            for d in participating_docs:
                doc_facts = docs_dict[d]
                group_facts.extend(doc_facts)
                best_f = max(doc_facts, key=lambda x: x.confidence_score)
                rep_facts_per_doc.append(best_f)

            entity_name = rep_facts_per_doc[0].entity

            verdict, status_summary, synthesis = cls._evaluate_multi_doc_status(
                metric_name=metric_name,
                corpus_name=corpus_name,
                rep_facts=rep_facts_per_doc,
                doc_count=doc_count,
                total_docs=total_docs,
            )

            disclosures = []
            for f in rep_facts_per_doc:
                disclosures.append({
                    "document_id": f.evidence.document_id,
                    "page_number": f.evidence.page_number,
                    "value": f.value,
                    "raw_value": f.raw_value,
                    "temporal_period": f.context.temporal_period or "Unspecified",
                    "reporting_scope": f.context.reporting_scope or "Standard",
                    "unit": f.context.unit or "",
                    "verbatim_quote": f.evidence.verbatim_quote,
                })

            group = CorpusConsensusGroup(
                corpus=corpus_name,
                metric=metric_name,
                entity=entity_name,
                total_corpus_docs=total_docs,
                participating_docs=participating_docs,
                doc_count=doc_count,
                coverage_ratio=coverage_ratio,
                verdict=verdict,
                status_summary=status_summary,
                synthesis=synthesis,
                disclosures=disclosures,
                facts=group_facts,
            )
            consensus_groups.append(group)

        # Sort: 3-of-3 PDFs first (highest consensus coverage), then alphabetical
        consensus_groups.sort(key=lambda g: (-g.doc_count, g.metric))
        return consensus_groups

    @classmethod
    def _evaluate_multi_doc_status(
        cls,
        metric_name: str,
        corpus_name: str,
        rep_facts: List[Fact],
        doc_count: int,
        total_docs: int,
    ) -> Tuple[ReconciliationVerdict, str, str]:
        """Determine corpus-level consensus verdict and synthesize narrative across documents."""
        is_full_coverage = (doc_count >= total_docs)
        docs_text = f"all {doc_count} PDFs" if is_full_coverage else f"{doc_count} of {total_docs} PDFs"

        # Evaluate pairwise status using ComparabilityGate
        all_corroborated = True
        any_temporal_update = False
        any_context_reconciled = False

        for i in range(len(rep_facts)):
            for j in range(i + 1, len(rep_facts)):
                res = ComparabilityGate.evaluate(rep_facts[i], rep_facts[j])
                if res.verdict == ReconciliationVerdict.TEMPORAL_UPDATE:
                    any_temporal_update = True
                    all_corroborated = False
                elif res.verdict == ReconciliationVerdict.CONTEXT_RECONCILED:
                    any_context_reconciled = True
                    all_corroborated = False
                elif res.verdict != ReconciliationVerdict.CORROBORATED:
                    all_corroborated = False

        evidence_lines = []
        for f in rep_facts:
            period_str = f", period: {f.context.temporal_period}" if f.context.temporal_period else ""
            evidence_lines.append(f"• {f.evidence.document_id} (p.{f.evidence.page_number}{period_str}): {f.value}")
        evidence_bullet_block = "\n".join(evidence_lines)

        if all_corroborated:
            verdict = ReconciliationVerdict.CORROBORATED
            status_summary = f"Corroborated across {docs_text} in {corpus_name} ({rep_facts[0].value})"
            synthesis = (
                f"Multi-document consensus across {docs_text}:\n"
                f"{evidence_bullet_block}\n"
                f"Disclosures across all {doc_count} documents corroborate the same numerical measurement within tolerance."
            )
            return verdict, status_summary, synthesis

        if any_temporal_update:
            verdict = ReconciliationVerdict.TEMPORAL_UPDATE
            status_summary = f"Temporal Progression across {docs_text} in {corpus_name}"
            synthesis = (
                f"Multi-document milestone verification across {docs_text}:\n"
                f"{evidence_bullet_block}\n"
                f"Disclosures reflect chronological milestone progression over time for cumulative metric '{metric_name}'."
            )
            return verdict, status_summary, synthesis

        if any_context_reconciled:
            verdict = ReconciliationVerdict.CONTEXT_RECONCILED
            status_summary = f"Contextually Reconciled across {docs_text} in {corpus_name}"
            synthesis = (
                f"Multi-document verification across {docs_text}:\n"
                f"{evidence_bullet_block}\n"
                f"Apparent variance reconciled by contextual dimensions (reporting periods or scopes). The documents depict complementary facets of the same underlying metric."
            )
            return verdict, status_summary, synthesis

        verdict = ReconciliationVerdict.GENUINE_CONTRADICTION
        status_summary = f"Discrepancy observed across {docs_text} in {corpus_name}"
        synthesis = (
            f"Cross-document discrepancy identified across {docs_text}:\n"
            f"{evidence_bullet_block}\n"
            f"Diverging measurements reported without explicit reconciling footnotes."
        )
        return verdict, status_summary, synthesis

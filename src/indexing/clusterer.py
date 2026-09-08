"""Cross-document candidate fact clusterer."""

from typing import List, Tuple, Dict, Optional
from itertools import combinations
from src.models.fact import Fact
from src.extraction.entity_detector import EntityDetector
from src.reconciliation.comparability_gate import ComparabilityGate


class CrossDocClusterer:
    """Finds and pairs candidate facts from distinct documents that refer to the same metric/entity."""

    @staticmethod
    def are_entities_compatible(ent_a: str, ent_b: str, metric_name: Optional[str] = None) -> bool:
        """Check if two entity labels are compatible or refer to common multilateral metrics."""
        return EntityDetector.are_entities_compatible(ent_a, ent_b, metric_name=metric_name)

    @staticmethod
    def detect_corpus(doc_id: str, entity: str = "") -> str:
        """Resolve logical corpus from document ID and entity."""
        d = (doc_id or "").lower()
        e = (entity or "").lower()
        if "delhivery" in d or "delhivery" in e:
            return "Delhivery"
        if any(k in d or k in e for k in ["india", "rbi", "economic survey", "imf india", "indian economy"]):
            return "India Macroeconomics"
        if entity and entity.strip():
            return entity.strip()
        return "General Knowledge"

    @classmethod
    def find_cross_document_pairs(
        cls, facts: List[Fact], corpus: Optional[str] = None, relax_entity: bool = False, *args, **kwargs
    ) -> List[Tuple[Fact, Fact]]:
        """
        Identify pairs of facts across DIFFERENT documents within the SAME logical corpus that share the same metric.
        Different logical corpora (e.g. Delhivery vs India Macroeconomics) and distinct sovereign states are
        strictly isolated and never compared.
        Returns candidate pairs (fact_a, fact_b).
        """
        if corpus:
            facts = [
                f for f in facts
                if (
                    (getattr(f, "corpus", None) and f.corpus.strip().lower() == corpus.strip().lower())
                    or cls.detect_corpus(f.evidence.document_id, f.entity).lower() == corpus.strip().lower()
                )
            ]

        # Partition strictly by logical corpus
        corpus_buckets: Dict[str, List[Fact]] = {}
        for f in facts:
            c_name = getattr(f, "corpus", "") or cls.detect_corpus(f.evidence.document_id, f.entity)
            corpus_buckets.setdefault(c_name, []).append(f)

        candidate_pairs: List[Tuple[Fact, Fact]] = []
        for c_name, c_facts in corpus_buckets.items():
            metric_buckets: Dict[Tuple[str, str], List[Fact]] = {}
            for f in c_facts:
                canon_def = ComparabilityGate.get_canonical_definition(f).lower()
                nature = f.value_nature
                metric_buckets.setdefault((canon_def, nature), []).append(f)

            for (metric, nature), bucket_facts in metric_buckets.items():
                for fa, fb in combinations(bucket_facts, 2):
                    if fa.evidence.document_id != fb.evidence.document_id:
                        if ComparabilityGate.is_candidate_pair(fa, fb):
                            candidate_pairs.append((fa, fb))

        return candidate_pairs

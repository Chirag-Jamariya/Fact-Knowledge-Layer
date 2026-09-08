"""Knowledge graph synchronization connecting Fact Store, Relationship Store, and Failure Store with strict corpus boundary isolation."""

import logging
from typing import Dict, Any, Optional
from src.indexing.store import KnowledgeStore
from src.indexing.clusterer import CrossDocClusterer
from src.reconciliation.reconciler import CrossDocReconciler
from src.reconciliation.failure_detector import FailureDetector
from src.models.reconciliation import ReconciliationVerdict

logger = logging.getLogger(__name__)


def sync_knowledge_graph(store: Optional[KnowledgeStore] = None) -> Dict[str, Any]:
    """
    Synchronize the knowledge graph across facts partitioned strictly by logical corpus:
    1. Group facts by logical corpus ('India Macroeconomics', 'Delhivery').
    2. For each corpus, find cross-document pairs within that corpus only.
    3. Evaluate cross-document relationships (Cases 1, 2, 3) with zero cross-corpus leakage.
    4. Audit each corpus for extraction and reasoning failures (Case 4).
    5. Commit cleanly to DuckDB stores.
    """
    if store is None:
        store = KnowledgeStore()

    all_facts = store.get_all_facts()
    if not all_facts:
        return {
            "total_facts": 0,
            "total_relationships": 0,
            "corroborated": 0,
            "genuine_contradiction": 0,
            "context_reconciled": 0,
            "total_failures": 0,
        }

    # Clear existing relationships and failures to re-populate cleanly
    store.conn.execute("DELETE FROM relationships;")
    store.conn.execute("DELETE FROM failures;")

    corpora = store.get_available_corpora()
    all_relationships = []
    reconciler = CrossDocReconciler()

    # Map documents to their corpus
    doc_to_corpus = {}
    try:
        doc_rows = store.conn.execute("SELECT doc_id, filename, corpus FROM documents;").fetchall()
        for r in doc_rows:
            if r[2]:
                doc_to_corpus[r[0].lower()] = r[2]
                doc_to_corpus[r[1].lower()] = r[2]
    except Exception:
        pass

    # Reconcile strictly within each logical corpus
    for corpus_name in corpora:
        corpus_facts = [
            f for f in all_facts
            if (
                (getattr(f, "corpus", None) and f.corpus.strip().lower() == corpus_name.strip().lower())
                or (not getattr(f, "corpus", None) and doc_to_corpus.get(f.evidence.document_id.lower(), "").lower() == corpus_name.lower())
                or (not getattr(f, "corpus", None) and "delhivery" in f.entity.lower() and corpus_name.lower() == "delhivery")
                or (not getattr(f, "corpus", None) and any(k in f.entity.lower() for k in ["india", "rbi", "economic survey", "imf india"]) and corpus_name.lower() == "india macroeconomics")
            )
        ]
        if not corpus_facts:
            continue

        # Pair facts exclusively within this corpus
        pairs = CrossDocClusterer.find_cross_document_pairs(corpus_facts, corpus=corpus_name)
        reconciliations = reconciler.reconcile_all(pairs)

        rel_in_corpus = [r for r in reconciliations if r.verdict != ReconciliationVerdict.EXTRACTION_FAILURE]
        for r in rel_in_corpus:
            r.corpus = corpus_name
        all_relationships.extend(rel_in_corpus)

    store.insert_relationships(all_relationships)

    # Audit for failures
    failures = FailureDetector.audit_corpus(all_facts)
    for fail in failures:
        src = (fail.source_document or "").lower()
        if src in doc_to_corpus:
            fail.corpus = doc_to_corpus[src]
        elif any(k in src for k in doc_to_corpus):
            for k, c in doc_to_corpus.items():
                if k in src or src in k:
                    fail.corpus = c
                    break
        else:
            fail.corpus = "Delhivery" if "delhivery" in src else ("India Macroeconomics" if any(k in src for k in ["india", "rbi", "economic survey", "imf"]) else "General Knowledge")
    store.insert_failures(failures)

    counts = store.get_relationship_counts()
    failure_count = store.get_failure_count()

    logger.info(
        f"Knowledge graph synchronized across {len(corpora)} corpora: "
        f"{len(all_facts)} facts, {len(all_relationships)} relationships, "
        f"{failure_count} failures."
    )

    return {
        "total_facts": len(all_facts),
        "total_relationships": len(all_relationships),
        "corroborated": counts.get("corroborated", 0),
        "genuine_contradiction": counts.get("genuine_contradiction", 0),
        "context_reconciled": counts.get("context_reconciled", 0),
        "total_failures": failure_count,
    }

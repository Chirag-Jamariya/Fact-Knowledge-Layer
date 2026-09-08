"""In-Corpus DuckDB Knowledge Store for Facts and Documents."""

import os
import json
import logging
from typing import List, Optional, Dict, Any
import duckdb

from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.models.corpus import DocumentRecord
from src.models.reconciliation import (
    CrossDocReconciliation,
    ReconciliationVerdict,
    ContextualShiftType,
    FailureRecord,
    FailureType,
)
from src.config import settings

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Persistent embedded DuckDB store for documents, facts, and relationships."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(settings.knowledge_store_path, "knowledge.duckdb")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            self.conn = duckdb.connect(self.db_path)
            self._init_schema()
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                try:
                    self.conn = duckdb.connect(self.db_path, read_only=True)
                    logger.info(f"DuckDB database opened in read-only mode for {self.db_path}.")
                except Exception:
                    logger.warning(
                        f"DuckDB database file {self.db_path} is locked by another running process (e.g. Streamlit). "
                        "Falling back to an isolated in-memory knowledge store for this process."
                    )
                    self.conn = duckdb.connect(":memory:")
                    self._init_schema()
            else:
                raise

    def _init_schema(self) -> None:
        """Create relational tables for documents, facts, relationships, and failures with corpus isolation."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                file_path VARCHAR NOT NULL,
                total_pages INTEGER NOT NULL,
                checksum_sha256 VARCHAR NOT NULL,
                ingested_at TIMESTAMP NOT NULL,
                metadata JSON,
                corpus VARCHAR
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id VARCHAR PRIMARY KEY,
                doc_id VARCHAR NOT NULL,
                page_number INTEGER NOT NULL,
                category VARCHAR NOT NULL,
                entity VARCHAR NOT NULL,
                metric VARCHAR NOT NULL,
                value VARCHAR NOT NULL,
                raw_value VARCHAR NOT NULL,
                temporal_period VARCHAR,
                reporting_scope VARCHAR,
                unit VARCHAR,
                accounting_standard VARCHAR,
                verbatim_quote TEXT NOT NULL,
                section_title VARCHAR,
                confidence_score DOUBLE NOT NULL,
                corpus VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id VARCHAR PRIMARY KEY,
                fact_a_id VARCHAR NOT NULL,
                fact_b_id VARCHAR NOT NULL,
                verdict VARCHAR NOT NULL,
                shift_type VARCHAR NOT NULL,
                confidence DOUBLE NOT NULL,
                reasoning TEXT NOT NULL,
                summary TEXT NOT NULL,
                corpus VARCHAR,
                details JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                failure_id VARCHAR PRIMARY KEY,
                failure_type VARCHAR NOT NULL,
                source_document VARCHAR NOT NULL,
                page_number INTEGER NOT NULL,
                expected_evidence TEXT NOT NULL,
                observed_result TEXT NOT NULL,
                problem_description TEXT NOT NULL,
                impact TEXT NOT NULL,
                detection_mechanism TEXT NOT NULL,
                mitigation_strategy TEXT NOT NULL,
                potential_improvement TEXT NOT NULL,
                corpus VARCHAR,
                fact_id VARCHAR,
                related_fact_id VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Drop legacy ART secondary indexes to prevent DuckDB index deletion collisions on REPLACE
        self.conn.execute("DROP INDEX IF EXISTS idx_facts_entity;")
        self.conn.execute("DROP INDEX IF EXISTS idx_facts_metric;")
        self.conn.execute("DROP INDEX IF EXISTS idx_facts_doc_id;")

        # Migrate existing tables if missing corpus column
        for tbl in ["documents", "facts", "relationships", "failures"]:
            try:
                self.conn.execute(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS corpus VARCHAR;")
            except Exception:
                pass

        # Backfill default corpus for starter datasets if any are NULL
        try:
            self.conn.execute("""
                UPDATE documents SET corpus = CASE WHEN LOWER(filename) LIKE '%delhivery%' THEN 'Delhivery' ELSE 'India Macroeconomics' END WHERE corpus IS NULL;
                UPDATE facts SET corpus = CASE WHEN LOWER(entity) LIKE '%delhivery%' THEN 'Delhivery' ELSE 'India Macroeconomics' END WHERE corpus IS NULL;
                UPDATE relationships SET corpus = CASE WHEN LOWER(summary) LIKE '%delhivery%' OR LOWER(reasoning) LIKE '%delhivery%' THEN 'Delhivery' ELSE 'India Macroeconomics' END WHERE corpus IS NULL;
                UPDATE failures SET corpus = CASE WHEN LOWER(source_document) LIKE '%delhivery%' THEN 'Delhivery' ELSE 'India Macroeconomics' END WHERE corpus IS NULL;
            """)
        except Exception:
            pass

    def insert_document(self, doc: DocumentRecord, corpus: Optional[str] = None) -> None:
        """Insert or replace document record with corpus classification."""
        corpus_val = (
            corpus
            or getattr(doc, "corpus", None)
            or doc.metadata.get("corpus")
            or doc.metadata.get("corpus_label")
            or ("Delhivery" if "delhivery" in doc.filename.lower() else ("India Macroeconomics" if any(k in doc.filename.lower() for k in ["india", "rbi", "economic survey", "imf"]) else "General Knowledge"))
        )
        doc.corpus = corpus_val
        doc.metadata["corpus"] = corpus_val
        self.conn.execute("""
            INSERT OR REPLACE INTO documents (doc_id, filename, file_path, total_pages, checksum_sha256, ingested_at, metadata, corpus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            doc.doc_id,
            doc.filename,
            doc.file_path,
            doc.total_pages,
            doc.checksum_sha256,
            doc.ingested_at,
            json.dumps(doc.metadata),
            corpus_val,
        ])

    def insert_facts(self, facts: List[Fact], doc_id: str, corpus: Optional[str] = None) -> int:
        """Insert a batch of facts linked to a document ID with logical corpus."""
        # Remove prior facts for this doc to keep dataset clean and avoid index collision
        self.conn.execute("DELETE FROM facts WHERE doc_id = ?;", [doc_id])
        inserted_count = 0
        doc_corpus = None
        if not corpus:
            try:
                row = self.conn.execute("SELECT corpus FROM documents WHERE doc_id = ?;", [doc_id]).fetchone()
                if row and row[0]:
                    doc_corpus = row[0]
            except Exception:
                pass

        for f in facts:
            corpus_val = (
                corpus
                or getattr(f, "corpus", None)
                or doc_corpus
                or ("Delhivery" if "delhivery" in (f.entity or "").lower() else ("India Macroeconomics" if any(k in (f.entity or "").lower() for k in ["india", "rbi", "economic survey", "imf"]) else (f.entity or "General Knowledge")))
            )
            f.corpus = corpus_val
            self.conn.execute("""
                INSERT OR REPLACE INTO facts (
                    fact_id, doc_id, page_number, category, entity, metric, value, raw_value,
                    temporal_period, reporting_scope, unit, accounting_standard,
                    verbatim_quote, section_title, confidence_score, corpus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.fact_id,
                doc_id,
                f.evidence.page_number,
                f.category.value,
                f.entity,
                f.metric,
                f.value,
                f.raw_value,
                f.context.temporal_period,
                f.context.reporting_scope,
                f.context.unit,
                f.context.accounting_standard,
                f.evidence.verbatim_quote,
                f.evidence.section_title,
                f.confidence_score,
                corpus_val,
            ])
            inserted_count += 1
        return inserted_count

    def get_all_facts(self, limit: Optional[int] = None, corpus: Optional[str] = None) -> List[Fact]:
        """Retrieve all stored facts, optionally filtered by logical corpus."""
        query = """
            SELECT f.fact_id, COALESCE(d.filename, f.doc_id), f.page_number, f.category, f.entity, f.metric,
                   f.value, f.raw_value, f.temporal_period, f.reporting_scope, f.unit, f.accounting_standard,
                   f.verbatim_quote, f.section_title, f.confidence_score, f.corpus
            FROM facts f
            LEFT JOIN documents d ON f.doc_id = d.doc_id
        """
        params = []
        if corpus:
            query += " WHERE LOWER(f.corpus) = LOWER(?)"
            params.append(corpus)
        query += " ORDER BY d.filename, f.page_number"
        if limit:
            query += f" LIMIT {limit}"
        return self._rows_to_facts(self.conn.execute(query, params).fetchall())

    def get_facts_by_entity(self, entity: str, corpus: Optional[str] = None) -> List[Fact]:
        """Query facts by entity name, optionally within a corpus."""
        query = """
            SELECT f.fact_id, COALESCE(d.filename, f.doc_id), f.page_number, f.category, f.entity, f.metric,
                   f.value, f.raw_value, f.temporal_period, f.reporting_scope, f.unit, f.accounting_standard,
                   f.verbatim_quote, f.section_title, f.confidence_score, f.corpus
            FROM facts f
            LEFT JOIN documents d ON f.doc_id = d.doc_id
            WHERE LOWER(f.entity) LIKE LOWER(?)
        """
        params = [f"%{entity}%"]
        if corpus:
            query += " AND LOWER(f.corpus) = LOWER(?)"
            params.append(corpus)
        query += " ORDER BY f.metric, f.page_number"
        return self._rows_to_facts(self.conn.execute(query, params).fetchall())

    def get_facts_by_metric(self, metric: str, corpus: Optional[str] = None) -> List[Fact]:
        """Query facts matching metric name, optionally within a corpus."""
        query = """
            SELECT f.fact_id, COALESCE(d.filename, f.doc_id), f.page_number, f.category, f.entity, f.metric,
                   f.value, f.raw_value, f.temporal_period, f.reporting_scope, f.unit, f.accounting_standard,
                   f.verbatim_quote, f.section_title, f.confidence_score, f.corpus
            FROM facts f
            LEFT JOIN documents d ON f.doc_id = d.doc_id
            WHERE LOWER(f.metric) LIKE LOWER(?)
        """
        params = [f"%{metric}%"]
        if corpus:
            query += " AND LOWER(f.corpus) = LOWER(?)"
            params.append(corpus)
        query += " ORDER BY d.filename, f.page_number"
        return self._rows_to_facts(self.conn.execute(query, params).fetchall())

    def get_cross_document_metric_groups(self, corpus: Optional[str] = None) -> Dict[str, List[Fact]]:
        """
        Group facts by canonical metric where the metric appears in MORE THAN ONE document within the same corpus.
        """
        query = """
            SELECT metric
            FROM facts
        """
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        query += " GROUP BY metric HAVING COUNT(DISTINCT doc_id) > 1;"
        
        metrics = [row[0] for row in self.conn.execute(query, params).fetchall()]
        groups: Dict[str, List[Fact]] = {}
        for m in metrics:
            groups[m] = self.get_facts_by_metric(m, corpus=corpus)
        return groups

    def get_document_count(self, corpus: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM documents"
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        return self.conn.execute(query, params).fetchone()[0]

    def get_fact_count(self, corpus: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM facts"
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        return self.conn.execute(query, params).fetchone()[0]

    def get_all_documents(self, corpus: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all ingested document metadata, optionally per corpus."""
        query = "SELECT doc_id, filename, file_path, total_pages, checksum_sha256, ingested_at, metadata, corpus FROM documents"
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        query += " ORDER BY ingested_at DESC;"
        rows = self.conn.execute(query, params).fetchall()
        docs = []
        for r in rows:
            meta = {}
            if r[6]:
                try:
                    meta = json.loads(r[6]) if isinstance(r[6], str) else r[6]
                except Exception:
                    meta = {}
            docs.append({
                "doc_id": r[0],
                "filename": r[1],
                "file_path": r[2],
                "total_pages": r[3],
                "checksum_sha256": r[4],
                "ingested_at": str(r[5]),
                "metadata": meta,
                "corpus": r[7] if len(r) > 7 and r[7] else meta.get("corpus") or meta.get("corpus_label") or ("Delhivery" if "delhivery" in r[1].lower() else ("India Macroeconomics" if any(k in r[1].lower() for k in ["india", "rbi", "economic survey", "imf"]) else "General Knowledge")),
            })
        return docs

    def get_available_corpora(self) -> List[str]:
        """Return distinct non-empty logical corpora."""
        try:
            query = """
                SELECT DISTINCT corpus FROM (
                    SELECT corpus FROM documents WHERE corpus IS NOT NULL AND TRIM(corpus) != ''
                    UNION
                    SELECT corpus FROM facts WHERE corpus IS NOT NULL AND TRIM(corpus) != ''
                    UNION
                    SELECT corpus FROM relationships WHERE corpus IS NOT NULL AND TRIM(corpus) != ''
                ) ORDER BY corpus;
            """
            rows = self.conn.execute(query).fetchall()
            corpora = [r[0].strip() for r in rows if r[0] and r[0].strip()]
            if not corpora:
                return ["India Macroeconomics", "Delhivery"]
            return corpora
        except Exception:
            return ["India Macroeconomics", "Delhivery"]

    def get_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a single fact by its unique ID."""
        query = """
            SELECT f.fact_id, COALESCE(d.filename, f.doc_id), f.page_number, f.category, f.entity, f.metric,
                   f.value, f.raw_value, f.temporal_period, f.reporting_scope, f.unit, f.accounting_standard,
                   f.verbatim_quote, f.section_title, f.confidence_score, f.corpus
            FROM facts f
            LEFT JOIN documents d ON f.doc_id = d.doc_id
            WHERE f.fact_id = ?
        """
        rows = self.conn.execute(query, [fact_id]).fetchall()
        if not rows:
            return None
        return self._rows_to_facts(rows)[0]

    def insert_relationships(self, reconciliations: List[CrossDocReconciliation]) -> int:
        """Insert or replace cross-document relationships in the relationship store."""
        count = 0
        for r in reconciliations:
            details = {
                "fact_a": r.fact_a.model_dump(),
                "fact_b": r.fact_b.model_dump(),
                "metadata": r.metadata,
            }
            corpus_val = getattr(r, "corpus", "") or ("Delhivery" if "delhivery" in (r.fact_a.entity or "").lower() else "India Macroeconomics")
            self.conn.execute("""
                INSERT OR REPLACE INTO relationships (
                    relationship_id, fact_a_id, fact_b_id, verdict, shift_type,
                    confidence, reasoning, summary, details, corpus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                r.reconciliation_id,
                r.fact_a_id or r.fact_a.fact_id,
                r.fact_b_id or r.fact_b.fact_id,
                r.verdict.value,
                r.shift_type.value,
                r.confidence,
                r.reasoning,
                r.reconciliation_summary,
                json.dumps(details),
                corpus_val,
            ])
            count += 1
        return count

    def get_all_relationships(self, verdict: Optional[str] = None, corpus: Optional[str] = None) -> List[CrossDocReconciliation]:
        """Retrieve all relationships with optional verdict and corpus filters."""
        query = "SELECT relationship_id, fact_a_id, fact_b_id, verdict, shift_type, confidence, reasoning, summary, details, corpus FROM relationships WHERE 1=1"
        params = []
        if corpus:
            query += " AND LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        if verdict:
            query += " AND LOWER(verdict) = LOWER(?)"
            params.append(verdict)
        query += " ORDER BY confidence DESC, relationship_id"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            details = json.loads(r[8]) if r[8] else {}
            fa_data = details.get("fact_a")
            fb_data = details.get("fact_b")
            meta = details.get("metadata", {})
            if fa_data and fb_data:
                fa = Fact(**fa_data)
                fb = Fact(**fb_data)
            else:
                fa = self.get_fact_by_id(r[1])
                fb = self.get_fact_by_id(r[2])
            if fa and fb:
                corp = r[9] if len(r) > 9 and r[9] else ("Delhivery" if "delhivery" in (fa.entity or "").lower() else "India Macroeconomics")
                results.append(CrossDocReconciliation(
                    reconciliation_id=r[0],
                    fact_a=fa,
                    fact_b=fb,
                    fact_a_id=r[1],
                    fact_b_id=r[2],
                    verdict=ReconciliationVerdict(r[3]),
                    shift_type=ContextualShiftType(r[4]),
                    confidence=r[5],
                    reasoning=r[6],
                    reconciliation_summary=r[7],
                    corpus=corp,
                    metadata=meta,
                ))
        return results

    def get_relationships_for_fact(self, fact_id: str) -> List[CrossDocReconciliation]:
        """Retrieve all relationships where a fact participates as fact_a or fact_b."""
        query = """
            SELECT relationship_id, fact_a_id, fact_b_id, verdict, shift_type, confidence, reasoning, summary, details, corpus
            FROM relationships
            WHERE fact_a_id = ? OR fact_b_id = ?
            ORDER BY confidence DESC
        """
        rows = self.conn.execute(query, [fact_id, fact_id]).fetchall()
        results = []
        for r in rows:
            details = json.loads(r[8]) if r[8] else {}
            fa_data = details.get("fact_a")
            fb_data = details.get("fact_b")
            meta = details.get("metadata", {})
            if fa_data and fb_data:
                fa = Fact(**fa_data)
                fb = Fact(**fb_data)
            else:
                fa = self.get_fact_by_id(r[1])
                fb = self.get_fact_by_id(r[2])
            if fa and fb:
                corp = r[9] if len(r) > 9 and r[9] else ("Delhivery" if "delhivery" in (fa.entity or "").lower() else "India Macroeconomics")
                results.append(CrossDocReconciliation(
                    reconciliation_id=r[0],
                    fact_a=fa,
                    fact_b=fb,
                    fact_a_id=r[1],
                    fact_b_id=r[2],
                    verdict=ReconciliationVerdict(r[3]),
                    shift_type=ContextualShiftType(r[4]),
                    confidence=r[5],
                    reasoning=r[6],
                    reconciliation_summary=r[7],
                    corpus=corp,
                    metadata=meta,
                ))
        return results

    def get_relationship_counts(self, corpus: Optional[str] = None) -> Dict[str, int]:
        """Dynamically compute counts of relationships across all categories, optionally filtered by corpus."""
        query = "SELECT verdict, COUNT(*) FROM relationships"
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        query += " GROUP BY verdict;"
        rows = self.conn.execute(query, params).fetchall()
        counts = {
            "total": 0,
            "corroborated": 0,
            "genuine_contradiction": 0,
            "context_reconciled": 0,
            "extraction_failure": 0,
        }
        for v, cnt in rows:
            counts[v] = cnt
            counts["total"] += cnt
        return counts

    def insert_failures(self, failures: List[FailureRecord]) -> int:
        """Insert or replace failure records in the failure store."""
        count = 0
        for f in failures:
            corpus_val = getattr(f, "corpus", "") or ("Delhivery" if "delhivery" in f.source_document.lower() else "India Macroeconomics")
            self.conn.execute("""
                INSERT OR REPLACE INTO failures (
                    failure_id, failure_type, source_document, page_number,
                    expected_evidence, observed_result, problem_description, impact,
                    detection_mechanism, mitigation_strategy, potential_improvement,
                    fact_id, related_fact_id, metadata, corpus
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, [
                f.failure_id,
                f.failure_type.value,
                f.source_document,
                f.page_number,
                f.expected_evidence,
                f.observed_result,
                f.problem_description,
                f.impact,
                f.detection_mechanism,
                f.mitigation_strategy,
                f.potential_improvement,
                f.fact_id,
                f.related_fact_id,
                json.dumps(f.metadata),
                corpus_val,
            ])
            count += 1
        return count

    def get_all_failures(self, failure_type: Optional[str] = None, corpus: Optional[str] = None) -> List[FailureRecord]:
        """Retrieve all recorded extraction and reasoning failures, optionally filtered by corpus."""
        query = """
            SELECT failure_id, failure_type, source_document, page_number,
                   expected_evidence, observed_result, problem_description, impact,
                   detection_mechanism, mitigation_strategy, potential_improvement,
                   fact_id, related_fact_id, metadata, corpus
            FROM failures
            WHERE 1=1
        """
        params = []
        if corpus:
            query += " AND LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        if failure_type:
            query += " AND LOWER(failure_type) = LOWER(?)"
            params.append(failure_type)
        query += " ORDER BY source_document, page_number"
        rows = self.conn.execute(query, params).fetchall()
        failures = []
        for r in rows:
            meta = json.loads(r[13]) if r[13] else {}
            failures.append(FailureRecord(
                failure_id=r[0],
                failure_type=FailureType(r[1]),
                source_document=r[2],
                page_number=r[3],
                expected_evidence=r[4],
                observed_result=r[5],
                problem_description=r[6],
                impact=r[7],
                detection_mechanism=r[8],
                mitigation_strategy=r[9],
                potential_improvement=r[10],
                fact_id=r[11],
                related_fact_id=r[12],
                metadata=meta,
                corpus=r[14] if len(r) > 14 and r[14] else ("Delhivery" if "delhivery" in r[2].lower() else "India Macroeconomics"),
            ))
        return failures

    def get_failure_count(self, corpus: Optional[str] = None) -> int:
        """Get total count of recorded failures, optionally per corpus."""
        query = "SELECT COUNT(*) FROM failures"
        params = []
        if corpus:
            query += " WHERE LOWER(corpus) = LOWER(?)"
            params.append(corpus)
        return self.conn.execute(query, params).fetchone()[0]

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its linked facts, relationships, and failures."""
        fact_rows = self.conn.execute("SELECT fact_id FROM facts WHERE doc_id = ?;", [doc_id]).fetchall()
        fact_ids = [r[0] for r in fact_rows]
        if fact_ids:
            placeholders = ",".join(["?"] * len(fact_ids))
            self.conn.execute(
                f"DELETE FROM relationships WHERE fact_a_id IN ({placeholders}) OR fact_b_id IN ({placeholders});",
                fact_ids + fact_ids,
            )
        self.conn.execute("DELETE FROM facts WHERE doc_id = ?;", [doc_id])
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?;", [doc_id])

    def clear(self) -> None:
        """Wipe tables for fresh runs/tests."""
        self.conn.execute("DELETE FROM relationships;")
        self.conn.execute("DELETE FROM failures;")
        self.conn.execute("DELETE FROM facts;")
        self.conn.execute("DELETE FROM documents;")

    def _rows_to_facts(self, rows: List[Any]) -> List[Fact]:
        facts = []
        for r in rows:
            ev = SourceEvidence(
                document_id=r[1],
                page_number=r[2],
                content_type="text",
                verbatim_quote=r[12],
                section_title=r[13],
            )
            ctx = ContextDimensions(
                temporal_period=r[8],
                reporting_scope=r[9],
                unit=r[10],
                accounting_standard=r[11],
            )
            corpus_val = r[15] if len(r) > 15 and r[15] else ("Delhivery" if "delhivery" in (r[4] or "").lower() else ("India Macroeconomics" if any(k in (r[4] or "").lower() for k in ["india", "rbi", "economic survey", "imf"]) else (r[4] or "General Knowledge")))
            fact = Fact(
                fact_id=r[0],
                category=FactCategory(r[3]),
                entity=r[4],
                metric=r[5],
                value=r[6],
                raw_value=r[7],
                context=ctx,
                evidence=ev,
                confidence_score=r[14],
                corpus=corpus_val,
            )
            facts.append(fact)
        return facts

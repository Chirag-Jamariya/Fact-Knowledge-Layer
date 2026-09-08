"""Unit tests for In-Corpus Store & Indexing (Component 4)."""

import pytest
from datetime import datetime, timezone
from src.indexing.store import KnowledgeStore
from src.indexing.clusterer import CrossDocClusterer
from src.indexing.retriever import InCorpusRetriever
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.models.corpus import DocumentRecord


@pytest.fixture
def memory_store():
    """Create an in-memory DuckDB store for fast, isolated testing."""
    return KnowledgeStore(db_path=":memory:")


def test_insert_and_retrieve_facts(memory_store):
    """Verify storing and retrieving documents and facts."""
    doc = DocumentRecord(
        doc_id="doc_1",
        filename="report1.pdf",
        file_path="/path/report1.pdf",
        total_pages=10,
        checksum_sha256="abcd1234",
    )
    memory_store.insert_document(doc)
    assert memory_store.get_document_count() == 1

    fact = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24", reporting_scope="Consolidated", unit="INR Cr"),
        evidence=SourceEvidence(document_id="doc_1", page_number=10, verbatim_quote="Revenue stood at 8,142 Cr"),
    )
    count = memory_store.insert_facts([fact], doc_id="doc_1")
    assert count == 1
    assert memory_store.get_fact_count() == 1

    # Query back
    stored_facts = memory_store.get_all_facts()
    assert len(stored_facts) == 1
    assert stored_facts[0].fact_id == fact.fact_id
    assert stored_facts[0].value == "8,142 Cr"
    assert stored_facts[0].context.temporal_period == "FY24"


def test_cross_document_clustering(memory_store):
    """Verify that facts from DIFFERENT documents sharing a metric form candidate pairs."""
    fa = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="3,646 Cr",
        raw_value="3,646",
        context=ContextDimensions(temporal_period="FY21"),
        evidence=SourceEvidence(document_id="doc_prospectus.pdf", page_number=28, verbatim_quote="FY21: 3,646 Cr"),
    )
    fb = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24"),
        evidence=SourceEvidence(document_id="doc_annual_report.pdf", page_number=105, verbatim_quote="FY24: 8,142 Cr"),
    )
    fc = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="3,646 Cr",
        raw_value="3,646",
        context=ContextDimensions(temporal_period="FY21"),
        evidence=SourceEvidence(document_id="doc_prospectus.pdf", page_number=50, verbatim_quote="Same doc repeated"),
    )

    pairs = CrossDocClusterer.find_cross_document_pairs([fa, fb, fc])
    # fa and fb are different docs -> formed pair
    # fb and fc are different docs -> formed pair
    # fa and fc are SAME doc -> NOT formed
    assert len(pairs) == 2
    for pair_a, pair_b in pairs:
        assert pair_a.evidence.document_id != pair_b.evidence.document_id


def test_in_corpus_retriever(memory_store):
    """Verify keyword retrieval across indexed facts."""
    doc = DocumentRecord(
        doc_id="doc_rbi",
        filename="rbi_report.pdf",
        file_path="/path/rbi.pdf",
        total_pages=50,
        checksum_sha256="efgh5678",
    )
    memory_store.insert_document(doc)

    f1 = Fact(
        entity="Indian Economy",
        metric="Headline CPI Inflation",
        value="5.4%",
        raw_value="5.4",
        context=ContextDimensions(temporal_period="2024"),
        evidence=SourceEvidence(document_id="doc_rbi", page_number=15, verbatim_quote="CPI inflation averaged 5.4%"),
    )
    memory_store.insert_facts([f1], doc_id="doc_rbi")

    retriever = InCorpusRetriever(memory_store)
    results = retriever.search_facts("Inflation")
    assert len(results) == 1
    assert results[0].value == "5.4%"


def test_purge_non_starter_uploads(tmp_path):
    """Verify that only PDFs outside the starter dataset list are purged."""
    from src.indexing.reset import purge_non_starter_uploads

    # Create dummy starter file and dummy foreign file
    starter_file = tmp_path / "01-delhivery-prospectus-2022-excerpt.pdf"
    starter_file.write_text("starter content")
    foreign_file = tmp_path / "unrelated_doc.pdf"
    foreign_file.write_text("foreign content")

    purged = purge_non_starter_uploads(upload_dir=str(tmp_path))
    assert "unrelated_doc.pdf" in purged
    assert not foreign_file.exists()
    assert starter_file.exists()


def test_reset_knowledge_to_starter_datasets(memory_store):
    """Verify reset clears db and restores all 6 starter PDFs and 217 facts from seed cache."""
    from src.indexing.reset import reset_knowledge_to_starter_datasets

    # Pre-populate with unrelated doc
    doc = DocumentRecord(
        doc_id="foreign_doc",
        filename="foreign.pdf",
        file_path="/path/foreign.pdf",
        total_pages=1,
        checksum_sha256="12345",
    )
    memory_store.insert_document(doc)
    assert memory_store.get_document_count() == 1

    res = reset_knowledge_to_starter_datasets(store=memory_store, purge_uploads=False)
    assert res["status"] == "success"
    assert res["documents_indexed"] == 6
    assert res["facts_indexed"] == 217
    assert memory_store.get_document_count() == 6
    assert memory_store.get_fact_count() == 217


def test_all_facts_visible_and_independent_of_cases(memory_store):
    """Verify that every fact is stored and accessible regardless of whether it participates in a case."""
    doc = DocumentRecord(doc_id="doc_test", filename="test.pdf", file_path="/path/test.pdf", total_pages=2, checksum_sha256="abc")
    memory_store.insert_document(doc)

    f_standalone = Fact(
        entity="Standalone Corp",
        metric="Patent Count",
        value="42",
        raw_value="42",
        evidence=SourceEvidence(document_id="test.pdf", page_number=1, verbatim_quote="The company holds 42 active patents."),
    )
    f_linked = Fact(
        entity="Standalone Corp",
        metric="Revenue",
        value="100 Cr",
        raw_value="100",
        evidence=SourceEvidence(document_id="test.pdf", page_number=2, verbatim_quote="Revenue reached 100 Cr."),
    )

    memory_store.insert_facts([f_standalone, f_linked], doc_id="doc_test")

    # All facts must be retrievable
    all_facts = memory_store.get_all_facts()
    assert len(all_facts) == 2
    fact_ids = {f.fact_id for f in all_facts}
    assert f_standalone.fact_id in fact_ids
    assert f_linked.fact_id in fact_ids

    # Query by ID
    retrieved = memory_store.get_fact_by_id(f_standalone.fact_id)
    assert retrieved is not None
    assert retrieved.metric == "Patent Count"
    assert retrieved.value == "42"

    # Standalone fact has 0 relationships
    rels = memory_store.get_relationships_for_fact(f_standalone.fact_id)
    assert len(rels) == 0


def test_relationship_store_and_fact_references(memory_store):
    """Verify relationship store links existing facts by ID and preserves many-to-many graph."""
    from src.models.reconciliation import CrossDocReconciliation, ReconciliationVerdict, ContextualShiftType

    doc_a = DocumentRecord(doc_id="doc_a", filename="a.pdf", file_path="/a.pdf", total_pages=1, checksum_sha256="1")
    doc_b = DocumentRecord(doc_id="doc_b", filename="b.pdf", file_path="/b.pdf", total_pages=1, checksum_sha256="2")
    memory_store.insert_document(doc_a)
    memory_store.insert_document(doc_b)

    fa = Fact(entity="Corp", metric="Revenue", value="100 Cr", raw_value="100", evidence=SourceEvidence(document_id="a.pdf", page_number=1, verbatim_quote="Revenue 100 Cr"))
    fb = Fact(entity="Corp", metric="Revenue", value="100 Cr", raw_value="100", evidence=SourceEvidence(document_id="b.pdf", page_number=1, verbatim_quote="Revenue 100 Cr"))
    fc = Fact(entity="Corp", metric="Revenue", value="150 Cr", raw_value="150", evidence=SourceEvidence(document_id="b.pdf", page_number=2, verbatim_quote="Revenue 150 Cr"))

    memory_store.insert_facts([fa], doc_id="doc_a")
    memory_store.insert_facts([fb, fc], doc_id="doc_b")

    # Relationship 1: fa corroborated by fb
    rel1 = CrossDocReconciliation(
        verdict=ReconciliationVerdict.CORROBORATED,
        shift_type=ContextualShiftType.NONE,
        fact_a=fa,
        fact_b=fb,
        reasoning="Both report 100 Cr",
        reconciliation_summary="Corroborated revenue",
    )
    # Relationship 2: fa in contradiction with fc
    rel2 = CrossDocReconciliation(
        verdict=ReconciliationVerdict.GENUINE_CONTRADICTION,
        shift_type=ContextualShiftType.NONE,
        fact_a=fa,
        fact_b=fc,
        reasoning="100 Cr vs 150 Cr",
        reconciliation_summary="Contradicting revenue",
    )

    memory_store.insert_relationships([rel1, rel2])

    # fa participates in 2 relationships (many-to-many)
    fa_rels = memory_store.get_relationships_for_fact(fa.fact_id)
    assert len(fa_rels) == 2
    verdicts = {r.verdict for r in fa_rels}
    assert ReconciliationVerdict.CORROBORATED in verdicts
    assert ReconciliationVerdict.GENUINE_CONTRADICTION in verdicts

    # Dynamic counts
    counts = memory_store.get_relationship_counts()
    assert counts["corroborated"] == 1
    assert counts["genuine_contradiction"] == 1
    assert counts["total"] == 2


def test_failure_store_records_diagnostics(memory_store):
    """Verify failure store retains structured failure records with all 10 diagnostic fields."""
    from src.models.reconciliation import FailureRecord, FailureType

    fail = FailureRecord(
        failure_type=FailureType.QUALIFIER_DROPPED,
        source_document="chad.pdf",
        page_number=9,
        expected_evidence="oil GDP growth average only 1.7 percent",
        observed_result="Real GDP Growth = 1.7%",
        problem_description="Dropped 'oil' modifier conflating sector output with headline Real GDP",
        impact="Produces false cross-national contradictions",
        detection_mechanism="Domain gazetteer modifier audit",
        mitigation_strategy="Enforce noun-phrase regexes ('Oil GDP Growth')",
        potential_improvement="Fine-tuned BIO token classifier",
    )

    memory_store.insert_failures([fail])
    assert memory_store.get_failure_count() == 1

    retrieved = memory_store.get_all_failures()
    assert len(retrieved) == 1
    rf = retrieved[0]
    assert rf.failure_type == FailureType.QUALIFIER_DROPPED
    assert rf.source_document == "chad.pdf"
    assert rf.page_number == 9
    assert "oil GDP" in rf.expected_evidence
    assert rf.mitigation_strategy != ""
    assert rf.potential_improvement != ""


def test_multi_pdf_batch_corpus_unification(memory_store):
    """Verify that uploading N documents into a new unified corpus registers in available_corpora and isolates cleanly."""
    from src.indexing.sync import sync_knowledge_graph

    corpus_name = "African Macroeconomics"

    # Ingest N=3 documents into this corpus
    for i in range(1, 4):
        doc = DocumentRecord(
            doc_id=f"afr_doc_{i}",
            filename=f"africa_report_{i}.pdf",
            file_path=f"/tmp/africa_report_{i}.pdf",
            total_pages=5,
            checksum_sha256=f"hash_{i}",
            corpus=corpus_name,
        )
        memory_store.insert_document(doc, corpus=corpus_name)

        f = Fact(
            entity="African Union",
            metric="Real GDP Growth",
            value="3.8%",
            raw_value="3.8",
            context=ContextDimensions(temporal_period="2024", unit="%"),
            evidence=SourceEvidence(
                document_id=f"africa_report_{i}.pdf",
                page_number=1,
                verbatim_quote="Growth is projected at 3.8 percent in 2024.",
            ),
            corpus=corpus_name,
        )
        memory_store.insert_facts([f], doc_id=f"afr_doc_{i}", corpus=corpus_name)

    # Ingest 1 document into Delhivery to test strict boundary
    delhivery_doc = DocumentRecord(
        doc_id="delhivery_1",
        filename="delhivery_report.pdf",
        file_path="/tmp/delhivery.pdf",
        total_pages=5,
        checksum_sha256="delh_hash",
        corpus="Delhivery",
    )
    memory_store.insert_document(delhivery_doc, corpus="Delhivery")
    f_del = Fact(
        entity="Delhivery Limited",
        metric="Revenue from Operations",
        value="8,142 Cr",
        raw_value="8,142",
        context=ContextDimensions(temporal_period="FY24"),
        evidence=SourceEvidence(document_id="delhivery_report.pdf", page_number=2, verbatim_quote="Revenue 8,142 Cr"),
        corpus="Delhivery",
    )
    memory_store.insert_facts([f_del], doc_id="delhivery_1", corpus="Delhivery")

    # Available corpora check (sidebar selector requirement)
    corpora = memory_store.get_available_corpora()
    assert corpus_name in corpora
    assert "Delhivery" in corpora

    # Synchronize
    sync_res = sync_knowledge_graph(memory_store)

    # Assert corpus doc counts & fact counts
    assert memory_store.get_document_count(corpus=corpus_name) == 3
    assert memory_store.get_fact_count(corpus=corpus_name) == 3
    assert memory_store.get_document_count(corpus="Delhivery") == 1
    assert memory_store.get_fact_count(corpus="Delhivery") == 1

    # Cross-document relationships within African Macroeconomics
    afr_rels = memory_store.get_all_relationships(corpus=corpus_name)
    assert len(afr_rels) > 0
    for r in afr_rels:
        assert r.corpus == corpus_name
        assert "africa" in r.fact_a.evidence.document_id
        assert "africa" in r.fact_b.evidence.document_id

    # Delhivery relationships must be completely isolated
    del_rels = memory_store.get_all_relationships(corpus="Delhivery")
    for r in del_rels:
        assert r.corpus == "Delhivery"
        assert "delhivery" in r.fact_a.evidence.document_id



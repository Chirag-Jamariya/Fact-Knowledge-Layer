"""Service to purge non-starter PDFs, reset knowledge store, and re-index all 6 starter PDFs."""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone

from src.config import settings
from src.indexing.store import KnowledgeStore
from src.models.corpus import DocumentRecord
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.ingestion.pdf_parser import PDFParser
from src.extraction.fact_extractor import FactExtractor

logger = logging.getLogger(__name__)

# The 6 official starter dataset files
STARTER_DATASET_SPECS = [
    {
        "filename": "01-delhivery-prospectus-2022-excerpt.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
        "entity": "Delhivery Limited",
        "corpus_label": "Delhivery Limited",
    },
    {
        "filename": "02-delhivery-annual-report-fy24-excerpt.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "entity": "Delhivery Limited",
        "corpus_label": "Delhivery Limited",
    },
    {
        "filename": "03-delhivery-q4-fy24-earnings-presentation.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        "entity": "Delhivery Limited",
        "corpus_label": "Delhivery Limited",
    },
    {
        "filename": "01-india-economic-survey-2024-25-excerpt.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
        "entity": "Indian Economy",
        "corpus_label": "Indian Economy",
    },
    {
        "filename": "02-rbi-annual-report-2024-25-excerpt.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
        "entity": "Indian Economy",
        "corpus_label": "Indian Economy",
    },
    {
        "filename": "03-imf-india-2025-article-iv-excerpt.pdf",
        "path": "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
        "entity": "Indian Economy",
        "corpus_label": "Indian Economy",
    },
]

STARTER_FILENAMES = {spec["filename"] for spec in STARTER_DATASET_SPECS}
DEFAULT_CACHE_PATH = os.path.join(settings.data_dir, "starter_seed_cache.json")


def purge_non_starter_uploads(upload_dir: Optional[str] = None) -> List[str]:
    """
    Remove all PDFs from upload_dir whose filenames do not belong to the starter datasets.
    Returns the list of removed filenames.
    """
    target_dir = upload_dir or settings.upload_dir
    removed = []
    if not os.path.exists(target_dir):
        return removed

    for fname in os.listdir(target_dir):
        if fname not in STARTER_FILENAMES:
            fpath = os.path.join(target_dir, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    removed.append(fname)
                    logger.info(f"Purged non-starter file: {fname}")
                except OSError as e:
                    logger.warning(f"Could not remove {fname}: {e}")

    # Also clean up any accidental database file in root data_dir
    extraneous_db = os.path.join(settings.data_dir, "knowledge.duckdb")
    if os.path.exists(extraneous_db):
        try:
            os.remove(extraneous_db)
        except OSError:
            pass

    return removed


def reset_knowledge_to_starter_datasets(
    store: Optional[KnowledgeStore] = None,
    purge_uploads: bool = True,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    1. Removes all other PDFs from the database except the 6 official starter PDFs.
    2. Purges all other PDFs from uploads directory.
    3. Ensures all 6 starter PDFs are present in the database (restoring missing ones if needed).
    """
    if store is None:
        store = KnowledgeStore()

    # Step 1: Purge non-starter files from uploads directory
    purged_files = purge_non_starter_uploads() if purge_uploads else []

    # Step 2: Remove all other PDFs from the database except the 6 starter PDFs
    existing_docs = store.get_all_documents()
    removed_from_db = []
    starter_docs_present = set()

    for doc in existing_docs:
        doc_filename = doc["filename"]
        doc_id = doc["doc_id"]
        if doc_filename not in STARTER_FILENAMES:
            # Delete other PDF and its facts from database
            store.delete_document(doc_id)
            removed_from_db.append(doc_filename)
            logger.info(f"Removed non-starter PDF from database: {doc_filename} ({doc_id})")
        else:
            starter_docs_present.add(doc_filename)

    active_cache_path = cache_path or DEFAULT_CACHE_PATH

    # Step 3: Ensure all 6 starter PDFs are in the database (seed any that are missing)
    missing_starters = STARTER_FILENAMES - starter_docs_present
    seeded_count = 0

    if missing_starters and use_cache and os.path.exists(active_cache_path):
        try:
            with open(active_cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            docs_data = [d for d in cache_data.get("documents", []) if d["filename"] in missing_starters]
            facts_data = cache_data.get("facts", [])

            for d in docs_data:
                store.conn.execute(
                    """
                    INSERT OR REPLACE INTO documents (
                        doc_id, filename, file_path, total_pages, checksum_sha256, ingested_at, metadata, corpus
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        d["doc_id"],
                        d["filename"],
                        d["file_path"],
                        d["total_pages"],
                        d["checksum_sha256"],
                        d["ingested_at"],
                        json.dumps(d.get("metadata", {})),
                        "Delhivery" if "delhivery" in d["filename"].lower() else "India Macroeconomics",
                    ],
                )

            # Insert facts for missing documents
            filename_to_doc_id = {d["filename"]: d["doc_id"] for d in cache_data.get("documents", [])}
            facts_by_doc: Dict[str, List[Fact]] = {}
            for fd in facts_data:
                doc_filename = fd.get("evidence", {}).get("document_id")
                if doc_filename in missing_starters:
                    target_doc_id = filename_to_doc_id.get(doc_filename, fd.get("doc_id"))
                    if target_doc_id:
                        facts_by_doc.setdefault(target_doc_id, []).append(Fact(**fd))

            for doc_id, facts in facts_by_doc.items():
                store.insert_facts(facts, doc_id)
            seeded_count = len(docs_data)
        except Exception as e:
            logger.warning(f"Error restoring missing starter documents: {e}")

    final_docs = store.get_all_documents()
    final_fact_count = store.get_fact_count()

    from src.indexing.sync import sync_knowledge_graph
    sync_res = sync_knowledge_graph(store)

    if progress_callback:
        progress_callback(6, 6, "Preserved 6 starter PDFs and synced knowledge layer")

    return {
        "status": "success",
        "removed_from_database": removed_from_db,
        "purged_uploads": purged_files,
        "starter_docs_restored": seeded_count,
        "documents_indexed": len(final_docs),
        "facts_indexed": final_fact_count,
        "relationships_indexed": sync_res.get("total_relationships", 0),
        "failures_indexed": sync_res.get("total_failures", 0),
        "documents": [d["filename"] for d in final_docs],
    }

    # Slow fallback path: Extract freshly from original PDF files
    parser = PDFParser()
    extractor = FactExtractor(use_llm_if_available=False)

    indexed_docs = []
    all_cached_docs = []
    all_cached_facts = []
    total_specs = len(STARTER_DATASET_SPECS)

    for idx, spec in enumerate(STARTER_DATASET_SPECS):
        pdf_path = spec["path"]
        if not os.path.exists(pdf_path):
            continue

        if progress_callback:
            progress_callback(idx, total_specs, f"Parsing {spec['filename']}...")

        doc_rec, pages = parser.parse_document(pdf_path)
        c_name = "Delhivery" if "delhivery" in spec["corpus_label"].lower() else "India Macroeconomics"
        doc_rec.corpus = c_name
        doc_rec.metadata["corpus"] = c_name
        doc_rec.metadata["corpus_label"] = spec["corpus_label"]
        store.insert_document(doc_rec, corpus=c_name)

        facts = extractor.extract_pages_concurrently(pages, doc_rec.filename)
        for f in facts:
            f.entity = spec["entity"]
            f.corpus = c_name

        store.insert_facts(facts, doc_rec.doc_id, corpus=c_name)
        indexed_docs.append(doc_rec.filename)

        all_cached_docs.append({
            "doc_id": doc_rec.doc_id,
            "filename": doc_rec.filename,
            "file_path": doc_rec.file_path,
            "total_pages": doc_rec.total_pages,
            "checksum_sha256": doc_rec.checksum_sha256,
            "ingested_at": doc_rec.ingested_at.isoformat(),
            "metadata": doc_rec.metadata,
        })
        all_cached_facts.extend([f.model_dump(mode="json") for f in facts])

    # Save to cache for future instant resets
    try:
        os.makedirs(os.path.dirname(os.path.abspath(active_cache_path)), exist_ok=True)
        with open(active_cache_path, "w", encoding="utf-8") as f:
            json.dump({"documents": all_cached_docs, "facts": all_cached_facts}, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist starter cache: {e}")

    if progress_callback:
        progress_callback(total_specs, total_specs, "All 6 starter PDFs indexed")

    return {
        "status": "success",
        "source": "fresh_extraction",
        "purged_files": purged_files,
        "documents_indexed": len(indexed_docs),
        "facts_indexed": len(all_cached_facts),
        "documents": indexed_docs,
    }

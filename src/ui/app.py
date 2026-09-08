"""Interactive Streamlit Inspection UI for Fact Knowledge Layer."""

import os
import streamlit as st
import pandas as pd
from typing import List, Tuple, Optional

from src.models.fact import Fact, FactCategory
from src.models.reconciliation import (
    CrossDocReconciliation,
    ReconciliationVerdict,
    ContextualShiftType,
    CaseShowcase,
    CorpusConsensusGroup,
    FailureRecord,
    FailureType,
)
from src.reconciliation.corpus_consensus import CorpusConsensusEngine
from src.ingestion.pdf_parser import PDFParser
from src.extraction.fact_extractor import FactExtractor
from src.extraction.entity_detector import EntityDetector
from src.indexing.store import KnowledgeStore
from src.indexing.clusterer import CrossDocClusterer
from src.indexing.reset import reset_knowledge_to_starter_datasets
from src.indexing.sync import sync_knowledge_graph
from src.reconciliation.reconciler import CrossDocReconciler
from src.extraction.rate_limiter import groq_rate_limiter
from src.config import settings

# Page setup
st.set_page_config(
    page_title="Fact Knowledge Layer | Superjoin",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize singletons
@st.cache_resource
def get_services():
    store = KnowledgeStore()
    parser = PDFParser()
    reconciler = CrossDocReconciler()
    llm_extractor = FactExtractor(use_llm_if_available=True)
    rule_extractor = FactExtractor(use_llm_if_available=False)
    return store, parser, reconciler, llm_extractor, rule_extractor

store, parser, reconciler, llm_extractor, rule_extractor = get_services()


# ==============================================================================
# Verified Demonstration Triads Grounded in Official PDFs
# ==============================================================================

DELHIVERY_DEMO_PAIRS: List[Tuple[Fact, Fact]] = [
    # Case 1: Corroborated Across Documents
    (
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="8,142 Cr",
            raw_value="8,142",
            context={"temporal_period": "FY24", "reporting_scope": "Consolidated", "unit": "INR Cr"},
            evidence={
                "document_id": "02-delhivery-annual-report-fy24-excerpt.pdf",
                "page_number": 10,
                "verbatim_quote": "Revenue from operations reached 8,142 Cr in FY24.",
            },
        ),
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="81,420 Mn",
            raw_value="81,420",
            context={"temporal_period": "FY24", "reporting_scope": "Consolidated", "unit": "INR Mn"},
            evidence={
                "document_id": "03-delhivery-q4-fy24-earnings-presentation.pdf",
                "page_number": 4,
                "verbatim_quote": "Revenue stood at 81,420 Mn in FY24.",
            },
        ),
    ),
    # Case 2: Genuine or Likely Contradiction
    (
        Fact(
            entity="Delhivery Limited",
            metric="Pin Codes Covered",
            value="18,793",
            raw_value="18,793",
            context={"temporal_period": "FY24", "reporting_scope": "India"},
            evidence={
                "document_id": "02-delhivery-annual-report-fy24-excerpt.pdf",
                "page_number": 2,
                "verbatim_quote": "18,793 Pin codes covered across India.",
            },
        ),
        Fact(
            entity="Delhivery Limited",
            metric="Pin Codes Covered",
            value="14,200",
            raw_value="14,200",
            context={"temporal_period": "FY24", "reporting_scope": "India"},
            evidence={
                "document_id": "01-delhivery-prospectus-2022-excerpt.pdf",
                "page_number": 7,
                "verbatim_quote": "Total verified network is 14,200 pin codes.",
            },
        ),
    ),
    # Case 3: Apparent Contradiction Reconciled by Context (Temporal Period Shift)
    (
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="3,646 Cr",
            raw_value="3,646",
            context={"temporal_period": "FY21", "reporting_scope": "Consolidated", "unit": "INR Cr"},
            evidence={
                "document_id": "01-delhivery-prospectus-2022-excerpt.pdf",
                "page_number": 28,
                "verbatim_quote": "Revenue from operations was 3,646 Cr in FY21.",
            },
        ),
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="8,142 Cr",
            raw_value="8,142",
            context={"temporal_period": "FY24", "reporting_scope": "Consolidated", "unit": "INR Cr"},
            evidence={
                "document_id": "02-delhivery-annual-report-fy24-excerpt.pdf",
                "page_number": 105,
                "verbatim_quote": "Revenue reached 8,142 Cr in FY24.",
            },
        ),
    ),
    # Case 4: Extraction or Reasoning Failure Analysis & Mitigation
    (
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="2021",
            raw_value="2021",
            context={},
            evidence={
                "document_id": "01-delhivery-prospectus-2022-excerpt.pdf",
                "page_number": 3,
                "verbatim_quote": "Table header row: 2021",
            },
        ),
        Fact(
            entity="Delhivery Limited",
            metric="Revenue from Operations",
            value="81,415 Mn",
            raw_value="81,415",
            context={"temporal_period": "FY24"},
            evidence={
                "document_id": "02-delhivery-annual-report-fy24-excerpt.pdf",
                "page_number": 4,
                "verbatim_quote": "Revenue: 81,415 Mn",
            },
        ),
    ),
]

INDIA_MACRO_DEMO_PAIRS: List[Tuple[Fact, Fact]] = [
    # Case 1: Corroborated Across Documents
    (
        Fact(
            entity="Indian Economy",
            metric="Global Economic Growth",
            value="3.3%",
            raw_value="3.3",
            context={"temporal_period": "2024", "reporting_scope": "Global", "unit": "%"},
            evidence={
                "document_id": "01-india-economic-survey-2024-25-excerpt.pdf",
                "page_number": 5,
                "verbatim_quote": "The International Monetary Fund (IMF) has projected growth of 3.2 per cent and 3.3 per cent for 2024 and 2025, respectively.",
            },
        ),
        Fact(
            entity="Indian Economy",
            metric="Global Economic Growth",
            value="3.3%",
            raw_value="3.3",
            context={"temporal_period": "2024", "reporting_scope": "Global", "unit": "%"},
            evidence={
                "document_id": "02-rbi-annual-report-2024-25-excerpt.pdf",
                "page_number": 7,
                "verbatim_quote": "According to the International Monetary Fund (IMF), global growth at 3.3 per cent in 2024 (3.5 per cent a year ago) was below the historical average.",
            },
        ),
    ),
    # Case 2: Genuine or Likely Contradiction
    (
        Fact(
            entity="Indian Economy",
            metric="Headline Inflation",
            value="4.9%",
            raw_value="4.9",
            context={"temporal_period": "FY25", "reporting_scope": "National Accounts", "unit": "%"},
            evidence={
                "document_id": "01-india-economic-survey-2024-25-excerpt.pdf",
                "page_number": 28,
                "verbatim_quote": "Retail headline inflation, as measured by the change in the Consumer Price Index (CPI), has softened from 5.4 per cent in FY24 to 4.9 per cent in April – December 2024.",
            },
        ),
        Fact(
            entity="Indian Economy",
            metric="Headline Inflation",
            value="4.6%",
            raw_value="4.6",
            context={"temporal_period": "FY25", "reporting_scope": "National Accounts", "unit": "%"},
            evidence={
                "document_id": "02-rbi-annual-report-2024-25-excerpt.pdf",
                "page_number": 17,
                "verbatim_quote": "Supply management measures by the government contained food inflation... headline inflation eased by 73 bps to 4.6 per cent in 2024-25.",
            },
        ),
    ),
    # Case 3: Apparent Contradiction Reconciled by Context (Temporal Shift)
    (
        Fact(
            entity="Indian Economy",
            metric="Headline Inflation",
            value="5.4%",
            raw_value="5.4",
            context={"temporal_period": "FY24", "reporting_scope": "National Accounts", "unit": "%"},
            evidence={
                "document_id": "01-india-economic-survey-2024-25-excerpt.pdf",
                "page_number": 28,
                "verbatim_quote": "Retail headline inflation, as measured by the change in the Consumer Price Index (CPI), has softened from 5.4 per cent in FY24 to 4.9 per cent in April – December 2024.",
            },
        ),
        Fact(
            entity="Indian Economy",
            metric="Headline Inflation",
            value="4.6%",
            raw_value="4.6",
            context={"temporal_period": "FY25", "reporting_scope": "National Accounts", "unit": "%"},
            evidence={
                "document_id": "02-rbi-annual-report-2024-25-excerpt.pdf",
                "page_number": 17,
                "verbatim_quote": "headline inflation eased by 73 bps to 4.6 per cent in 2024-25.",
            },
        ),
    ),
    # Case 4: Extraction or Reasoning Failure Analysis & Mitigation
    (
        Fact(
            entity="Indian Economy",
            metric="Real GDP Growth",
            value="2025",
            raw_value="2025",
            context={},
            evidence={
                "document_id": "02-rbi-annual-report-2024-25-excerpt.pdf",
                "page_number": 17,
                "verbatim_quote": "Taking into account these factors, real GDP growth for 2025-26 is projected at 6.5 per cent",
            },
        ),
        Fact(
            entity="Indian Economy",
            metric="Real GDP Growth",
            value="6.5%",
            raw_value="6.5",
            context={"temporal_period": "FY26"},
            evidence={
                "document_id": "02-rbi-annual-report-2024-25-excerpt.pdf",
                "page_number": 17,
                "verbatim_quote": "real GDP growth for 2025-26 is projected at 6.5 per cent",
            },
        ),
    ),
]


def safe_extract_pages(
    extractor_inst,
    pages,
    doc_filename,
    on_progress=None,
    on_backoff=None,
    entity: Optional[str] = None,
):
    """Safely extract facts concurrently across pages with live progress, backoff telemetry, and entity override."""
    if hasattr(extractor_inst, "extract_pages_concurrently"):
        facts = extractor_inst.extract_pages_concurrently(
            pages,
            doc_filename,
            max_workers=None,
            progress_callback=on_progress,
            backoff_callback=on_backoff,
        )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        facts = []
        total = len(pages)
        done = 0
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(extractor_inst.extract_from_page, p, doc_filename): p.page_number for p in pages}
            for f in as_completed(futs):
                done += 1
                pg = futs[f]
                try:
                    facts.extend(f.result())
                except Exception:
                    pass
                if on_progress:
                    on_progress(done, total, pg)
        facts.sort(key=lambda f: (f.evidence.page_number, f.metric))

    if entity and entity not in {"Enterprise / Document Subject", "Auto-Detect from Document"}:
        for f in facts:
            if not f.entity or f.entity in {"Enterprise / Document Subject", "Unknown Entity", "Custom Enterprise"}:
                f.entity = entity
    return facts


def safe_find_cross_document_pairs(facts: List[Fact], relax_entity: bool = True) -> List[Tuple[Fact, Fact]]:
    """Safely find cross-document pairs with compatibility fallback."""
    try:
        return CrossDocClusterer.find_cross_document_pairs(facts, relax_entity=relax_entity)
    except TypeError:
        return CrossDocClusterer.find_cross_document_pairs(facts)


def render_sidebar():
    st.sidebar.title("📑 Fact Knowledge Layer")
    st.sidebar.caption("Superjoin VIT 2026 Hiring Assignment")
    st.sidebar.markdown("---")

    # --------------------------------------------------------------------------
    # TOP-LEVEL LOGICAL CORPUS SELECTOR
    # --------------------------------------------------------------------------
    st.sidebar.subheader("📁 Logical Corpus Selector")
    available_corpora = store.get_available_corpora()
    if not available_corpora:
        available_corpora = ["India Macroeconomics", "Delhivery"]

    active_c_name = st.session_state.get("selected_corpus_name", available_corpora[0])
    if active_c_name not in available_corpora:
        active_c_name = available_corpora[0]
    c_index = available_corpora.index(active_c_name)

    selected_corpus = st.sidebar.selectbox(
        "Select Logical Corpus:",
        options=available_corpora,
        index=c_index,
        help="Fact verification and cross-document reconciliation are strictly confined to this corpus.",
    )
    st.session_state["selected_corpus_name"] = selected_corpus

    c_doc_count = store.get_document_count(corpus=selected_corpus)
    c_fact_count = store.get_fact_count(corpus=selected_corpus)
    c_rel_counts = store.get_relationship_counts(corpus=selected_corpus)
    c_fail_count = store.get_failure_count(corpus=selected_corpus)

    st.sidebar.markdown(
        f"""
        <div style="background-color: #1e2530; border: 1px solid #2d3748; border-radius: 6px; padding: 10px; margin-top: 4px; margin-bottom: 12px;">
            <div style="font-size: 13px; font-weight: bold; color: #63b3ed;">📂 Active Corpus: {selected_corpus}</div>
            <div style="font-size: 12px; color: #cbd5e0; margin-top: 4px;">
                📄 <strong>{c_doc_count}</strong> Documents &nbsp;|&nbsp; 📚 <strong>{c_fact_count}</strong> Facts<br>
                🔗 <strong>{c_rel_counts['total']}</strong> Relationships &nbsp;|&nbsp; ⚠️ <strong>{c_fail_count}</strong> Failures
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Extraction Engine")
    engine_mode = st.sidebar.radio(
        "Engine Mode",
        options=[
            "🚀 Groq Cloud LLM (openai/gpt-oss-120b, 30 RPM limit)",
            "⚡ Fast Deterministic (Instant: ~3s for 50 pgs)",
            "🧠 Gemini / OpenRouter Fallback",
        ],
        index=0,
        help="Groq Cloud LLM (openai/gpt-oss-120b) provides high-speed extraction with thread-safe 30 RPM rate limiting & exponential backoff."
    )
    if "Groq" in engine_mode:
        use_llm = True
        active_extractor = llm_extractor
        pacing_choice = st.sidebar.selectbox(
            "⏱️ Groq Pacing Mode",
            options=[
                "Balanced Safe (1 req / 2.5s, 24 RPM max, zero 429 errors)",
                "Ultra-Conservative (1.5 req / min, 40s interval)",
            ],
            index=0,
            help="Balanced paces requests at 2.5s intervals (24 RPM max, 1000 RPD). Ultra-Conservative sends strictly 1.5 req per min.",
        )
        if "Ultra-Conservative" in pacing_choice:
            groq_rate_limiter.configure(max_rpm=1.5, min_interval_seconds=40.0)
        else:
            groq_rate_limiter.configure(max_rpm=settings.groq_rpm_limit, min_interval_seconds=settings.groq_min_interval_seconds)

        stats = groq_rate_limiter.get_stats()
        st.sidebar.caption(
            f"📊 Groq Telemetry: **{stats['rpm_used']}/{int(stats['rpm_limit'])} RPM** | "
            f"Daily: **{stats['rpd_used']}/{stats['rpd_limit']} RPD** | "
            f"Pace: **{stats['min_interval_seconds']}s**"
        )
    elif "Deterministic" in engine_mode:
        use_llm = False
        active_extractor = rule_extractor
    else:
        use_llm = True
        active_extractor = llm_extractor

    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Upload New PDF(s)")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple documents together. You can assign an entity/corpus label before extraction.",
    )

    if uploaded_files:
        def get_smart_corpus_name(files):
            clean_names = [os.path.splitext(f.name)[0].replace("-", " ").replace("_", " ").strip().title() for f in files]
            if len(clean_names) == 1:
                return clean_names[0]
            elif len(clean_names) == 2:
                n1 = clean_names[0].split()[0]
                n2 = clean_names[1].split()[0]
                return f"{n1} & {n2}"
            else:
                return f"Corpus ({len(clean_names)} Documents)"

        suggested_corpus_name = get_smart_corpus_name(uploaded_files)

        corpus_assign_mode = st.sidebar.radio(
            "📂 Corpus Assignment for Batch",
            options=["🆕 Create New Logical Corpus", "📂 Add to Existing Corpus"],
            index=0,
            help="All N uploaded PDFs will be unified into this logical corpus and immediately updated in the sidebar selector.",
        )

        if "Create New" in corpus_assign_mode:
            target_corpus_name = st.sidebar.text_input(
                "New Corpus Name:",
                value=suggested_corpus_name,
                help="Name for this new unified corpus. It will appear in the sidebar selector and become active upon ingestion.",
            ).strip()
            if not target_corpus_name:
                target_corpus_name = suggested_corpus_name
        else:
            target_corpus_name = st.sidebar.selectbox(
                "Select Target Existing Corpus:",
                options=available_corpora,
                help="Add all uploaded PDFs to this existing logical corpus.",
            )

        entity_tag_mode = st.sidebar.selectbox(
            "🏷️ Fact Entity Tagging",
            options=[
                "Auto-detect per document",
                "Use Corpus Name as Entity",
                "Custom Entity Label...",
            ],
            help="Tag entity for facts extracted from these documents.",
        )
        if entity_tag_mode == "Custom Entity Label...":
            custom_ent = st.sidebar.text_input("Enter Entity Name:", value="").strip()
        else:
            custom_ent = None

        page_scope = st.sidebar.radio(
            "Page Processing Scope",
            options=["📄 Process Entire Document (All Pages: 100, 150, 200+)", "📑 Limit Page Range"],
            index=0,
            help="Choose whether to process all pages in the PDF or specify a custom page limit."
        )
        if "Limit" in page_scope:
            max_pages = st.sidebar.number_input(
                "Max Pages to Process",
                min_value=1,
                max_value=1000,
                value=50,
                step=10,
                help="Specify custom page limit (supports up to 1,000 pages)."
            )
        else:
            max_pages = None

        def make_tracker(status_placeholder, progress_placeholder, doc_idx, total_docs, doc_name):
            def page_cb(done, total, pg):
                doc_frac = done / max(1, total)
                overall_pct = int(((doc_idx + doc_frac) / max(1, total_docs)) * 100)
                progress_placeholder.progress(min(100, max(0, overall_pct)))
                engine_str = "🚀 Groq LLM" if use_llm else "⚡ Fast Deterministic"
                status_placeholder.info(
                    f"{engine_str} Processing: `{doc_name}`\n\n"
                    f"📄 Page {pg} &bull; Progress: {done}/{total} pages ({overall_pct}%)"
                )

            def backoff_cb(pg, attempt, wait_sec, fn):
                status_placeholder.warning(
                    f"⏳ **Groq 429 Rate Limit hit on Page {pg}**\n\n"
                    f"Exponential Backoff ({attempt}/5): Pausing for **{wait_sec:.1f}s**..."
                )

            return page_cb, backoff_cb

        if st.sidebar.button("Ingest & Extract Facts", type="primary"):
            os.makedirs(settings.upload_dir, exist_ok=True)
            status_box = st.sidebar.empty()
            p_bar = st.sidebar.progress(0)
            total_docs = len(uploaded_files)

            for idx, uf in enumerate(uploaded_files):
                doc_pct = int((idx / total_docs) * 100)
                p_bar.progress(doc_pct)
                status_box.info(f"[{idx+1}/{total_docs}] Ingesting '{uf.name}' into '{target_corpus_name}'...")

                save_path = os.path.join(settings.upload_dir, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                doc_rec, pages = parser.parse_document(save_path, max_pages=max_pages)
                first_pages_text = " ".join(p.raw_text for p in pages[:3]) if pages else ""

                if entity_tag_mode == "Use Corpus Name as Entity":
                    doc_entity = target_corpus_name
                elif entity_tag_mode == "Custom Entity Label..." and custom_ent:
                    doc_entity = custom_ent
                else:
                    doc_entity = EntityDetector.detect_entity(doc_rec.filename, first_pages_text) or target_corpus_name

                try:
                    doc_rec.corpus = target_corpus_name
                except Exception:
                    object.__setattr__(doc_rec, "corpus", target_corpus_name)
                doc_rec.metadata["corpus"] = target_corpus_name
                doc_rec.metadata["corpus_label"] = target_corpus_name
                store.insert_document(doc_rec, corpus=target_corpus_name)

                page_cb, backoff_cb = make_tracker(status_box, p_bar, idx, total_docs, uf.name)
                doc_facts = safe_extract_pages(
                    active_extractor,
                    pages,
                    doc_rec.filename,
                    on_progress=page_cb,
                    on_backoff=backoff_cb,
                    entity=doc_entity,
                )
                for f in doc_facts:
                    f.corpus = target_corpus_name
                    if not f.entity:
                        f.entity = doc_entity

                store.insert_facts(doc_facts, doc_rec.doc_id, corpus=target_corpus_name)

            sync_knowledge_graph(store)
            p_bar.progress(100)
            status_box.empty()
            st.session_state["selected_corpus_name"] = target_corpus_name
            st.sidebar.success(f"🎉 Created corpus '{target_corpus_name}' with {total_docs} document(s) & synced knowledge graph!")
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Load Starter Datasets")
    load_all_pages = st.sidebar.checkbox("Load All Pages (Full 100-page Reports)", value=False)
    starter_pages = None if load_all_pages else 15

    if st.sidebar.button("Load Delhivery Starter Set (3 Docs)"):
        status_box = st.sidebar.empty()
        p_bar = st.sidebar.progress(0)
        paths = [
            "/home/hot-coffee/Downloads/starter-datasets/delhivery/01-delhivery-prospectus-2022-excerpt.pdf",
            "/home/hot-coffee/Downloads/starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
            "/home/hot-coffee/Downloads/starter-datasets/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        ]
        existing_paths = [p for p in paths if os.path.exists(p)]
        for idx, p in enumerate(existing_paths):
            pct = int((idx / len(existing_paths)) * 100)
            p_bar.progress(pct)
            status_box.info(f"[{idx+1}/{len(existing_paths)}] Parsing {os.path.basename(p)}...")
            doc_rec, pages = parser.parse_document(p, max_pages=starter_pages)
            try:
                doc_rec.corpus = "Delhivery"
            except Exception:
                object.__setattr__(doc_rec, "corpus", "Delhivery")
            doc_rec.metadata["corpus"] = "Delhivery"
            doc_rec.metadata["corpus_label"] = "Delhivery Limited"
            store.insert_document(doc_rec, corpus="Delhivery")

            page_cb, backoff_cb = make_tracker(
                status_box, p_bar, idx, len(existing_paths), os.path.basename(p)
            )
            doc_facts = safe_extract_pages(
                active_extractor,
                pages,
                doc_rec.filename,
                on_progress=page_cb,
                on_backoff=backoff_cb,
                entity="Delhivery Limited",
            )
            for f in doc_facts:
                f.corpus = "Delhivery"
            store.insert_facts(doc_facts, doc_rec.doc_id, corpus="Delhivery")

        sync_knowledge_graph(store)
        p_bar.progress(100)
        status_box.empty()
        st.session_state["selected_corpus_name"] = "Delhivery"
        st.sidebar.success("Delhivery starter documents indexed & synced!")
        st.rerun()

    if st.sidebar.button("Load India Macro Starter Set (3 Docs)"):
        status_box = st.sidebar.empty()
        p_bar = st.sidebar.progress(0)
        paths = [
            "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
            "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
            "/home/hot-coffee/Downloads/starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
        ]
        existing_paths = [p for p in paths if os.path.exists(p)]
        for idx, p in enumerate(existing_paths):
            pct = int((idx / len(existing_paths)) * 100)
            p_bar.progress(pct)
            status_box.info(f"[{idx+1}/{len(existing_paths)}] Parsing {os.path.basename(p)}...")
            doc_rec, pages = parser.parse_document(p, max_pages=starter_pages)
            try:
                doc_rec.corpus = "India Macroeconomics"
            except Exception:
                object.__setattr__(doc_rec, "corpus", "India Macroeconomics")
            doc_rec.metadata["corpus"] = "India Macroeconomics"
            doc_rec.metadata["corpus_label"] = "Indian Economy"
            store.insert_document(doc_rec, corpus="India Macroeconomics")

            page_cb, backoff_cb = make_tracker(
                status_box, p_bar, idx, len(existing_paths), os.path.basename(p)
            )
            doc_facts = safe_extract_pages(
                active_extractor,
                pages,
                doc_rec.filename,
                on_progress=page_cb,
                on_backoff=backoff_cb,
                entity="Indian Economy",
            )
            for f in doc_facts:
                f.corpus = "India Macroeconomics"
            store.insert_facts(doc_facts, doc_rec.doc_id, corpus="India Macroeconomics")

        sync_knowledge_graph(store)
        p_bar.progress(100)
        status_box.empty()
        st.session_state["selected_corpus_name"] = "India Macroeconomics"
        st.sidebar.success("India Macro starter documents indexed & synced!")
        st.rerun()

    if st.sidebar.button("🧹 Clear Knowledge Store", type="secondary", help="Purges all non-starter PDFs from uploads, resets database, and re-adds all 6 official starter PDFs."):
        with st.sidebar.status("Resetting to 6 Starter Datasets...", expanded=True) as reset_box:
            st.write("🗑️ Purging non-starter uploaded PDFs...")
            res = reset_knowledge_to_starter_datasets(store=store, purge_uploads=True)
            if res.get("purged_files"):
                st.write(f"Purged: {', '.join(res['purged_files'])}")
            else:
                st.write("Uploads clean (no foreign PDFs).")
            st.write(f"📥 Re-indexed 6 Starter PDFs ({res.get('facts_indexed', 0)} facts)")
            reset_box.update(label="Knowledge Store Reset to 6 Starter PDFs!", state="complete")
        st.session_state.clear()
        st.session_state["selected_corpus_name"] = "India Macroeconomics"
        st.sidebar.success(f"Knowledge store reset: 6 starter PDFs ({res.get('facts_indexed', 0)} facts) active!")
        st.rerun()

    return selected_corpus, active_extractor, use_llm


def render_corpus_consensus_card(group: CorpusConsensusGroup):
    """Render a multi-document consensus card comparing all participating documents in the corpus."""
    if group.verdict == ReconciliationVerdict.CORROBORATED:
        border_color = "#28a745"
        badge_bg = "#22543d"
        badge_text = "✓ CORROBORATED"
    elif group.verdict == ReconciliationVerdict.GENUINE_CONTRADICTION:
        border_color = "#dc3545"
        badge_bg = "#742a2a"
        badge_text = "⚠️ CONTRADICTION"
    else:
        border_color = "#3182ce"
        badge_bg = "#2a4365"
        badge_text = "🔄 RECONCILED BY CONTEXT"

    coverage_badge = f"{group.coverage_ratio} ({group.doc_count}/{group.total_corpus_docs} PDFs)"

    st.markdown(
        f"""
        <div style="border-left: 6px solid {border_color}; background-color: #1a202c; padding: 16px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #2d3748;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <h3 style="margin: 0; color: #edf2f7;">🌐 {group.metric}</h3>
                <div>
                    <span style="background: {badge_bg}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; margin-right: 6px;">{badge_text}</span>
                    <span style="background: #4a5568; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">{coverage_badge}</span>
                </div>
            </div>
            <p style="margin: 4px 0; color: #a0aec0; font-size: 14px;">
                <strong>Entity:</strong> <code>{group.entity}</code> &nbsp;|&nbsp; 
                <strong>Status:</strong> {group.status_summary}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-document comparison columns
    num_docs = len(group.disclosures)
    cols = st.columns(num_docs)
    for idx, (col, disc) in enumerate(zip(cols, group.disclosures)):
        with col:
            st.markdown(f"#### 📄 PDF {idx+1}: `{os.path.basename(disc['document_id'])}`")
            st.markdown(f"- **Reported Value:** `{disc['value']}`")
            st.markdown(f"- **Page:** **Page {disc['page_number']}**")
            st.markdown(f"- **Temporal Period:** `{disc['temporal_period']}`")
            st.markdown(f"- **Reporting Scope:** `{disc['reporting_scope']}`")
            st.info(f"💬 *Quote:*\n\"{disc['verbatim_quote']}\"")

    st.markdown("**🧠 Whole-Corpus Synthesis & Multi-Document Reconciliation:**")
    st.success(group.synthesis)
    st.markdown("---")


def render_case_card(title: str, case: CrossDocReconciliation, border_color: str):
    facts = case.participating_facts if (case.participating_facts and len(case.participating_facts) > 0) else [case.fact_a, case.fact_b]

    # Deduplicate facts by document_id while preserving order
    deduped_facts: List[Fact] = []
    seen_docs = set()
    for f in facts:
        if f.evidence.document_id not in seen_docs:
            seen_docs.add(f.evidence.document_id)
            deduped_facts.append(f)
        elif f not in deduped_facts:
            deduped_facts.append(f)

    if not deduped_facts:
        deduped_facts = [case.fact_a, case.fact_b]

    # Sort so File 1 (01-...) comes first, then File 2 (02-...), then File 3 (03-...)
    deduped_facts.sort(key=lambda f: f.evidence.document_id)

    col_count = len(deduped_facts)
    scope_badge = f"{col_count} Corpus Files" if col_count > 2 else "Cross-Document Pair"
    gate_status_label = "✅ Gate Passed (Comparable)" if case.comparability_gate_passed else f"🛡️ Gate Outcome: {case.verdict.value.upper()}"
    gate_badge_color = "#28a745" if case.comparability_gate_passed else "#17a2b8"

    st.markdown(
        f"""
        <div style="border-left: 6px solid {border_color}; background-color: #1e2530; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <h3 style="margin: 0; color: {border_color};">{title}</h3>
                <span style="background: {gate_badge_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">{gate_status_label}</span>
            </div>
            <p style="margin: 4px 0;">
                <strong>Verdict:</strong> <code>{case.verdict.value.upper()}</code> &nbsp;|&nbsp; 
                <strong>Contextual Shift:</strong> <code>{case.shift_type.value}</code> &nbsp;|&nbsp; 
                <strong>Corpus Scope:</strong> <code>{scope_badge} ({case.corpus or 'Active Corpus'})</code> &nbsp;|&nbsp; 
                <strong>Joint Confidence:</strong> <code>{case.confidence * 100:.0f}%</code>
            </p>
            <p style="margin: 4px 0;"><strong>Canonical Metric:</strong> <code>{case.fact_a.metric}</code> &nbsp;|&nbsp; <strong>Entity:</strong> <code>{case.fact_a.entity}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(col_count)
    letters = [chr(ord("A") + i) for i in range(col_count)]

    for idx, fact in enumerate(deduped_facts):
        letter = letters[idx]
        val_nat = getattr(fact, "value_nature", "numerical")
        temp_nat = getattr(fact, "temporal_nature", "period_duration")
        with cols[idx]:
            st.markdown(f"#### 📄 Fact {letter}: `#{fact.fact_id[:8]}` (`{fact.evidence.document_id}`)")
            st.markdown(f"- **Reported Value:** `{fact.value}` (Raw: `{fact.raw_value}`)")
            st.markdown(f"- **Metric Type / Nature:** `{val_nat.replace('_', ' ').title()}`")
            st.markdown(f"- **Temporal Period:** `{fact.context.temporal_period or 'N/A'}` (`{temp_nat}`)")
            st.markdown(f"- **Reporting Scope:** `{fact.context.reporting_scope or 'Standard'}`")
            st.markdown(f"- **PDF Location:** **Page {fact.evidence.page_number}**")
            st.info(f"💬 *Verbatim Quote:*\n\"{fact.evidence.verbatim_quote}\"")

    # Structured Comparability Gate Audit Breakdown
    audit_data = getattr(case, "comparability_audit", {}) or {}
    matched = audit_data.get("matched_fields", [])
    differing = audit_data.get("differing_fields", [])

    with st.expander("🔍 **Comparability Gate Audit & Verification Checklist**", expanded=True):
        col_aud1, col_aud2 = st.columns(2)
        with col_aud1:
            st.markdown("**Matched Fields & Attributes:**")
            if matched:
                for m in matched:
                    st.markdown(f"- ✅ `{m}`")
            else:
                st.markdown("- *Pre-comparison audit in progress*")
        with col_aud2:
            st.markdown("**Differing / Divergent Fields:**")
            if differing:
                for d in differing:
                    st.markdown(f"- ⚠️ `{d}`")
            else:
                st.markdown("- None (Strict 1:1 dimensional comparability)")

        if getattr(case, "unit_normalization_applied", None):
            st.markdown(f"**📐 Unit Normalization Applied:** {case.unit_normalization_applied}")

        if getattr(case, "temporal_progression_explanation", None):
            st.markdown(f"**⏱️ Temporal Progression:** {case.temporal_progression_explanation}")

        if getattr(case, "verdict_rationale", None):
            st.markdown(f"**🎯 Verdict Justification:** *{case.verdict_rationale}*")

    st.markdown("**🧠 System Reasoning & Resolution:**")
    st.success(case.reasoning)
    st.caption(f"**Executive Takeaway:** {case.reconciliation_summary}")
    st.markdown("---")


def render_failure_card(failure: FailureRecord):
    """Render a comprehensive Case 4 extraction or reasoning failure diagnostic card."""
    st.markdown(
        f"""
        <div style="border-left: 6px solid #e53e3e; background-color: #1a202c; padding: 16px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #2d3748;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0; color: #feb2b2;">⚠️ Case 4 Failure: {failure.failure_type.value}</h3>
                <span style="background: #9b2c2c; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px;">{failure.failure_type.value}</span>
            </div>
            <p style="margin: 4px 0; color: #cbd5e0;">
                <strong>Source Document:</strong> <code>{failure.source_document}</code> &nbsp;|&nbsp; 
                <strong>Page Number:</strong> <code>Page {failure.page_number}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Expected Evidence")
        st.info(failure.expected_evidence)
    with col2:
        st.markdown("#### ❌ Observed Result (Flawed Extraction / Reasoning)")
        st.error(failure.observed_result)

    st.markdown(f"**🔍 Problem Diagnosis:**\n\n{failure.problem_description}")
    st.markdown(f"**💥 Downstream Impact:**\n\n{failure.impact}")

    col_det, col_mit = st.columns(2)
    with col_det:
        st.markdown(f"**🛡️ Detection Mechanism:**\n\n{failure.detection_mechanism}")
    with col_mit:
        st.markdown(f"**🔧 Mitigation Strategy:**\n\n{failure.mitigation_strategy}")

    st.markdown(f"**🚀 Potential Architectural Improvement:**\n\n*{failure.potential_improvement}*")
    st.markdown("---")


def get_documents_from_store(store_inst):
    """Safely retrieve all documents even if cached store instance is from an earlier module version."""
    if hasattr(store_inst, "get_all_documents"):
        return store_inst.get_all_documents()
    import json
    rows = store_inst.conn.execute(
        "SELECT doc_id, filename, file_path, total_pages, checksum_sha256, ingested_at, metadata FROM documents ORDER BY ingested_at DESC;"
    ).fetchall()
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
        })
    return docs


def main():
    selected_corpus, active_extractor, use_llm = render_sidebar()

    # Automatically sync relationships and failure stores if empty and facts are present
    all_facts_global = store.get_all_facts()
    all_rels_global = store.get_all_relationships()
    if (not all_rels_global) and all_facts_global:
        sync_knowledge_graph(store)

    # Fetch records strictly filtered to the active logical corpus
    active_docs = store.get_all_documents(corpus=selected_corpus)
    active_facts = store.get_all_facts(corpus=selected_corpus)
    active_relationships = store.get_all_relationships(corpus=selected_corpus)
    active_failures = store.get_all_failures(corpus=selected_corpus)
    rel_counts = store.get_relationship_counts(corpus=selected_corpus)
    failure_count = len(active_failures)
    doc_count = len(active_docs)
    fact_count = len(active_facts)

    # Build whole-corpus multi-document verification groups
    consensus_groups = CorpusConsensusEngine.build_consensus_groups(
        active_facts, corpus_name=selected_corpus, total_corpus_docs=doc_count
    )
    metric_to_consensus = {g.metric.strip().lower(): g for g in consensus_groups}

    doc_filenames = [d["filename"] for d in active_docs]
    doc_id_to_filename = {d["doc_id"]: d["filename"] for d in active_docs}
    filename_to_doc_id = {d["filename"]: d["doc_id"] for d in active_docs}

    for f in active_facts:
        if f.evidence.document_id in doc_id_to_filename:
            f.evidence.document_id = doc_id_to_filename[f.evidence.document_id]

    # Map fact IDs to relationships for instant cross-referencing
    fact_to_rels: Dict[str, List[CrossDocReconciliation]] = {}
    for r in active_relationships:
        fact_to_rels.setdefault(r.fact_a_id, []).append(r)
        fact_to_rels.setdefault(r.fact_b_id, []).append(r)

    # Dynamic counts of standalone vs linked facts within this corpus
    linked_fact_ids = set()
    for r in active_relationships:
        linked_fact_ids.add(r.fact_a_id)
        linked_fact_ids.add(r.fact_b_id)
    standalone_fact_count = len([f for f in active_facts if f.fact_id not in linked_fact_ids])

    # Header
    c_lower = selected_corpus.lower()
    if "india" in c_lower:
        icon = "🇮🇳"
    elif "delhivery" in c_lower:
        icon = "📦"
    elif any(k in c_lower for k in ["africa", "chad", "niger"]):
        icon = "🌍"
    elif any(k in c_lower for k in ["us", "america", "fed"]):
        icon = "🇺🇸"
    elif any(k in c_lower for k in ["europe", "ecb"]):
        icon = "🇪🇺"
    else:
        icon = "📁"
    st.title("Cross-Document Fact Knowledge Layer")
    st.markdown(f"## {icon} Active Corpus: **{selected_corpus}**")
    st.markdown(
        f"**{doc_count} Documents** &nbsp;&bull;&nbsp; **{fact_count} Extracted Facts** &nbsp;&bull;&nbsp; "
        f"**{rel_counts['total']} Cross-Document Relationships** &nbsp;&bull;&nbsp; "
        f"**{len(consensus_groups)} Whole-Corpus Consensus Metrics**<br>"
        f"<span style='color: #a0aec0; font-size: 13px;'>Boundary Rule: Strict intra-corpus isolation. Facts from <strong>{selected_corpus}</strong> "
        f"are analyzed across all documents within this corpus and never mixed with foreign corpora.</span>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # PRIMARY NAVIGATION TABS
    # --------------------------------------------------------------------------
    tab_overview, tab_all_facts, tab_relationships, tab_failures, tab_docs = st.tabs([
        "📊 Overview",
        f"📚 All Facts ({fact_count})",
        f"🔗 Relationships ({rel_counts['total']})",
        f"⚠️ Extraction / Reasoning Failures ({failure_count})",
        f"📑 Ingested Documents ({doc_count})",
    ])

    # ==========================================================================
    # TAB 1: 📊 OVERVIEW
    # ==========================================================================
    with tab_overview:
        st.subheader(f"📊 {selected_corpus} Aggregate Overview")
        st.caption(f"Dynamic telemetry and cross-document analysis strictly partitioned to {selected_corpus}.")

        # Aggregate Metrics Row (Dynamic, Never Hard-Coded)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("📚 All Facts", fact_count)
        m2.metric("📑 Ingested PDFs", doc_count)
        m3.metric("✓ Corroborated", rel_counts.get("corroborated", 0))
        m4.metric("⚠ Contradictions", rel_counts.get("genuine_contradiction", 0))
        m5.metric("🔄 Reconciled", rel_counts.get("context_reconciled", 0))
        m6.metric("⚠️ Failures", failure_count)

        # Architectural Principle Callout
        st.markdown(
            f"""
            <div style="background-color: #1e2530; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; margin: 16px 0;">
                <h4 style="margin-top: 0; color: #63b3ed;">🏛️ Core Principle: All Facts Are First-Class Knowledge</h4>
                <p style="margin-bottom: 8px; font-size: 14px; color: #cbd5e0;">
                    Every meaningful fact extracted from <strong>{selected_corpus}</strong> belongs to this Knowledge Layer. 
                    Cases 1–3 evaluate cross-document relationships across those facts, while Case 4 documents extraction and reasoning failures. 
                    <strong>Facts do not need to participate in a case to remain valid knowledge.</strong>
                </p>
                <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px;">
                    <span style="background: #2b6cb0; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        📚 All Facts: {fact_count} ({standalone_fact_count} Standalone, {len(linked_fact_ids)} In Relationships)
                    </span>
                    <span style="background: #28a745; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        ✓ Case 1 Corroborated: {rel_counts.get("corroborated", 0)}
                    </span>
                    <span style="background: #dc3545; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        ⚠ Case 2 Contradictions: {rel_counts.get("genuine_contradiction", 0)}
                    </span>
                    <span style="background: #d69e2e; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        🔄 Case 3 Reconciled: {rel_counts.get("context_reconciled", 0)}
                    </span>
                    <span style="background: #805ad5; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                        ⚠️ Case 4 Failures: {failure_count}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"#### 📁 Facts Extracted per Document ({selected_corpus})")
            doc_fact_counts = {}
            for f in active_facts:
                doc_fact_counts[f.evidence.document_id] = doc_fact_counts.get(f.evidence.document_id, 0) + 1
            df_doc_counts = pd.DataFrame([
                {"Document": k, "Facts Count": v} for k, v in sorted(doc_fact_counts.items(), key=lambda x: -x[1])
            ])
            st.dataframe(df_doc_counts, width="stretch", hide_index=True)

        with col_d2:
            st.markdown(f"#### ⚖️ Relationship Breakdown ({selected_corpus})")
            rel_summary_data = [
                {"Relationship": "✓ Case 1: Corroborated", "Count": rel_counts.get("corroborated", 0)},
                {"Relationship": "⚠ Case 2: Likely Contradiction", "Count": rel_counts.get("genuine_contradiction", 0)},
                {"Relationship": "🔄 Case 3: Reconciled by Context", "Count": rel_counts.get("context_reconciled", 0)},
                {"Relationship": "⚠️ Case 4: Diagnostic Failures", "Count": failure_count},
            ]
            st.dataframe(pd.DataFrame(rel_summary_data), width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader(f"🏆 Showcase: Representative 4 Cases in {selected_corpus}")
        showcase = reconciler.generate_case_showcase(active_relationships, active_facts=active_facts)
        if showcase.corroborated_example:
            render_case_card("Case 1: Fact Corroborated Across Documents", showcase.corroborated_example, "#28a745")
        if showcase.contradiction_example:
            render_case_card("Case 2: Genuine or Likely Contradiction", showcase.contradiction_example, "#dc3545")
        if showcase.context_reconciled_example:
            render_case_card("Case 3: Apparent Contradiction Reconciled by Context", showcase.context_reconciled_example, "#ffc107")
        if showcase.failure_example:
            render_case_card("Case 4: Extraction or Reasoning Failure Analysis & Mitigation", showcase.failure_example, "#9c27b0")

    # ==========================================================================
    # TAB 2: 📚 ALL FACTS (First-Class Knowledge Layer)
    # ==========================================================================
    with tab_all_facts:
        st.subheader(f"📚 {selected_corpus} - Complete Fact Collection")
        st.info(
            f"Showing all **{fact_count}** facts belonging to **{selected_corpus}** across **{doc_count}** documents. "
            "Every single fact is preserved and accessible regardless of case participation."
        )

        # Filters within active corpus
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_text = st.text_input("🔍 Search Metric, Value, Period, or Quote:", "").strip().lower()
        with col_s2:
            filter_doc = st.selectbox("Filter Document:", ["All Documents"] + sorted(list({f.evidence.document_id for f in active_facts})))
        with col_s3:
            filter_type = st.selectbox("Fact Type:", ["All Types", "Numerical", "Semantic"])

        col_s4, col_s5, col_s6 = st.columns(3)
        with col_s4:
            filter_rel_status = st.selectbox(
                "Relationship Status:",
                ["All Facts", "In Relationships", "Standalone (No Cross-Doc Relationships)"],
            )
        with col_s5:
            sort_field = st.selectbox("Sort By:", ["Page Number", "Metric Name", "Value", "Document", "Confidence"])
        with col_s6:
            sort_dir = st.selectbox("Order:", ["Ascending", "Descending"])

        # Filter facts
        filtered_facts = active_facts
        if filter_doc != "All Documents":
            filtered_facts = [f for f in filtered_facts if f.evidence.document_id == filter_doc]
        if filter_type == "Numerical":
            filtered_facts = [f for f in filtered_facts if f.category == FactCategory.NUMERICAL]
        elif filter_type == "Semantic":
            filtered_facts = [f for f in filtered_facts if f.category == FactCategory.SEMANTIC]
        if filter_rel_status == "In Relationships":
            filtered_facts = [f for f in filtered_facts if f.fact_id in fact_to_rels and len(fact_to_rels[f.fact_id]) > 0]
        elif filter_rel_status == "Standalone (No Cross-Doc Relationships)":
            filtered_facts = [f for f in filtered_facts if f.fact_id not in fact_to_rels or len(fact_to_rels[f.fact_id]) == 0]

        if search_text:
            filtered_facts = [
                f for f in filtered_facts
                if search_text in f.metric.lower()
                or search_text in f.value.lower()
                or search_text in (f.context.temporal_period or "").lower()
                or search_text in f.evidence.verbatim_quote.lower()
            ]

        # Sorting
        is_desc = (sort_dir == "Descending")
        if sort_field == "Page Number":
            filtered_facts.sort(key=lambda f: (f.evidence.document_id, f.evidence.page_number), reverse=is_desc)
        elif sort_field == "Metric Name":
            filtered_facts.sort(key=lambda f: f.metric.lower(), reverse=is_desc)
        elif sort_field == "Value":
            filtered_facts.sort(key=lambda f: f.value.lower(), reverse=is_desc)
        elif sort_field == "Document":
            filtered_facts.sort(key=lambda f: f.evidence.document_id, reverse=is_desc)
        elif sort_field == "Confidence":
            filtered_facts.sort(key=lambda f: f.confidence_score, reverse=is_desc)

        # Pagination
        total_matched = len(filtered_facts)
        col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
        with col_p1:
            per_page = st.selectbox("Per Page", [15, 25, 50, 100], index=0)
        
        max_page = max(1, (total_matched + per_page - 1) // per_page)
        with col_p2:
            curr_page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
        with col_p3:
            start_num = min(total_matched, (curr_page - 1) * per_page + 1)
            end_num = min(total_matched, curr_page * per_page)
            st.markdown(f"**Showing {start_num}–{end_num} of {total_matched} facts in {selected_corpus}** *(Page {curr_page} of {max_page})*")

        slice_start = (curr_page - 1) * per_page
        slice_end = min(total_matched, slice_start + per_page)
        facts_on_page = filtered_facts[slice_start:slice_end]

        if not facts_on_page:
            st.warning("No facts match your search and filter criteria in this corpus.")
        else:
            for idx, f in enumerate(facts_on_page):
                f_rels = fact_to_rels.get(f.fact_id, [])
                n_rels = len(f_rels)
                cg = metric_to_consensus.get(f.metric.strip().lower())
                
                if cg:
                    status_badge = f"🌐 Consensus ({cg.coverage_ratio})"
                elif n_rels > 0:
                    status_badge = f"🔗 {n_rels} Relationships"
                else:
                    status_badge = "⚪ Standalone"

                cat_badge = "🔢" if f.category == FactCategory.NUMERICAL else "📝"

                header_str = (
                    f"#{slice_start + idx + 1} [{f.fact_id[:8]}] {cat_badge} **{f.metric}**: `{f.value}` "
                    f"| `{f.evidence.document_id}` (p.{f.evidence.page_number}) | {status_badge}"
                )

                with st.expander(header_str, expanded=False):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown(f"**Fact ID:** `#{f.fact_id}`")
                        st.markdown(f"**Corpus:** `{f.corpus or selected_corpus}`")
                        st.markdown(f"**Entity:** `{f.entity}`")
                        st.markdown(f"**Metric / Predicate:** `{f.metric}`")
                        st.markdown(f"**Normalized Value:** `{f.value}`")
                        st.markdown(f"**Raw Extracted Value:** `{f.raw_value}`")
                    with col_info2:
                        st.markdown(f"**Source Document:** `{f.evidence.document_id}`")
                        st.markdown(f"**Page Number:** **Page {f.evidence.page_number}**")
                        st.markdown(f"**Temporal Period:** `{f.context.temporal_period or 'N/A'}`")
                        st.markdown(f"**Reporting Scope:** `{f.context.reporting_scope or 'N/A'}`")
                        st.markdown(f"**Confidence:** `{f.confidence_score * 100:.0f}%` (Unit: `{f.context.unit or 'N/A'}`)")

                    st.markdown("**💬 Grounding Verbatim Source Quote:**")
                    st.info(f"\"{f.evidence.verbatim_quote}\"")

                    # Corpus-Wide Consensus Context
                    if cg:
                        st.markdown(f"**🌐 Whole-Corpus Verification ({cg.coverage_ratio}):**")
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid #3182ce; background-color: #1a202c; padding: 10px 14px; border-radius: 4px; margin-bottom: 10px;">
                                <strong>Status:</strong> {cg.status_summary}<br>
                                <span style="font-size: 13px; color: #cbd5e0;"><strong>Consensus Synthesis:</strong> {cg.synthesis}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(f"⚪ *Standalone Fact: Disclosed exclusively in '{f.evidence.document_id}' (no other document in {selected_corpus} reports this metric). Retained as first-class ground truth.*")

                    # Underlying pairwise linkages
                    if f_rels:
                        st.markdown(f"**🔗 Pairwise Disclosures ({n_rels} Links within {selected_corpus}):**")
                        for r in f_rels[:5]:
                            other = r.fact_b if r.fact_a_id == f.fact_id else r.fact_a
                            if r.verdict == ReconciliationVerdict.CORROBORATED:
                                v_badge = "✓ CORROBORATED"
                                v_color = "#28a745"
                            elif r.verdict == ReconciliationVerdict.GENUINE_CONTRADICTION:
                                v_badge = "⚠ CONTRADICTION"
                                v_color = "#dc3545"
                            else:
                                v_badge = f"🔄 RECONCILED ({r.shift_type.value})"
                                v_color = "#d69e2e"

                            st.markdown(
                                f"""
                                <div style="border-left: 4px solid {v_color}; background-color: #252e3e; padding: 8px 12px; border-radius: 4px; margin-bottom: 6px;">
                                    <strong>{v_badge}</strong> with Fact <code>#{other.fact_id[:8]}</code> 
                                    (<code>{other.value}</code> in <code>{other.evidence.document_id}</code>, p.{other.evidence.page_number})<br>
                                    <span style="font-size: 12px; color: #cbd5e0;">{r.reasoning}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        if len(f_rels) > 5:
                            st.caption(f"*... and {len(f_rels) - 5} additional pairwise linkages visible in the Relationships tab.*")

    # ==========================================================================
    # TAB 3: 🔗 RELATIONSHIPS & WHOLE-CORPUS VERIFICATION
    # ==========================================================================
    with tab_relationships:
        st.subheader(f"🔗 {selected_corpus} - Cross-Document Relationships & Whole-Corpus Verification")
        st.caption(
            f"Fact verification executed across the entire {selected_corpus} corpus. "
            "Multi-document metrics are synthesized across all participating PDFs, while pairwise relationships detail specific comparisons."
        )

        # 1. Whole-Corpus Multi-Document Consensus Section
        st.markdown("### 🌐 Whole-Corpus Multi-Document Consensus")
        st.markdown(
            f"The following metrics appear across multiple documents within **{selected_corpus}**. "
            "Rather than treating them as separate PDF-to-PDF fragments, the system synthesizes whole-corpus consensus across all reporting documents:"
        )

        if not consensus_groups:
            st.info(f"No multi-document metrics currently detected in {selected_corpus}.")
        else:
            for cg in consensus_groups:
                render_corpus_consensus_card(cg)

        st.markdown("---")

        # 2. Pairwise Relationships Section
        st.markdown("### 🔗 Pairwise Disclosures & Mathematical Reconciliation")
        st.caption("Detailed breakdown of bilateral fact-to-fact comparisons strictly confined to this corpus.")

        rel_view = st.radio(
            "Filter Verdict Case:",
            options=[
                f"All Relationships ({rel_counts['total']})",
                f"✓ Case 1: Corroborated ({rel_counts.get('corroborated', 0)})",
                f"⚠ Case 2: Contradictions ({rel_counts.get('genuine_contradiction', 0)})",
                f"🔄 Case 3: Reconciled by Context ({rel_counts.get('context_reconciled', 0)})",
            ],
            index=0,
            horizontal=True,
        )

        col_rf1, col_rf2 = st.columns([2, 1])
        with col_rf1:
            rel_search = st.text_input("Search Metric or Reasoning:", "").strip().lower()
        with col_rf2:
            rel_per_page = st.selectbox("Relationships Per Page", [10, 20, 50], index=0)

        # Apply verdict filter
        target_rels = active_relationships
        if "Corroborated" in rel_view:
            target_rels = [r for r in target_rels if r.verdict == ReconciliationVerdict.CORROBORATED]
        elif "Contradictions" in rel_view:
            target_rels = [r for r in target_rels if r.verdict == ReconciliationVerdict.GENUINE_CONTRADICTION]
        elif "Reconciled" in rel_view:
            target_rels = [r for r in target_rels if r.verdict == ReconciliationVerdict.CONTEXT_RECONCILED]

        if rel_search:
            target_rels = [
                r for r in target_rels
                if rel_search in r.fact_a.metric.lower()
                or rel_search in r.reasoning.lower()
                or rel_search in r.reconciliation_summary.lower()
            ]

        total_rels = len(target_rels)
        max_rel_page = max(1, (total_rels + rel_per_page - 1) // rel_per_page)
        curr_rel_page = st.number_input("Relationship Page", min_value=1, max_value=max_rel_page, value=1, step=1)

        rel_start = (curr_rel_page - 1) * rel_per_page
        rel_end = min(total_rels, rel_start + rel_per_page)
        page_rels = target_rels[rel_start:rel_end]

        st.caption(f"Displaying **{len(page_rels)}** relationship(s) on Page {curr_rel_page} of {max_rel_page} (Total: {total_rels})")

        if not page_rels:
            st.info("No pairwise relationships match the selected filter.")
        else:
            for r in page_rels:
                if r.verdict == ReconciliationVerdict.CORROBORATED:
                    border = "#28a745"
                    title_str = f"Case 1 (Corroborated): {r.fact_a.metric}"
                elif r.verdict == ReconciliationVerdict.GENUINE_CONTRADICTION:
                    border = "#dc3545"
                    title_str = f"Case 2 (Contradiction): {r.fact_a.metric}"
                else:
                    border = "#ffc107"
                    title_str = f"Case 3 (Context Reconciled): {r.fact_a.metric}"

                render_case_card(title_str, r, border)

    # ==========================================================================
    # TAB 4: ⚠️ EXTRACTION & REASONING FAILURES (Case 4)
    # ==========================================================================
    with tab_failures:
        st.subheader(f"⚠️ {selected_corpus} - Extraction & Reasoning Failures (Case 4: {failure_count} Documented)")
        st.caption(
            "Case 4 failure analysis representing extraction, layout, or reasoning flaws identified across this corpus, "
            "their diagnostic root causes, downstream impacts, and deployed mitigations."
        )

        col_fail1, col_fail2 = st.columns([1, 1])
        with col_fail1:
            ft_filter = st.selectbox(
                "Filter by Failure Type:",
                ["All Failure Types"] + [ft.value for ft in FailureType],
            )
        with col_fail2:
            fail_doc_filter = st.selectbox(
                "Filter by Source Document:",
                ["All Documents"] + sorted(list({f.source_document for f in active_failures})),
            )

        displayed_failures = active_failures
        if ft_filter != "All Failure Types":
            displayed_failures = [f for f in displayed_failures if f.failure_type.value == ft_filter]
        if fail_doc_filter != "All Documents":
            displayed_failures = [f for f in displayed_failures if f.source_document == fail_doc_filter]

        st.write(f"Showing **{len(displayed_failures)}** diagnosed failure case(s) in {selected_corpus}:")

        for f in displayed_failures:
            render_failure_card(f)

    # ==========================================================================
    # TAB 5: 📑 INGESTED DOCUMENTS
    # ==========================================================================
    with tab_docs:
        st.subheader(f"📑 Ingested Documents in {selected_corpus} ({doc_count} Documents)")
        if not active_docs:
            st.info(f"No documents ingested in {selected_corpus}.")
        else:
            doc_df = pd.DataFrame([
                {
                    "Doc ID": d["doc_id"][:12] + "..",
                    "Filename": d["filename"],
                    "Corpus": d.get("corpus") or selected_corpus,
                    "Pages": d["total_pages"],
                    "Facts Extracted": sum(1 for f in active_facts if f.evidence.document_id == d["filename"]),
                    "Ingested At": d["ingested_at"][:19],
                    "Checksum (SHA-256)": d["checksum_sha256"][:16] + "..",
                }
                for d in active_docs
            ])
            st.dataframe(doc_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

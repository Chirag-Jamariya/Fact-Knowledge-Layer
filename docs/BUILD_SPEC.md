# AGENT_BUILD_SPEC: Cross-Document Fact Knowledge Layer
**Target System**: Superjoin Engineering Intern Hiring Assignment  
**Execution Mode**: Autonomous Agent Implementation Guide  
**Runtime**: Python 3.12+ managed via `uv` on Arch Linux  
**Deliverable**: Production-grade prototype with Ingestion, Fact Extraction, In-Corpus Cross-Document Reconciliation, FastAPI Backend, and Streamlit Inspection UI.

---

## 1. System Vision & Problem Statement Alignment

### 1.1 Assignment Core Requirements
Build an in-corpus **Fact Knowledge Layer** across multi-page document filings (such as corporate annual reports, earnings presentations, and macroeconomic surveys).
The system must:
1. Ingest multi-page PDFs (typically 25–100+ pages) without relying on hardcoded facts, schemas, or document-specific heuristics.
2. Extract meaningful **numerical** and **semantic** facts grounded directly in source documents.
3. Link every fact to verifiable evidence: `[Document Name, Page Number, Section / Table / Cell Context, Verbatim Source Quote]`.
4. Group and compare facts across documents to classify their cross-document relationship into the **Four Required Cases**:
   - **Case 1: Corroborated across documents**: Facts agreeing even if worded differently.
   - **Case 2: Genuine / Likely Contradiction**: Conflicting facts on the same entity, metric, and time period under identical scope.
   - **Case 3: Apparent Contradiction Reconciled by Context**: Differing facts resolved by **Time Vintage** (e.g. FY22 vs FY24), **Reporting Scope** (e.g. Standalone vs Consolidated), or **Units / Denominations** (e.g. ₹ Crores vs Millions, USD vs INR).
   - **Case 4: Extraction or Reasoning Failure Analysis**: Detecting and explaining OCR errors, ambiguous sentences, or parsing errors, along with handling/mitigation strategies.
5. Provide a simple **Upload & Inspection Interface** (Streamlit UI + FastAPI backend).

### 1.2 Explicit Architectural Decisions (Option A Confirmations)
- **NO Voice / Speech Ingestion**: Whisper and ASR dependencies are stripped.
- **NO Video Ingestion**: OpenCV frame sampling and video pipelines are stripped.
- **NO Open-Web Crawling / Google Search**: Serper API, external web scrapers, and Google search are stripped. All evidence is discovered **strictly in-corpus** across the uploaded PDFs.
- **NO Generic Truth Ratio**: The single-document "truth ratio" $S/(S+R)$ is replaced by a structured **Cross-Document Fact Knowledge Matrix / Graph** showing pairwise reconciliation and source provenance.

---

## 2. Arch Linux & `uv` Environment Setup

Arch Linux ships with rolling Python releases (e.g. Python 3.14+), which can cause compatibility issues with C-extensions and prebuilt ML wheels. **The agent MUST use `uv` to pin and manage Python 3.12**.

### 2.1 Initialization Commands
```bash
# Verify uv installation
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project root
cd /home/hot-coffee/project

# Initialize uv project with Python 3.12
uv init --python 3.12 --app .

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate environment (or run commands with `uv run`)
source .venv/bin/activate
```

### 2.2 Dependency Specification (`pyproject.toml`)
The agent should maintain the following dependencies in `pyproject.toml` or install them using `uv add`:

```toml
[project]
name = "fact-knowledge-layer"
version = "0.1.0"
description = "Cross-Document Fact Knowledge Layer with Contextual Reconciliation"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    # PDF Ingestion & OCR/Layout
    "pypdf>=5.1.0",
    "pymupdf>=1.25.0",           # High-performance fitz PDF parser & page rendering
    "pdfplumber>=0.11.0",         # High-fidelity tabular extraction
    "pytesseract>=0.3.10",        # Fallback image OCR for visual charts
    "pillow>=11.0.0",

    # Data Modeling & Validation
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",

    # LLM & Embedding Runtime
    "openai>=1.60.0",             # Unified client for OpenAI / OpenRouter / vLLM
    "litellm>=1.59.0",            # Polymorphic LLM router (Gemini, Claude, GPT, Ollama)
    "sentence-transformers>=3.4.0", # Local dense embeddings (e.g. bge-small-en-v1.5)
    "numpy>=1.26.0,<2.0.0",

    # In-Corpus Storage & Vector Search
    "duckdb>=1.1.3",              # Fast embedded SQL/Vector store for facts & metadata
    "chromadb>=0.6.0",            # Lightweight vector retrieval store

    # API & Serving
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "python-multipart>=0.0.20",

    # Interactive UI
    "streamlit>=1.41.0",

    # Utilities
    "rich>=13.9.0",
    "typer>=0.15.0",
    "httpx>=0.28.0",
    "python-dotenv>=1.0.1",
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0"
]
```

To sync all dependencies:
```bash
uv sync
```

---

## 3. Project File & Directory Structure

The agent must structure the repository according to clean separation of concerns:

```
/home/hot-coffee/project/
├── .env.example                     # Environment variable template (OPENAI_API_KEY, etc.)
├── pyproject.toml                   # Project dependencies and tool configs
├── uv.lock                          # Deterministic dependency lockfile
├── README.md                        # Submission documentation adhering to assignment rubric
├── BUILD_SPEC.md                    # This master build specification
├── data/
│   ├── starter_datasets/            # Symlink or copies of Delhivery and India-Macro PDFs
│   ├── uploads/                     # Temp storage for user-uploaded PDFs
│   └── knowledge_store/             # Persistent DuckDB & Chroma vector artifacts
├── src/
│   ├── __init__.py
│   ├── config.py                    # App configuration, LLM parameters, model selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fact.py                  # Pydantic schemas: Fact, Context, SourceEvidence
│   │   ├── reconciliation.py        # Pydantic schemas: Verdict, ReconciliationResult
│   │   └── corpus.py                # Pydantic schemas: DocumentRecord, PageChunk
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py            # PyMuPDF + pdfplumber dual text & table extractor
│   │   ├── table_extractor.py       # Converts table grids to structured markdown/JSON
│   │   └── image_extractor.py       # Extracts embedded figures and charts with page tags
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── fact_extractor.py        # LLM prompt pipeline extracting atomic facts + context
│   │   └── noise_filter.py          # Drops boilerplates, legal disclaimers, forward-looking fluff
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── store.py                 # DuckDB engine storing facts, embeddings, and relationships
│   │   └── retriever.py             # Hybrid (semantic + keyword) cross-document retriever
│   ├── reconciliation/
│   │   ├── __init__.py
│   │   ├── clusterer.py             # Groups candidate facts across docs by Entity & Metric
│   │   ├── reconciler.py            # Core engine evaluating the 4 cases with CoT reasoning
│   │   └── failure_detector.py      # Identifies OCR corruptions, ambiguous units, parsing bugs
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application entrypoint
│   │   └── routes.py                # Upload, query facts, inspect reconciliations
│   └── ui/
│       ├── __init__.py
│       └── app.py                   # Streamlit interactive application
└── tests/
    ├── test_ingestion.py
    ├── test_extraction.py
    ├── test_reconciliation.py
    └── test_cases_starter_dataset.py # Automated test verifying the 4 showcase cases
```

---

## 4. Core Data Schemas (Pydantic Models)

The agent must define strongly typed data models in `src/models/`:

### 4.1 Source Evidence & Fact Schema (`src/models/fact.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class FactCategory(str, Enum):
    NUMERICAL = "numerical"        # Financial metrics, revenue, headcounts, percentages
    SEMANTIC = "semantic"          # Key executives, acquisitions, company locations, policies

class SourceEvidence(BaseModel):
    document_id: str = Field(description="Unique identifier or filename of the PDF")
    page_number: int = Field(description="1-indexed PDF page number")
    section_title: Optional[str] = Field(default=None, description="Header or section name")
    content_type: str = Field(default="text", description="text | table | chart_caption")
    verbatim_quote: str = Field(description="Exact snippet or table row containing the fact")
    bounding_box: Optional[Dict[str, float]] = Field(default=None, description="Optional [x0, y0, x1, y1] on page")

class ContextDimensions(BaseModel):
    temporal_period: Optional[str] = Field(default=None, description="e.g., 'FY 2023-24', 'Q4 FY24', '2022', 'Calendar Year 2024'")
    reporting_scope: Optional[str] = Field(default=None, description="e.g., 'Consolidated', 'Standalone', 'India Operations', 'Global'")
    unit: Optional[str] = Field(default=None, description="e.g., 'INR Crores', 'Millions', 'Percentage', 'Count'")
    accounting_standard: Optional[str] = Field(default=None, description="e.g., 'Ind AS', 'IFRS', 'US GAAP'")

class Fact(BaseModel):
    fact_id: str = Field(description="Unique deterministic hash of entity + metric + doc + page")
    category: FactCategory
    entity: str = Field(description="Canonical entity (e.g. 'Delhivery Limited', 'Indian Economy')")
    metric: str = Field(description="Normalized property (e.g. 'Revenue from Operations', 'EBITDA Margin', 'Headline CPI')")
    value: str = Field(description="Normalized value string (e.g. '8,142 Cr', '5.4%', '740 Million')")
    raw_value: str = Field(description="Raw literal as printed in the text")
    context: ContextDimensions
    evidence: SourceEvidence
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
```

### 4.2 Reconciliation Verdicts Schema (`src/models/reconciliation.py`)
```python
class ReconciliationVerdict(str, Enum):
    CORROBORATED = "corroborated"
    GENUINE_CONTRADICTION = "genuine_contradiction"
    CONTEXT_RECONCILED = "context_reconciled"
    EXTRACTION_FAILURE = "extraction_failure"

class ContextualShiftType(str, Enum):
    TEMPORAL_SHIFT = "temporal_shift"          # e.g., FY22 vs FY24 or Q3 vs Q4
    SCOPE_SHIFT = "scope_shift"                # e.g., Standalone vs Consolidated
    UNIT_DENOMINATION = "unit_denomination"    # e.g., Crores vs Millions
    METHODOLOGY_VINTAGE = "methodology_vintage"# e.g., Revised Estimate vs Actual
    NONE = "none"

class CrossDocReconciliation(BaseModel):
    reconciliation_id: str
    verdict: ReconciliationVerdict
    shift_type: ContextualShiftType = ContextualShiftType.NONE
    fact_a: Fact
    fact_b: Fact
    reasoning: str = Field(description="Detailed natural-language justification of why facts agree, conflict, or reconcile")
    reconciliation_summary: str = Field(description="Executive 1-sentence takeaway")
```

---

## 5. Subsystem Architecture & Implementation Details

### 5.1 Subsystem 01: PDF Document Parser (`src/ingestion/`)
- **Strategy**: Dual engine. Use `pymupdf` (fitz) for high-speed layout rendering and text extraction with page numbers, and `pdfplumber` for tabular data structures.
- **Table Handling**: Tables are the primary source of financial and economic facts. `pdfplumber.extract_tables()` extracts rows, converts them to Markdown tables with header preservation, and associates each row with the document name and page number.
- **Image/Chart Capture**: Extract embedded image objects via PyMuPDF. If an image has an adjacent caption (e.g. "Figure 3.2: Inflation Trends"), log the caption as semantic evidence.

### 5.2 Subsystem 02: Fact Extraction Layer (`src/extraction/`)
- **Chunking**: Chunk documents by logical sections or pages with 200-token overlap, retaining metadata (`{doc_id, page_number}`).
- **LLM Extraction Prompt**: Instruct the model to extract atomic triples `(Entity, Metric, Value)` along with `ContextDimensions` (`temporal_period`, `reporting_scope`, `unit`).
- **Noise Suppression**:
  - Filter out generic corporate statements ("we strive to be leaders").
  - Filter out forward-looking guidance without concrete commitments ("we expect to grow in the future").
  - Retain concrete numerical facts (financials, operational stats, growth rates) and high-value semantic facts (directors, acquisitions, facility locations).

### 5.3 Subsystem 03: In-Corpus Store & Retrieval (`src/indexing/`)
- **DuckDB Catalog**: Stores extracted facts in a relational table for rapid SQL querying:
  `facts(fact_id, doc_id, page_number, entity, metric, value, period, scope, unit, quote)`
- **Vector Embeddings**: Compute embeddings on `f"{entity} {metric}"` using a local model (`BAAI/bge-small-en-v1.5` via `sentence-transformers`) or LiteLLM embedding call.
- **Cross-Document Fact Clustering**:
  - Group facts sharing identical or semantically similar `(Entity, Metric)` across *different* `document_id`s.
  - This forms the candidate pairs `(fact_a, fact_b)` for cross-document reconciliation.

### 5.4 Subsystem 04: The 4-Case Reconciliation Engine (`src/reconciliation/`)
For each candidate pair `(fact_a, fact_b)` from distinct documents:
1. **Case 1 Check (Corroboration)**:
   - If entity, metric, temporal period, and scope match, and the values are identical (or mathematically equivalent after unit conversion, e.g. 74 Cr vs 740 Million) $\rightarrow$ `CORROBORATED`.
2. **Case 3 Check (Apparent Contradiction Reconciled by Context)**:
   - If values differ, but there is an explicit difference in:
     - **Time Vintage**: e.g., Fact A is for FY 2021-22, Fact B is for FY 2023-24 $\rightarrow$ `CONTEXT_RECONCILED (temporal_shift)`.
     - **Scope**: e.g., Fact A is Standalone, Fact B is Consolidated $\rightarrow$ `CONTEXT_RECONCILED (scope_shift)`.
     - **Denomination**: e.g., Value is stated in USD vs INR $\rightarrow$ `CONTEXT_RECONCILED (unit_denomination)`.
3. **Case 2 Check (Genuine Contradiction)**:
   - If entity, metric, time period, and scope all match, but values diverge without any contextual reason $\rightarrow$ `GENUINE_CONTRADICTION`.
4. **Case 4 Check (Extraction or Reasoning Failure)**:
   - If the fact was derived from a truncated table, OCR error, ambiguous pronoun, or hallucinated unit $\rightarrow$ `EXTRACTION_FAILURE`.
   - The engine logs the failure cause and suggests/executes the fallback mitigation (e.g., regex re-extraction, raw cell inspection).

### 5.5 Subsystem 05: FastAPI Backend (`src/api/`)
Endpoints:
- `POST /api/documents/upload`: Upload one or multiple PDFs. Triggers parsing, fact extraction, and indexing.
- `GET /api/documents`: List uploaded documents and page counts.
- `GET /api/facts`: Query extracted facts with filters (`doc_id`, `entity`, `metric`, `category`).
- `GET /api/reconciliations`: Retrieve cross-document reconciliation results, filterable by the 4 required cases.
- `GET /api/cases/showcase`: Retrieve the curated demonstration of the 4 required cases.

### 5.6 Subsystem 06: Streamlit Inspection UI (`src/ui/app.py`)
Interactive features:
1. **Document Uploader**: Drag and drop PDFs with instant processing status and progress bar.
2. **Four Required Cases Showcase Tab**:
   - Tab 1: **Corroborated Facts** (Side-by-side card comparison with Doc Name and Page citation).
   - Tab 2: **Genuine Contradictions** (Highlighted discrepancies with verbatim quotes).
   - Tab 3: **Reconciled by Context** (Explains why numbers differ: timeline badge, scope badge, unit conversion).
   - Tab 4: **Extraction / Reasoning Failures** (Shows what failed, why it failed, and how the system mitigates it).
3. **Fact Explorer Tab**: Searchable table of all extracted facts with filters for entity, metric, and document.
4. **Interactive Document Viewer / Citation Inspector**: Clicking on a fact displays the exact source page number and verbatim excerpt.

---

## 6. Validation on Starter Datasets

The repository includes starter datasets at `/home/hot-coffee/Downloads/starter-datasets`:
1. **Delhivery Dataset**:
   - `01-delhivery-prospectus-2022-excerpt.pdf` (100 pages)
   - `02-delhivery-annual-report-fy24-excerpt.pdf` (100 pages)
   - `03-delhivery-q4-fy24-earnings-presentation.pdf` (27 pages)
2. **India Macroeconomy Dataset**:
   - `01-india-economic-survey-2024-25-excerpt.pdf` (89 pages)
   - `02-rbi-annual-report-2024-25-excerpt.pdf` (100 pages)
   - `03-imf-india-2025-article-iv-excerpt.pdf` (95 pages)

### Automated Test Suite (`tests/test_cases_starter_dataset.py`)
The agent must include tests that execute against the starter dataset and assert that:
- At least 1 valid **Corroborated** case is discovered and verified.
- At least 1 valid **Genuine / Likely Contradiction** case is discovered and verified.
- At least 1 valid **Apparent Contradiction Reconciled by Context** is identified (e.g. FY22 vs FY24 revenue, or Standalone vs Consolidated numbers).
- At least 1 **Extraction Failure** is surfaced with reasoning and mitigation.

---

## 7. Submission Artifacts & README Specification

The final `README.md` must adhere to the exact structure required by the hiring team:
1. **Setup and Run Instructions**: How to set up with `uv` on Linux/macOS and launch the backend and UI.
2. **Video Demo Link**: Placeholder/link for the <= 3-minute video demo.
3. **Approach**: Architecture breakdown, key trade-offs (e.g. in-corpus vs web search, schema design, table extraction strategies), and AI models used.
4. **Limitations and Next Steps**: Discussion of failure modes, scaling to 10,000+ PDFs, and incremental indexing.
5. **Additional Notes**: Documentation of the 4 required cases with exact page citations from the starter dataset.

---

## 8. Agent Execution Checklist

When coding autonomously, the agent should follow this sequence:
- [ ] Initialize Python 3.12 environment using `uv venv --python 3.12`.
- [ ] Install dependencies from `pyproject.toml` using `uv sync`.
- [ ] Build data models in `src/models/`.
- [ ] Implement PDF and table parser in `src/ingestion/`.
- [ ] Implement fact extractor with context dimensions in `src/extraction/`.
- [ ] Implement DuckDB store and cross-document candidate clusterer in `src/indexing/`.
- [ ] Implement the 4-case reconciliation engine in `src/reconciliation/`.
- [ ] Build FastAPI routes in `src/api/`.
- [ ] Build Streamlit dashboard in `src/ui/`.
- [ ] Run extraction on `starter-datasets/delhivery` or `starter-datasets/india-macroeconomy`.
- [ ] Verify that all 4 required cases are cleanly demonstrated with exact page numbers.
- [ ] Write comprehensive `README.md` meeting all hiring assignment criteria.

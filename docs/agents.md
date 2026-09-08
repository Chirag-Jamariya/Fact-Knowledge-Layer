# AGENTS.md: Development Protocol & Governance Guidelines

## 1. Non-Negotiable Operational Mandate

This project follows an iterative, component-by-component construction lifecycle with strict human-in-the-loop manual verification gates.

### Core Rules:
1. **Isolated Component Construction**:
   - Each architectural component must be built, tested, and demonstrated individually in isolation.
   - No component shall assume unverified behavior from downstream or upstream modules.
2. **Mandatory Walkthrough Documentation**:
   - After completing any individual component or integration step, generate/update `docs/walkthrough.md`.
   - The walkthrough must contain:
     - Component objective and scope.
     - Exact shell commands to execute (using `uv run`).
     - Test input data and reproducible reproduction commands.
     - Expected terminal/visual output and assertions.
     - Known limitations or edge cases handled.
3. **Mandatory Halting & Approval Gate**:
   - **HALT** after each individual component is completed. Do NOT proceed to the next component automatically.
   - Present the walkthrough to the USER and request manual verification and explicit approval.
   - Only upon receiving explicit user confirmation/approval may development proceed.
4. **Component Integration Halting Gate**:
   - Whenever two individual components are wired/integrated together, the agent must **HALT** again.
   - Update `docs/walkthrough.md` with the end-to-end integration test commands and expected outputs.
   - Wait for explicit user manual verification and approval before building subsequent components.
5. **Documentation Directory Standard**:
   - All markdown specification, planning, and walkthrough documents must reside exclusively in the `docs/` directory.
   - Top-level root directory contains only configuration (`pyproject.toml`, `uv.lock`, `.env.example`), execution entrypoints, and the primary project `README.md`.

---

## 2. Component Pipeline Sequence

| Phase | Subsystem / Component | Description | Verification Milestone |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Environment & Core Schemas** | Python 3.12 via `uv`, `pyproject.toml`, Pydantic models in `src/models/` | Type validation, serialization, schema consistency |
| **Phase 2** | **PDF Ingestion Engine** | `src/ingestion/` dual PyMuPDF + pdfplumber text, table, and page metadata extraction | Extraction on 100-page starter PDF with page numbering & table grids |
| **Phase 3** | **Fact Extraction Layer** | `src/extraction/` atomic numerical & semantic fact extraction with context dimensions | Extraction of atomic triples + temporal/scope/unit metadata |
| **Gate 1** | **Integration 1 (Ingestion + Extraction)** | End-to-end PDF-to-Fact pipeline on real PDF pages | End-to-end extraction from starter dataset with source citations |
| **Phase 4** | **In-Corpus Storage & Retrieval** | `src/indexing/` DuckDB storage, hybrid search, and cross-document candidate clustering | Clustering facts by Entity & Metric across distinct documents |
| **Gate 2** | **Integration 2 (Extraction + Indexing)** | Pipeline from PDFs to populated, searchable cross-document DuckDB store | Querying facts by entity/metric across multiple PDFs |
| **Phase 5** | **4-Case Reconciliation Engine** | `src/reconciliation/` evaluating Corroboration, Contradiction, Context Reconciled, & Failures | Demonstration of the 4 assignment cases on real candidate pairs |
| **Gate 3** | **Integration 3 (Indexing + Reconciliation)** | Automated clustering and pairwise evaluation across the full corpus | Automated generation of the 4-case knowledge matrix |
| **Phase 6** | **FastAPI Backend Service** | `src/api/` REST API for document uploads, fact querying, and reconciliation inspection | Interactive OpenAPI/Swagger docs test and curl checks |
| **Phase 7** | **Streamlit Inspection Dashboard** | `src/ui/` Interactive web UI for uploading PDFs, side-by-side evidence inspection, and 4-case showcase | Browser inspection of facts, page quotes, and reconciliation cards |
| **Gate 4** | **Final System Integration & Starter Dataset Benchmark** | Full validation on Delhivery and India-Macro datasets + `pytest` test suite | All 4 cases verified on starter dataset; assignment README finalized |

---

## 3. Environment & Execution Conventions
- **OS**: Arch Linux
- **Package & Environment Manager**: `uv`
- **Python Version**: Pinned to 3.12 (`uv venv --python 3.12`)
- **Execution Style**: All commands must run through `uv run` or within an active `.venv`.

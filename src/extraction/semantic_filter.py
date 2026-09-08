"""Semantic text filter using SentenceTransformer to retain only high-value knowledge content."""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.models.corpus import PageContent, ExtractedTable
from src.extraction.noise_filter import NoiseFilter

logger = logging.getLogger(__name__)

# Core semantic anchor concepts representing meaningful enterprise & economic knowledge
SEMANTIC_KNOWLEDGE_ANCHORS = [
    "key financial performance, revenue from operations, profit, loss, EBITDA, margin, cash flows, debt, expenditure",
    "macroeconomic indicators, real GDP growth rate, headline inflation, CPI, consumer price index, fiscal deficit, monetary policy",
    "operational statistics, package shipments volume, delivery centres, pin codes covered, active customers, workforce strength",
    "sector specific growth, oil GDP growth, non-oil sector growth, compound average growth rate, multilateral convergence target",
    "economic definitions, technical terms, policy interest rate, projections, forecasts, formulas, and analytical conclusions",
]

# Patterns for high-value factual indicators (numbers, currencies, percentages, dates)
FACTUAL_SIGNAL_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|bps|basis\s+points)\b|"
    r"(?:₹|\$|INR|USD|Rs\.?)\s*\d+(?:\.\d+)?\s*(?:Cr|Crore|Crores|Mn|Million|Bn|Billion|Trillion)?|"
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|"
    r"\b(?:FY|Q[1-4]|20\d{2}[–\-]\d{2,4}|20\d{2})\b|"
    r"\b(?:revenue|gdp|ebitda|cpi|inflation|shipment|deficit|growth|margin|volume|assets|liabilities)\b"
    r")",
    re.IGNORECASE,
)

# Unnecessary boilerplate patterns to immediately reject
BOILERPLATE_REJECT_PATTERN = re.compile(
    r"(?:"
    r"^(?:page\s+\d+|\d+\s+of\s+\d+|\-\s*\d+\s*\-)$|"
    r"(?:please\s+read\s+section|prospectus\s+dated|corporate\s+identity\s+number|scan\s+this\s+qr\s+code)|"
    r"(?:all\s+rights\s+reserved|confidential\s+and\s+proprietary|printed\s+on\s+recycled\s+paper)|"
    r"(?:this\s+page\s+is\s+intentionally\s+left\s+blank|the\s+accompanying\s+notes\s+are\s+an\s+integral)|"
    r"^(?:figure|table)\s+\d+[:.]?\s*$"
    r")",
    re.IGNORECASE,
)


class SemanticTextFilter:
    """Filters raw document pages to extract only high-value semantic knowledge text."""

    _model = None
    _anchor_embeddings = None

    @classmethod
    def get_model(cls):
        """Lazy loader for SentenceTransformer to avoid memory overhead until needed."""
        if cls._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer("all-MiniLM-L6-v2")
                cls._anchor_embeddings = cls._model.encode(
                    SEMANTIC_KNOWLEDGE_ANCHORS, normalize_embeddings=True
                )
                logger.info("SentenceTransformer all-MiniLM-L6-v2 initialized for semantic filtering.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer: {e}. Falling back to heuristic filtering.")
                cls._model = False
        return cls._model

    @classmethod
    def score_sentences(cls, sentences: List[str]) -> List[float]:
        """Compute cosine similarity of each sentence against the knowledge anchor vectors."""
        model = cls.get_model()
        if not model or cls._anchor_embeddings is None or not sentences:
            # Fallback heuristic score based on factual signals
            return [1.0 if FACTUAL_SIGNAL_PATTERN.search(s) else 0.2 for s in sentences]

        try:
            sent_embs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
            # Max cosine similarity across all anchor topics
            scores = np.max(np.dot(sent_embs, cls._anchor_embeddings.T), axis=1)
            return scores.tolist()
        except Exception as e:
            logger.warning(f"Embedding scoring failed: {e}. Falling back to heuristic.")
            return [1.0 if FACTUAL_SIGNAL_PATTERN.search(s) else 0.2 for s in sentences]

    @classmethod
    def extract_semantic_units_from_page(
        cls,
        page: PageContent,
        doc_filename: str,
        min_semantic_score: float = 0.22,
    ) -> List[Dict[str, Any]]:
        """
        Extract only important words, phrases, and sentences with semantic relevance.
        Discards low-value boilerplate, headers, footers, and page numbers.
        Returns a list of semantic unit dictionaries:
        [{"text": "...", "page_number": int, "section_title": str, "score": float, "type": str}]
        """
        semantic_units = []

        # 1. Process Tables (High semantic density)
        if page.tables:
            for t in page.tables:
                table_lines = t.markdown_repr.split("\n")
                # Keep header and rows with numbers or metrics
                valid_rows = []
                for line in table_lines:
                    line_clean = line.strip()
                    if not line_clean or line_clean.startswith("| ---"):
                        continue
                    if FACTUAL_SIGNAL_PATTERN.search(line_clean):
                        valid_rows.append(line_clean)

                if valid_rows:
                    table_snippet = "Table Data:\n" + "\n".join(valid_rows[:8])
                    semantic_units.append({
                        "text": table_snippet,
                        "page_number": page.page_number,
                        "section_title": t.section_title or f"Table p.{page.page_number}",
                        "score": 0.95,
                        "type": "table",
                    })

        # 2. Process Raw Text
        if not page.raw_text:
            return semantic_units

        # Split text into candidate paragraphs and sentences
        paragraphs = [p.strip() for p in page.raw_text.split("\n\n") if p.strip()]
        candidate_sentences: List[Tuple[str, Optional[str]]] = []

        for p in paragraphs:
            # Check for section title candidate
            lines = p.split("\n")
            section_title = lines[0].strip() if len(lines[0].strip()) < 80 and not lines[0].strip().endswith(".") else None

            # Split paragraph into sentences
            raw_sents = re.split(r"(?<=[.!?])\s+", p.replace("\n", " "))
            for s in raw_sents:
                clean_s = s.strip()
                # Skip too short or too long noisy lines
                if len(clean_s) < 15 or len(clean_s) > 800:
                    continue
                # Skip boilerplate or citations
                if BOILERPLATE_REJECT_PATTERN.search(clean_s):
                    continue
                if NoiseFilter.is_boilerplate(clean_s):
                    continue
                candidate_sentences.append((clean_s, section_title))

        if not candidate_sentences:
            return semantic_units

        texts_only = [s[0] for s in candidate_sentences]
        scores = cls.score_sentences(texts_only)

        for (sent_text, sec_title), score in zip(candidate_sentences, scores):
            # Retain if it has high semantic similarity OR explicit factual/numerical signals
            has_factual_signal = bool(FACTUAL_SIGNAL_PATTERN.search(sent_text))
            if score >= min_semantic_score or has_factual_signal:
                semantic_units.append({
                    "text": sent_text,
                    "page_number": page.page_number,
                    "section_title": sec_title or f"Page {page.page_number}",
                    "score": float(score),
                    "type": "text",
                })

        return semantic_units

"""Noise and boilerplate filtering for document text."""

import re
from typing import List

# Common boilerplate and legal disclaimer markers in Indian corporate and macro filings
BOILERPLATE_PATTERNS = [
    r"safe\s+harbour\s+and\s+disclaimer",
    r"this\s+presentation\s+is\s+prepared\s+by",
    r"forward-looking\s+statements",
    r"all\s+rights\s+reserved",
    r"statutory\s+reports\s+financial\s+statements",
    r"corporate\s+overview\s+statutory\s+reports",
    r"table\s+of\s+contents",
    r"neither\s+the\s+company\s+nor\s+any\s+of\s+its\s+advisors",
    r"no\s+representation\s+or\s+warranty,\s+express\s+or\s+implied",
    r"annual\s+report\s+2023-24\s+corporate\s+overview",
]

COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


class NoiseFilter:
    """Filters out legal disclaimers, safe harbor fluff, citations, and non-informative fragments."""

    @staticmethod
    def is_boilerplate(text: str) -> bool:
        """Check if text matches known boilerplate or disclaimer patterns."""
        clean = text.strip()
        if len(clean) < 8:
            return True
        for pattern in COMPILED_BOILERPLATE:
            if pattern.search(clean):
                return True
        return False

    @classmethod
    def filter_sentences(cls, sentences: List[str]) -> List[str]:
        """Filter a list of sentences, dropping boilerplate and empty items."""
        return [s.strip() for s in sentences if not cls.is_boilerplate(s)]

    @staticmethod
    def strip_citations(text: str) -> str:
        """Strip parenthetical and inline citations (e.g. '(Table 3, Figures 4 and 5)', 'Box 1')."""
        # 1. Parenthetical citations: (Table 3, Figures 4 and 5), (Figure 1), (Box 2), (Appendix II), etc.
        t = re.sub(
            r"\((?:Table|Figure|Fig\.?|Box|Chart|Annex|Appendix|Column|Row|Schedule|Note|Page|p\.)[^)]*\)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        # 2. In-text standalone citations: Table 3, Figure 4, Box 1, Appendix II
        t = re.sub(
            r"\b(?:Table|Figure|Fig\.?|Box|Chart|Annex|Appendix|Column|Row|Schedule|Note|Page|p\.)\s*#?\s*(?:\d+|[I|V|X]+)(?:[–\-]\d+)?\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        return t

    @staticmethod
    def strip_footnotes(text: str) -> str:
        """Strip footnote references like 3/, [1], (1), (1,5)."""
        # Footnote indicators like 3/, 2/, 9/
        t = re.sub(r"\b\d+/\s*", " ", text)
        # Bracketed footnote citations: [1], [1, 2]
        t = re.sub(r"\[\d+(?:,\s*\d+)*\]", " ", t)
        # Parenthetical footnote markers: (1), (1, 2), (1,5)
        t = re.sub(r"\(\d+(?:,\s*\d+)*\)", " ", t)
        return t

    @staticmethod
    def strip_paragraph_numbering(text: str) -> str:
        """Strip paragraph/section prefixes like '7. ', '10. ', '\\n8. ', 'B. '."""
        # Strip paragraph numbers at start of line or after sentence terminators
        t = re.sub(r"(?:^|\n|\.\s+)(?:\d+|[I|V|X]+|[A-Z])[\.\)]\s+(?=[A-Z])", ". ", text)
        return t

    @classmethod
    def clean_text_for_extraction(cls, text: str) -> str:
        """Apply all noise cleaning transforms to text before extracting facts."""
        t = cls.strip_citations(text)
        t = cls.strip_footnotes(t)
        t = cls.strip_paragraph_numbering(t)
        return t

    @staticmethod
    def clean_punctuation(val: str) -> str:
        """Strip trailing and leading punctuation (e.g. '3,' -> '3')."""
        if not val:
            return ""
        return val.strip(" ,.:;()[]/\\")

    @staticmethod
    def is_bare_year(val: str) -> bool:
        """Check if value is a 4-digit calendar year like 2021 or 2024."""
        clean = val.strip(" ,.:;()[]/\\")
        return bool(re.fullmatch(r"(?:19|20)\d{2}", clean))


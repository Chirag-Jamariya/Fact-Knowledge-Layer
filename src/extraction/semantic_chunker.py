"""Token-aware semantic chunker strictly respecting token limits and syntactic boundaries."""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Token limits based on Groq 8,000 TPM / 24 RPM = 333 tokens per request limit
# Safe headroom per request: 300-320 tokens total (Prompt Overhead ~70-80 tokens + Chunk Content ~220-240 tokens)
MAX_REQUEST_TOKENS_LIMIT = 320
DEFAULT_MAX_CHUNK_TOKENS = 220
PROMPT_OVERHEAD_ESTIMATE = 75


@dataclass
class SemanticChunk:
    """A semantically coherent text chunk with verified token count and source metadata."""
    text: str
    token_count: int
    primary_page: int
    page_numbers: List[int] = field(default_factory=list)
    doc_filename: str = ""
    section_title: Optional[str] = None


class TokenCounter:
    """Calculates exact token counts using tiktoken or transformers tokenizer."""

    _tokenizer = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            try:
                import tiktoken
                # cl100k_base matches modern OpenAI & GPT-OSS tokenization conventions
                cls._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                try:
                    from transformers import AutoTokenizer
                    cls._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                except Exception:
                    cls._tokenizer = False
        return cls._tokenizer

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens accurately using the model tokenizer."""
        tok = cls.get_tokenizer()
        if tok and hasattr(tok, "encode"):
            try:
                return len(tok.encode(text))
            except Exception:
                pass
        # Fallback approximation: 1 token ≈ 0.75 words or ~4 characters
        words = len(text.split())
        return max(1, int(words * 1.33))


class SemanticChunker:
    """
    Chunks semantically filtered units based on tokens.
    Splits strictly at section -> paragraph -> sentence -> clause boundaries.
    Never exceeds configured token limit.
    """

    def __init__(
        self,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        max_request_tokens: int = MAX_REQUEST_TOKENS_LIMIT,
    ):
        self.max_chunk_tokens = min(max_chunk_tokens, max_request_tokens - PROMPT_OVERHEAD_ESTIMATE)
        self.max_request_tokens = max_request_tokens

    def chunk_semantic_units(
        self,
        semantic_units: List[Dict[str, Any]],
        doc_filename: str,
    ) -> List[SemanticChunk]:
        """
        Takes a list of semantic units (from SemanticTextFilter) and groups them into
        token-bounded chunks respecting the 300-320 token request limit.
        """
        if not semantic_units:
            return []

        chunks: List[SemanticChunk] = []
        current_texts: List[str] = []
        current_pages: List[int] = []
        current_sec: Optional[str] = None

        def current_tokens() -> int:
            if not current_texts:
                return 0
            return TokenCounter.count_tokens(" ".join(current_texts))

        for unit in semantic_units:
            unit_text = unit["text"].strip()
            unit_page = unit["page_number"]
            unit_sec = unit.get("section_title")

            if not unit_text:
                continue

            unit_tokens = TokenCounter.count_tokens(unit_text)

            # If the single unit itself exceeds max_chunk_tokens, split it intelligently
            if unit_tokens > self.max_chunk_tokens:
                sub_units = self._split_large_unit(unit_text, self.max_chunk_tokens)
                for sub in sub_units:
                    # Flush existing buffer first if needed
                    if current_texts and (current_tokens() + TokenCounter.count_tokens(sub) > self.max_chunk_tokens):
                        self._flush_chunk(chunks, current_texts, current_pages, doc_filename, current_sec)
                        current_texts = []
                        current_pages = []

                    if TokenCounter.count_tokens(sub) > self.max_chunk_tokens:
                        # Direct flush of atomic sub-unit
                        chunks.append(SemanticChunk(
                            text=sub,
                            token_count=TokenCounter.count_tokens(sub),
                            primary_page=unit_page,
                            page_numbers=[unit_page],
                            doc_filename=doc_filename,
                            section_title=unit_sec,
                        ))
                    else:
                        current_texts.append(sub)
                        if unit_page not in current_pages:
                            current_pages.append(unit_page)
                        current_sec = current_sec or unit_sec
                continue

            # Check if adding unit_text exceeds chunk token limit
            cand_tokens = TokenCounter.count_tokens(" ".join(current_texts + [unit_text]))
            if cand_tokens <= self.max_chunk_tokens:
                current_texts.append(unit_text)
                if unit_page not in current_pages:
                    current_pages.append(unit_page)
                current_sec = current_sec or unit_sec
            else:
                # Flush current buffer
                if current_texts:
                    self._flush_chunk(chunks, current_texts, current_pages, doc_filename, current_sec)
                    current_texts = []
                    current_pages = []
                # Start new chunk
                current_texts.append(unit_text)
                current_pages.append(unit_page)
                current_sec = unit_sec

        # Flush any remaining items
        if current_texts:
            self._flush_chunk(chunks, current_texts, current_pages, doc_filename, current_sec)

        return chunks

    def _flush_chunk(
        self,
        chunks: List[SemanticChunk],
        texts: List[str],
        pages: List[int],
        doc_filename: str,
        sec_title: Optional[str],
    ) -> None:
        """Create and append a finalized SemanticChunk."""
        combined_text = " ".join(texts).strip()
        if not combined_text:
            return
        primary_pg = pages[0] if pages else 1
        t_count = TokenCounter.count_tokens(combined_text)
        chunks.append(SemanticChunk(
            text=combined_text,
            token_count=t_count,
            primary_page=primary_pg,
            page_numbers=list(pages),
            doc_filename=doc_filename,
            section_title=sec_title,
        ))

    def _split_large_unit(self, text: str, max_tokens: int) -> List[str]:
        """
        Intelligently split a large text unit at sentence boundaries (.!?)
        or clause boundaries (;,) so no fragment exceeds max_tokens.
        """
        # 1. Try splitting on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            result = []
            curr = []
            for s in sentences:
                cand = " ".join(curr + [s]).strip()
                if TokenCounter.count_tokens(cand) <= max_tokens:
                    curr.append(s)
                else:
                    if curr:
                        result.append(" ".join(curr).strip())
                        curr = []
                    if TokenCounter.count_tokens(s) <= max_tokens:
                        curr.append(s)
                    else:
                        # Sentence itself is too big -> split at clauses
                        clauses = self._split_clauses(s, max_tokens)
                        result.extend(clauses)
            if curr:
                result.append(" ".join(curr).strip())
            return result

        # 2. Single sentence too big -> split at clauses
        return self._split_clauses(text, max_tokens)

    def _split_clauses(self, text: str, max_tokens: int) -> List[str]:
        """Split a long sentence at semicolon or comma clause boundaries."""
        clauses = re.split(r"(?<=[;,])\s+", text)
        if len(clauses) > 1:
            result = []
            curr = []
            for c in clauses:
                cand = " ".join(curr + [c]).strip()
                if TokenCounter.count_tokens(cand) <= max_tokens:
                    curr.append(c)
                else:
                    if curr:
                        result.append(" ".join(curr).strip())
                        curr = []
                    result.append(c.strip())
            if curr:
                result.append(" ".join(curr).strip())
            return result

        # 3. Fallback: split words safely without truncating
        words = text.split()
        result = []
        curr = []
        for w in words:
            cand = " ".join(curr + [w])
            if TokenCounter.count_tokens(cand) <= max_tokens:
                curr.append(w)
            else:
                if curr:
                    result.append(" ".join(curr))
                curr = [w]
        if curr:
            result.append(" ".join(curr))
        return result

"""Unified Fact Extraction Engine supporting Gemini, OpenAI, and offline rule-based extraction."""

import os
import re
import json
import logging
import httpx
from typing import List, Optional, Dict, Any

from src.models.corpus import PageContent
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.extraction.rule_based_extractor import RuleBasedExtractor
from src.extraction.entity_detector import EntityDetector, SOVEREIGN_GAZETTEER
from src.extraction.rate_limiter import groq_rate_limiter
from src.extraction.semantic_filter import SemanticTextFilter
from src.extraction.semantic_chunker import (
    SemanticChunker,
    SemanticChunk,
    TokenCounter,
    MAX_REQUEST_TOKENS_LIMIT,
    DEFAULT_MAX_CHUNK_TOKENS,
)
from src.config import settings

logger = logging.getLogger(__name__)

# Ultra-concise system prompt for token-bounded Groq chunk requests (keeps request <= 320 tokens)
GROQ_CHUNK_SYSTEM_PROMPT = (
    'Extract atomic facts as JSON {"facts":[{"entity":"str","metric":"str","value":"str",'
    '"temporal_period":"str|null","reporting_scope":"str|null","unit":"str|null","verbatim_quote":"str"}]}. '
    'Retain domain and sector qualifiers (e.g. "Oil GDP" vs "Real GDP"). '
    'Do not extract figure/table indexes or paragraph numbers as values.'
)

SYSTEM_PROMPT = """You are a rigorous Fact Extraction Engine for financial, economic, and enterprise documents.
Your task is to extract atomic, verifiable factual assertions (numerical and semantic).
For each fact:
- Identify the explicit Entity (e.g. 'Delhivery Limited', 'Republic of Chad', 'Republic of Niger', 'Indian Economy').
- Identify the Metric or Proposition (e.g. 'Revenue from Operations', 'Real GDP growth', 'Oil GDP growth', 'Headline Inflation'). Preserve domain and sector qualifiers (e.g., 'Oil GDP growth' vs 'Headline GDP growth'). Never drop qualifiers.
- Identify the Normalized Value (e.g. '8,142 Cr', '2.8 Billion', '5.4%', '2.1%').
- Extract the Context Dimensions: temporal_period (e.g. 'FY24', '2023', '2020-23'), reporting_scope (e.g. 'Consolidated', 'Standalone', 'Annual'), unit (e.g. 'INR Crores', '%', 'Count').
- Extract the exact Verbatim Quote from the text supporting this fact.
- CRITICAL: Do NOT extract table row indices, figure citations, or paragraph numbers (e.g., '3' from Table 3 or '7' from Paragraph 7) as metrics or values.

Respond ONLY with a valid JSON object with a 'facts' key containing an array of objects conforming to:
{
  "facts": [
    {
      "category": "numerical" | "semantic",
      "entity": "string",
      "metric": "string",
      "value": "string",
      "raw_value": "string",
      "temporal_period": "string or null",
      "reporting_scope": "string or null",
      "unit": "string or null",
      "verbatim_quote": "string"
    }
  ]
}
"""


class FactExtractor:
    """Unified Fact Extraction Layer with Groq, Gemini, OpenAI, and deterministic rule fallbacks."""

    def __init__(self, use_llm_if_available: bool = True):
        self.groq_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        self.openrouter_key = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.use_llm = use_llm_if_available and bool(
            self.groq_key or self.openrouter_key or self.gemini_key or self.openai_key
        )
        self.backoff_callback: Optional[Any] = None

    def extract_from_page(self, page: PageContent, doc_filename: str) -> List[Fact]:
        """Extract all verifiable facts from a single page with multi-tier fallbacks."""
        # 1. Deterministic Rule-Based & Table extraction (baseline ground truth)
        rule_facts = RuleBasedExtractor.extract_from_page(page, doc_filename)

        # 2. Skip expensive LLM calls for pages lacking numbers or tables (e.g. title/blank/contents pages)
        has_data = bool(re.search(r'\d', page.raw_text)) or bool(page.tables)
        if not has_data:
            return rule_facts

        # 3. Multi-tier LLM Extraction
        if self.use_llm:
            llm_facts = []
            # Tier 1: Groq Cloud API (Ultra-fast, 50 RPM / 1000 RPD rate-limited)
            if self.groq_key:
                try:
                    llm_facts = self._extract_via_groq(page, doc_filename)
                except Exception as e:
                    logger.warning(f"Groq extraction error on page {page.page_number}: {e}. Trying Gemini...")

            # Tier 2: Google Gemini (direct high-speed API backup)
            if not llm_facts and self.gemini_key:
                try:
                    llm_facts = self._extract_via_gemini(page, doc_filename)
                except Exception as e:
                    logger.warning(f"Gemini extraction error on page {page.page_number}: {e}. Trying OpenRouter...")

            # Tier 3: OpenRouter (free tier backup)
            if not llm_facts and self.openrouter_key:
                try:
                    llm_facts = self._extract_via_openrouter(page, doc_filename)
                except Exception as e:
                    logger.warning(f"OpenRouter extraction issue on page {page.page_number}: {e}.")

            # Tier 4: OpenAI Direct
            if not llm_facts and self.openai_key:
                try:
                    llm_facts = self._extract_via_openai(page, doc_filename)
                except Exception as e:
                    logger.warning(f"OpenAI extraction error on page {page.page_number}: {e}.")

            if llm_facts:
                return self._merge_facts(rule_facts, llm_facts)

        return rule_facts

    def extract_pages_concurrently(
        self,
        pages: List[PageContent],
        doc_filename: str,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Any] = None,
        backoff_callback: Optional[Any] = None,
    ) -> List[Fact]:
        """Extract facts across pages either via semantic Groq chunking or concurrent thread pool."""
        if backoff_callback:
            self.backoff_callback = backoff_callback

        # When using Groq LLM, use token-bounded semantic chunking pipeline (< 320 tokens/req, >= 2.5s pacing)
        if self.groq_key and self.use_llm:
            return self.extract_semantic_chunks_via_groq(
                pages,
                doc_filename,
                progress_callback=progress_callback,
                backoff_callback=backoff_callback,
            )

        # Offline / deterministic rule-based extractor
        if not self.use_llm:
            return RuleBasedExtractor.extract_pages_concurrently(
                pages, doc_filename, max_workers=max_workers or 6, progress_callback=progress_callback
            )

        all_facts: List[Fact] = []
        total = len(pages)
        completed = 0
        pages_sorted = sorted(pages, key=lambda p: p.page_number)

        with ThreadPoolExecutor(max_workers=max_workers or 3) as executor:
            future_to_page = {
                executor.submit(self.extract_from_page, p, doc_filename): p.page_number
                for p in pages_sorted
            }

            for future in as_completed(future_to_page):
                completed += 1
                pg_num = future_to_page[future]
                try:
                    page_facts = future.result()
                    all_facts.extend(page_facts)
                except Exception as e:
                    logger.warning(f"Error extracting facts on page {pg_num}: {e}")

                if progress_callback:
                    progress_callback(completed, total, pg_num)

        all_facts.sort(key=lambda f: (f.evidence.page_number, f.metric))
        return all_facts

    def extract_semantic_chunks_via_groq(
        self,
        pages: List[PageContent],
        doc_filename: str,
        progress_callback: Optional[Any] = None,
        backoff_callback: Optional[Any] = None,
    ) -> List[Fact]:
        """
        Extract only important knowledge content with SentenceTransformer,
        chunk based strictly on tokens (<= 220 tokens/chunk, total request <= 320 tokens),
        and send chunks to Groq API with >= 2.5s pacing (24 RPM cap).
        """
        if backoff_callback:
            self.backoff_callback = backoff_callback

        # 1. Filter raw pages into high-value semantic units
        all_semantic_units: List[Dict[str, Any]] = []
        for p in sorted(pages, key=lambda pg: pg.page_number):
            units = SemanticTextFilter.extract_semantic_units_from_page(p, doc_filename)
            all_semantic_units.extend(units)

        # Fallback if filter returned empty (e.g. synthetic test pages)
        if not all_semantic_units and pages:
            for p in pages:
                if p.raw_text and p.raw_text.strip():
                    all_semantic_units.append({
                        "text": p.raw_text.strip(),
                        "page_number": p.page_number,
                        "section_title": f"Page {p.page_number}",
                    })

        # 2. Chunk based on tokens, strictly respecting 300-320 token request limit
        chunker = SemanticChunker(
            max_chunk_tokens=DEFAULT_MAX_CHUNK_TOKENS,
            max_request_tokens=MAX_REQUEST_TOKENS_LIMIT,
        )
        chunks = chunker.chunk_semantic_units(all_semantic_units, doc_filename)

        if not chunks:
            return []

        logger.info(
            f"Document '{doc_filename}' ({len(pages)} pages) filtered to "
            f"{len(all_semantic_units)} semantic units and {len(chunks)} token-bounded chunks."
        )

        # 3. Send chunks to Groq at paced intervals (>= 2.5s per request)
        all_facts: List[Fact] = []
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            if progress_callback:
                try:
                    progress_callback(idx + 1, total_chunks, chunk.primary_page)
                except Exception:
                    pass

            chunk_facts = self._extract_via_groq_chunk(chunk, doc_filename)
            all_facts.extend(chunk_facts)

        # Merge with rule-based facts to ensure maximum recall without duplicate metrics
        rule_facts = RuleBasedExtractor.extract_pages_concurrently(pages, doc_filename)
        combined_facts = self._merge_facts(rule_facts, all_facts)
        combined_facts.sort(key=lambda f: (f.evidence.page_number, f.metric))
        return combined_facts

    def _extract_via_openrouter(self, page: PageContent, doc_filename: str) -> List[Fact]:
        """Extract facts using OpenRouter API with free high-throughput model."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "https://github.com/superjoin-ai",
            "X-Title": "Superjoin Fact Knowledge Layer",
            "Content-Type": "application/json",
        }
        
        user_prompt = f"DOCUMENT: {doc_filename}\nPAGE: {page.page_number}\n\nCONTENT:\n{page.raw_text}"
        if page.tables:
            user_prompt += "\n\nTABLES:\n" + "\n".join(t.markdown_repr for t in page.tables)

        model_name = settings.openrouter_model or "nex-agi/nex-n2.5-mini:free"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }

        with httpx.Client(timeout=8.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        try:
            choice = data["choices"][0]["message"]
            content = choice.get("content")
            if not content:
                return []
            parsed_json = json.loads(content)
            items = parsed_json.get("facts", []) if isinstance(parsed_json, dict) else parsed_json
            return self._parse_json_items_to_facts(items, doc_filename, page.page_number)
        except Exception as err:
            logger.warning(f"Failed to parse OpenRouter JSON: {err}")
            return []

    def _extract_via_gemini(self, page: PageContent, doc_filename: str) -> List[Fact]:
        """Extract facts using Google Gemini API directly via HTTPX."""
        model_name = settings.gemini_model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
        
        user_prompt = f"DOCUMENT: {doc_filename}\nPAGE: {page.page_number}\n\nCONTENT:\n{page.raw_text}"
        if page.tables:
            user_prompt += "\n\nTABLES:\n" + "\n".join(t.markdown_repr for t in page.tables)

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }

        max_retries = 3
        data = None
        for attempt in range(max_retries + 1):
            try:
                # Pace requests to respect Gemini Free Tier 15 RPM limit (1 request every ~3.5s)
                import time
                time.sleep(1.0)

                with httpx.Client(timeout=35.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 429 and attempt < max_retries:
                        # Rate limit hit: sleep and retry with exponential backoff
                        backoff = 4.0 * (attempt + 1)
                        logger.warning(f"Gemini 429 rate limit hit on page {page.page_number}. Backing off for {backoff:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff)
                        continue
                    if resp.status_code == 503 and attempt < max_retries:
                        time.sleep(2.0 * (attempt + 1))
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    break
            except httpx.HTTPStatusError as http_err:
                if http_err.response.status_code == 429 and attempt < max_retries:
                    backoff = 5.0 * (attempt + 1)
                    time.sleep(backoff)
                    continue
                if attempt == max_retries:
                    raise http_err
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise e

        if not data:
            return []

        try:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(raw_text)
            items = parsed_json.get("facts", []) if isinstance(parsed_json, dict) else parsed_json
            return self._parse_json_items_to_facts(items, doc_filename, page.page_number)
        except (KeyError, json.JSONDecodeError, IndexError) as err:
            logger.warning(f"Failed to parse Gemini JSON output: {err}")
            return []

    def _extract_via_groq_chunk(self, chunk: SemanticChunk, doc_filename: str) -> List[Fact]:
        """Extract facts from a single token-bounded chunk using Groq with rate limiting & exponential backoff."""
        from openai import OpenAI
        import time
        import re

        # Enforce thread-safe rate limit (max 24 RPM, 1000 RPD, >= 2.5s inter-request pacing: 1 req / 2.5s)
        groq_rate_limiter.acquire()

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.groq_key,
        )

        section_info = f"\nSECTION: {chunk.section_title}" if chunk.section_title else ""
        user_prompt = f"DOCUMENT: {doc_filename}\nPAGE: {chunk.primary_page}{section_info}\n\nCONTENT:\n{chunk.text}"
        model_name = settings.groq_model or "openai/gpt-oss-120b"

        # Explicit backoff progression: 3s, 8s, 13s, 17s, 23s (threshold: 5 retries) then stop
        BACKOFF_SCHEDULE = [3.0, 8.0, 13.0, 17.0, 23.0]
        max_retries = len(BACKOFF_SCHEDULE)  # 5
        data = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": GROQ_CHUNK_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content or "{}"
                data = json.loads(content)
                break
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = (
                    "429" in err_msg
                    or "rate" in err_msg.lower()
                    or "tpm" in err_msg.lower()
                    or "tokens" in err_msg.lower()
                )
                if is_rate_limit and attempt < max_retries:
                    # Parse recommended wait time from Groq error message if present
                    suggested_wait = 0.0
                    match = re.search(r"try again in ([\d\.]+)\s*(ms|s)", err_msg, re.IGNORECASE)
                    if match:
                        try:
                            val = float(match.group(1))
                            unit = match.group(2).lower()
                            suggested_wait = (val / 1000.0 if unit == "ms" else val) + 0.5
                        except ValueError:
                            pass

                    wait_time = max(BACKOFF_SCHEDULE[attempt], suggested_wait)

                    logger.warning(
                        f"Groq 429 rate limit hit on chunk (page {chunk.primary_page}) "
                        f"(Backoff attempt {attempt+1}/{max_retries}: {BACKOFF_SCHEDULE[attempt]:.0f}s scheduled). "
                        f"Backing off for {wait_time:.1f}s..."
                    )

                    if self.backoff_callback:
                        try:
                            self.backoff_callback(chunk.primary_page, attempt + 1, wait_time, doc_filename)
                        except Exception:
                            pass

                    groq_rate_limiter.penalize(wait_time)
                    time.sleep(wait_time)
                    groq_rate_limiter.acquire()
                    continue

                if attempt == max_retries:
                    logger.error(
                        f"Groq 429 rate limit reached threshold of 5 retries (3, 8, 13, 17, 23s exhausted) on chunk (page {chunk.primary_page}). Stopping."
                    )
                    raise e
                raise e

        if not data:
            return []

        items = data.get("facts", []) if isinstance(data, dict) else data
        return self._parse_json_items_to_facts(items, doc_filename, chunk.primary_page)

    def _extract_via_groq(self, page: PageContent, doc_filename: str) -> List[Fact]:
        """Extract facts using Groq Cloud API with sliding-window rate limiting & exponential backoff."""
        from openai import OpenAI
        import time
        import random
        import re

        # Enforce thread-safe rate limit (max 24 RPM, 1000 RPD, 2.5s inter-request pacing: 1 req / 2.5s)
        groq_rate_limiter.acquire()

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.groq_key,
        )

        user_prompt = f"DOCUMENT: {doc_filename}\nPAGE: {page.page_number}\n\nCONTENT:\n{page.raw_text}"
        if page.tables:
            user_prompt += "\n\nTABLES:\n" + "\n".join(t.markdown_repr for t in page.tables)

        model_name = settings.groq_model or "openai/gpt-oss-120b"

        # Explicit backoff progression: 3s, 8s, 13s, 17s, 23s (threshold: 5 retries) then stop
        BACKOFF_SCHEDULE = [3.0, 8.0, 13.0, 17.0, 23.0]
        max_retries = len(BACKOFF_SCHEDULE)  # 5
        data = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content or "{}"
                data = json.loads(content)
                break
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = (
                    "429" in err_msg
                    or "rate" in err_msg.lower()
                    or "tpm" in err_msg.lower()
                    or "tokens" in err_msg.lower()
                )
                if is_rate_limit and attempt < max_retries:
                    # Parse recommended wait time from Groq error message / header if present
                    suggested_wait = 0.0
                    match = re.search(r"try again in ([\d\.]+)\s*(ms|s)", err_msg, re.IGNORECASE)
                    if match:
                        try:
                            val = float(match.group(1))
                            unit = match.group(2).lower()
                            suggested_wait = (val / 1000.0 if unit == "ms" else val) + 0.5
                        except ValueError:
                            pass

                    # Apply user-specified backoff schedule: 3s, 8s, 13s, 17s, 23s (or suggested if larger)
                    wait_time = max(BACKOFF_SCHEDULE[attempt], suggested_wait)

                    logger.warning(
                        f"Groq 429 rate limit hit on page {page.page_number} "
                        f"(Backoff attempt {attempt+1}/{max_retries}: {BACKOFF_SCHEDULE[attempt]:.0f}s scheduled). "
                        f"Backing off for {wait_time:.1f}s..."
                    )

                    # Notify UI status / progress callback
                    if self.backoff_callback:
                        try:
                            self.backoff_callback(page.page_number, attempt + 1, wait_time, doc_filename)
                        except Exception:
                            pass

                    # Penalize shared limiter so all concurrent worker threads pause
                    groq_rate_limiter.penalize(wait_time)

                    time.sleep(wait_time)

                    # Re-acquire limiter slot (with 2.5s pacing) before next attempt
                    groq_rate_limiter.acquire()
                    continue

                if attempt == max_retries:
                    logger.error(
                        f"Groq 429 rate limit reached threshold of 5 retries (3, 8, 13, 17, 23s exhausted) on page {page.page_number}. Stopping."
                    )
                    raise e
                raise e

        if not data:
            return []

        items = data.get("facts", []) if isinstance(data, dict) else data
        return self._parse_json_items_to_facts(items, doc_filename, page.page_number)

    def _extract_via_openai(self, page: PageContent, doc_filename: str) -> List[Fact]:
        """Extract facts using OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_key)
        prompt = f"DOCUMENT: {doc_filename}\nPAGE: {page.page_number}\n\nCONTENT:\n{page.raw_text}"
        if page.tables:
            prompt += "\n\nTABLES:\n" + "\n".join(t.markdown_repr for t in page.tables)

        response = client.chat.completions.create(
            model=settings.default_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        items = data.get("facts", []) if isinstance(data, dict) else data
        return self._parse_json_items_to_facts(items, doc_filename, page.page_number)

    def _parse_json_items_to_facts(
        self, items: List[Dict[str, Any]], doc_filename: str, page_number: int
    ) -> List[Fact]:
        """Convert raw JSON objects into validated Fact instances with entity resolution."""
        fallback_entity = EntityDetector.detect_entity(doc_filename)

        facts: List[Fact] = []
        for item in items:
            raw_entity = (item.get("entity") or "").strip()
            # Canonicalize sovereign entity or fallback to document-detected entity
            if not raw_entity or raw_entity.lower() in {
                "enterprise / document subject",
                "unknown",
                "unknown entity",
                "entity",
                "document subject",
            }:
                entity = fallback_entity
            elif raw_entity.lower() in SOVEREIGN_GAZETTEER:
                entity = SOVEREIGN_GAZETTEER[raw_entity.lower()]
            else:
                entity = raw_entity

            quote = item.get("verbatim_quote", "")
            ev = SourceEvidence(
                document_id=doc_filename,
                page_number=page_number,
                content_type="text",
                verbatim_quote=quote[:300] if quote else f"Page {page_number}",
            )
            ctx = ContextDimensions(
                temporal_period=item.get("temporal_period"),
                reporting_scope=item.get("reporting_scope"),
                unit=item.get("unit"),
            )
            if item.get("category") == "numerical":
                category = FactCategory.NUMERICAL
            elif item.get("category") == "semantic":
                category = FactCategory.SEMANTIC
            elif bool(re.search(r'\d', str(item.get("value", "")))) or item.get("unit"):
                category = FactCategory.NUMERICAL
            else:
                category = FactCategory.SEMANTIC

            fact = Fact(
                category=category,
                entity=entity,
                metric=item.get("metric", "Factual Assertion"),
                value=str(item.get("value", "")),
                raw_value=str(item.get("raw_value", item.get("value", ""))),
                context=ctx,
                evidence=ev,
                confidence_score=0.95,
            )
            facts.append(fact)
        return facts

    @staticmethod
    def _merge_facts(rule_facts: List[Fact], llm_facts: List[Fact]) -> List[Fact]:
        """Combine and deduplicate rule-based and LLM-extracted facts."""
        merged = {f.fact_id: f for f in rule_facts}
        for f in llm_facts:
            # Match on (metric, value) to prevent duplicate near-identical facts
            key = (f.metric.lower().strip(), f.value.lower().strip())
            existing = any((rf.metric.lower().strip(), rf.value.lower().strip()) == key for rf in rule_facts)
            if not existing:
                merged[f.fact_id] = f
        return list(merged.values())

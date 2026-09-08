"""Unit tests for Fact Extraction Layer (Component 3)."""

import pytest
import json
from src.extraction.noise_filter import NoiseFilter
from src.extraction.rule_based_extractor import RuleBasedExtractor
from src.extraction.fact_extractor import FactExtractor
from src.models.corpus import PageContent, ExtractedTable
from src.models.fact import Fact, FactCategory


def test_noise_filter_boilerplate_detection():
    """Verify that NoiseFilter catches disclaimers and allows factual content."""
    disclaimer_1 = "Safe harbour and disclaimer: This Presentation is prepared by Delhivery Limited."
    disclaimer_2 = "Corporate Overview Statutory Reports Financial Statements"
    factual_sentence = "Revenue from operations increased by 13% to ₹8,142 Cr in FY24."

    assert NoiseFilter.is_boilerplate(disclaimer_1) is True
    assert NoiseFilter.is_boilerplate(disclaimer_2) is True
    assert NoiseFilter.is_boilerplate(factual_sentence) is False


def test_rule_based_extraction_from_text():
    """Verify extracting numerical facts with context from raw text."""
    sample_text = """
    Delhivery In Numbers
    >2.8Bn Express parcel shipments delivered since inception
    18,793 Pin codes covered
    Workforce strength was 98,135 employees in FY24.
    Revenue from operations reached ₹8,142 Cr on a consolidated basis.
    """
    page = PageContent(
        doc_id="doc_test_1",
        page_number=2,
        raw_text=sample_text,
        tables=[],
        images=[],
    )

    facts = RuleBasedExtractor.extract_from_page(page, "02-delhivery-annual-report-fy24.pdf")
    assert len(facts) >= 3

    metrics_found = {f.metric: f for f in facts}
    assert "Express Parcel Shipments" in metrics_found
    assert ">2.8Bn" in metrics_found["Express Parcel Shipments"].value or "2.8" in metrics_found["Express Parcel Shipments"].value

    assert "Pin Codes Covered" in metrics_found
    assert "18,793" in metrics_found["Pin Codes Covered"].value

    assert "Revenue from Operations" in metrics_found
    rev_fact = metrics_found["Revenue from Operations"]
    assert "8,142" in rev_fact.value
    assert rev_fact.context.reporting_scope == "Consolidated"
    assert rev_fact.evidence.page_number == 2
    assert rev_fact.category == FactCategory.NUMERICAL


def test_rule_based_extraction_from_table():
    """Verify extracting facts from an ExtractedTable object."""
    table = ExtractedTable(
        page_number=10,
        headers=["Financial Summary", "FY23", "FY24"],
        rows=[
            ["Revenue from operations (INR Cr)", "7,225", "8,142"],
            ["Adjusted EBITDA (INR Cr)", "-68", "127"],
        ],
        markdown_repr="| Financial Summary | FY23 | FY24 |\n| --- | --- | --- |\n| Revenue from operations | 7,225 | 8,142 |",
    )
    page = PageContent(
        doc_id="doc_test_2",
        page_number=10,
        raw_text="Financial performance overview",
        tables=[table],
        images=[],
    )

    facts = RuleBasedExtractor.extract_from_page(page, "delhivery_earnings.pdf")
    assert len(facts) >= 2
    rev_facts = [f for f in facts if f.metric == "Revenue from Operations"]
    assert len(rev_facts) >= 1
    assert any("8,142" in f.value for f in rev_facts)


def test_unified_fact_extractor():
    """Verify unified FactExtractor returns valid Fact objects."""
    extractor = FactExtractor(use_llm_if_available=False)
    page = PageContent(
        doc_id="test_doc",
        page_number=5,
        raw_text="Headline CPI inflation for the Indian Economy moderated to 5.4% in 2024.",
        tables=[],
        images=[],
    )
    facts = extractor.extract_from_page(page, "01-india-economic-survey-2024-25.pdf")
    assert len(facts) >= 1
    assert facts[0].entity == "Indian Economy"
    assert "5.4%" in facts[0].value


def test_openrouter_fact_parsing():
    """Verify parsing OpenRouter JSON response into Fact objects."""
    extractor = FactExtractor(use_llm_if_available=True)
    raw_items = [
        {
            "category": "numerical",
            "entity": "Delhivery Limited",
            "metric": "Revenue from Operations",
            "value": "8,142 Cr",
            "raw_value": "8,142 Cr",
            "temporal_period": "FY24",
            "reporting_scope": "Consolidated",
            "unit": "INR Crores",
            "verbatim_quote": "Revenue from operations reached 8,142 Cr in FY24."
        }
    ]
    parsed = extractor._parse_json_items_to_facts(raw_items, "doc_test.pdf", page_number=3)
    assert len(parsed) == 1
    fact = parsed[0]
    assert fact.entity == "Delhivery Limited"
    assert fact.metric == "Revenue from Operations"
    assert fact.value == "8,142 Cr"
    assert fact.context.temporal_period == "FY24"
    assert fact.context.reporting_scope == "Consolidated"
    assert fact.evidence.page_number == 3
    assert fact.evidence.verbatim_quote == "Revenue from operations reached 8,142 Cr in FY24."


def test_dynamic_sovereign_entity_detection():
    """Verify that sovereign entities like Chad and Niger are accurately identified without Indian bias."""
    from src.extraction.entity_detector import EntityDetector

    # Sovereign checks from filename
    assert EntityDetector.detect_entity("chad.pdf") == "Republic of Chad"
    assert EntityDetector.detect_entity("niger.pdf") == "Republic of Niger"
    assert EntityDetector.detect_entity("01-delhivery-annual-report.pdf") == "Delhivery Limited"
    assert EntityDetector.detect_entity("1indea2022001.pdf") == "Indian Economy"

    # IMF report text check
    imf_chad_text = "IMF Country Report No. 24/335 \nCHAD \n2024 ARTICLE IV CONSULTATION"
    assert EntityDetector.detect_entity("random_report.pdf", imf_chad_text) == "Republic of Chad"

    imf_niger_text = "IMF Country Report No. 25/25 \nNIGER \n2024 ARTICLE IV CONSULTATION"
    assert EntityDetector.detect_entity("random_report.pdf", imf_niger_text) == "Republic of Niger"


def test_qualified_metric_sector_modifiers_preservation():
    """Verify that 'oil GDP' and 'compound average GDP' preserve qualifiers and are not truncated."""
    page_text = """
    A drop in oil prices and production saw oil GDP growth average only 1.7 percent from 2020–23.
    Average non-oil growth also underperformed expectations at 1.4 percent.
    Memo: Compound average GDP growth rate 16.5%
    In contrast, real GDP growth reached 8.8 percent in 2024.
    """
    page = PageContent(
        doc_id="test_qualifiers",
        page_number=9,
        raw_text=page_text,
        tables=[],
        images=[],
    )
    facts = RuleBasedExtractor.extract_from_page(page, "chad.pdf", doc_entity="Republic of Chad")
    metrics_by_name = {f.metric: f for f in facts}

    # Oil GDP Growth must be isolated with qualifier
    assert "Oil GDP Growth" in metrics_by_name
    assert "1.7%" in metrics_by_name["Oil GDP Growth"].value
    assert metrics_by_name["Oil GDP Growth"].entity == "Republic of Chad"

    # Non-Oil GDP Growth must be isolated
    assert "Non-Oil GDP Growth" in metrics_by_name
    assert "1.4%" in metrics_by_name["Non-Oil GDP Growth"].value

    # Compound Average GDP Growth must be isolated
    assert "Compound Average GDP Growth" in metrics_by_name
    assert "16.5%" in metrics_by_name["Compound Average GDP Growth"].value

    # Headline Real GDP Growth must only capture true headline GDP
    assert "Real GDP Growth" in metrics_by_name
    assert "8.8%" in metrics_by_name["Real GDP Growth"].value


def test_cross_sovereign_isolation_and_failure_detection():
    """Verify that sovereign accounts from distinct nations cannot be paired as contradictions/shifts."""
    from src.indexing.clusterer import CrossDocClusterer
    from src.reconciliation.failure_detector import FailureDetector
    from src.reconciliation.reconciler import CrossDocReconciler
    from src.models.reconciliation import ReconciliationVerdict

    fact_chad = Fact(
        entity="Republic of Chad",
        metric="Real GDP Growth",
        value="1.7%",
        raw_value="1.7",
        context={"temporal_period": "2020-23", "unit": "%"},
        evidence={"document_id": "chad.pdf", "page_number": 9, "verbatim_quote": "growth average only 1.7 percent"},
    )
    fact_niger = Fact(
        entity="Republic of Niger",
        metric="Real GDP Growth",
        value="8.8%",
        raw_value="8.8",
        context={"temporal_period": "2024", "unit": "%"},
        evidence={"document_id": "niger.pdf", "page_number": 3, "verbatim_quote": "GDP growth estimated at 8.8 percent in 2024"},
    )

    # Clusterer must not pair Chad's GDP with Niger's GDP
    pairs = CrossDocClusterer.find_cross_document_pairs([fact_chad, fact_niger], relax_entity=False)
    assert len(pairs) == 0

    # Failure detector must flag any attempted cross-sovereign comparison
    is_fail, fail_reason, mitigation = FailureDetector.check_failure(fact_chad, fact_niger)
    assert is_fail is True
    assert "cross-sovereign comparison error" in fail_reason.lower()
    assert "mitigation" in mitigation.lower()

    # Reconciler classifies attempted pair as EXTRACTION_FAILURE
    reconciler = CrossDocReconciler()
    recon = reconciler.reconcile_pair(fact_chad, fact_niger)
    assert recon.verdict == ReconciliationVerdict.EXTRACTION_FAILURE


def test_rate_limiter_pacing_and_telemetry():
    """Verify RateLimiter pacing, sliding window, and dynamic reconfiguration."""
    import time
    from src.extraction.rate_limiter import RateLimiter

    limiter = RateLimiter(max_rpm=60.0, max_rpd=1000, min_interval_seconds=0.05)
    t0 = time.time()
    limiter.acquire()
    limiter.acquire()
    elapsed = time.time() - t0
    # Must have waited at least min_interval_seconds
    assert elapsed >= 0.04

    stats = limiter.get_stats()
    assert stats["rpm_used"] == 2
    assert stats["rpd_used"] == 2
    assert stats["min_interval_seconds"] == 0.05

    # Reconfigure
    limiter.configure(max_rpm=1.5, min_interval_seconds=40.0)
    assert limiter.max_rpm == 1.5
    assert limiter.min_interval == 40.0


def test_groq_extraction_parsing_and_mock():
    """Verify Groq extraction workflow with mocked OpenAI client and entity resolution."""
    from unittest.mock import MagicMock, patch
    from src.extraction.fact_extractor import FactExtractor
    from src.models.corpus import PageContent

    extractor = FactExtractor(use_llm_if_available=True)
    page = PageContent(
        doc_id="chad.pdf",
        page_number=9,
        raw_text="In Chad, oil GDP growth averaged 1.7 percent from 2020-23.",
        tables=[],
        images=[],
    )

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "facts": [
            {
                "category": "numerical",
                "entity": "Chad",
                "metric": "Oil GDP Growth",
                "value": "1.7%",
                "raw_value": "1.7 percent",
                "temporal_period": "2020-23",
                "reporting_scope": "annual average",
                "unit": "%",
                "verbatim_quote": "oil GDP growth averaged 1.7 percent from 2020-23"
            }
        ]
    })
    mock_chat_completion.choices = [mock_choice]

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.return_value = mock_chat_completion
        mock_openai_cls.return_value = mock_client_inst

        facts = extractor._extract_via_groq(page, "chad.pdf")
        assert len(facts) == 1
        assert facts[0].entity == "Republic of Chad"
        assert facts[0].metric == "Oil GDP Growth"
        assert facts[0].value == "1.7%"
        assert facts[0].context.temporal_period == "2020-23"
        assert facts[0].evidence.document_id == "chad.pdf"


def test_groq_exponential_backoff_threshold_5():
    """Verify Groq 429 exponential backoff with a threshold of 5 retries."""
    from unittest.mock import MagicMock, patch
    from src.extraction.fact_extractor import FactExtractor
    from src.models.corpus import PageContent

    extractor = FactExtractor(use_llm_if_available=True)
    page = PageContent(
        doc_id="test.pdf",
        page_number=1,
        raw_text="Test content",
        tables=[],
        images=[],
    )

    backoff_log = []
    extractor.backoff_callback = lambda pg, attempt, wait_sec, fn: backoff_log.append((pg, attempt, wait_sec))

    # Mock OpenAI client that fails with 429 for 2 attempts then succeeds
    mock_success = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"facts": []})
    mock_success.choices = [mock_choice]

    side_effects = [
        Exception("Error code: 429 - Rate limit reached: Please try again in 1.5s"),
        Exception("Error code: 429 - Rate limit reached"),
        mock_success,
    ]

    with patch("openai.OpenAI") as mock_openai_cls, patch("time.sleep") as mock_sleep:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = side_effects
        mock_openai_cls.return_value = mock_client

        facts = extractor._extract_via_groq(page, "test.pdf")
        assert facts == []
        # Backoff was triggered twice
        assert len(backoff_log) == 2
        assert backoff_log[0][1] == 1  # Attempt 1
        assert backoff_log[1][1] == 2  # Attempt 2
        assert backoff_log[0][2] == 3.0  # Schedule 1: 3s
        assert backoff_log[1][2] == 8.0  # Schedule 2: 8s

    # Test full exhaustion of threshold (5 retries: 3s, 8s, 13s, 17s, 23s) then stop
    backoff_log.clear()
    exhaust_effects = [Exception("Error code: 429 - Rate limit reached") for _ in range(6)]
    with patch("openai.OpenAI") as mock_openai_cls, patch("time.sleep"):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = exhaust_effects
        mock_openai_cls.return_value = mock_client

        import pytest
        with pytest.raises(Exception, match="429"):
            extractor._extract_via_groq(page, "test.pdf")

        # Exactly 5 retries triggered
        assert len(backoff_log) == 5
        wait_times = [log[2] for log in backoff_log]
        assert wait_times == [3.0, 8.0, 13.0, 17.0, 23.0]


def test_semantic_filter_keeps_knowledge_drops_noise():
    """Verify SemanticTextFilter retains facts & concepts while discarding boilerplate."""
    from src.extraction.semantic_filter import SemanticTextFilter
    from src.models.corpus import PageContent

    raw_text = """
    Document Header - CONFIDENTIAL - Page 12 of 100
    All Rights Reserved. Copyright (C) 2024.

    In fiscal year 2024, Delhivery Limited reported revenue from operations of INR 8,142 Crores,
    representing an increase of 13% compared to INR 7,226 Crores in the previous fiscal year.
    EBITDA margin expanded by 240 basis points to 4.2%.

    --- Confidential and Proprietary ---
    Page 12
    """
    page = PageContent(doc_id="delhivery.pdf", page_number=12, raw_text=raw_text, tables=[], images=[])
    units = SemanticTextFilter.extract_semantic_units_from_page(page, "delhivery.pdf")

    assert len(units) > 0
    extracted_text = " ".join(u["text"] for u in units)
    assert "8,142" in extracted_text
    assert "Delhivery Limited" in extracted_text
    # Boilerplate headers and footers should be excluded
    assert "All Rights Reserved" not in extracted_text
    assert "CONFIDENTIAL - Page 12 of 100" not in extracted_text


def test_semantic_chunker_token_limit_and_boundaries():
    """Verify SemanticChunker strictly enforces token limit (<= 220 tokens) and preserves boundaries."""
    from src.extraction.semantic_chunker import SemanticChunker, TokenCounter, DEFAULT_MAX_CHUNK_TOKENS

    chunker = SemanticChunker(max_chunk_tokens=DEFAULT_MAX_CHUNK_TOKENS)

    # Create dummy units totaling ~400 tokens
    paragraph = (
        "The real Gross Domestic Product growth rate for the financial year was recorded at 6.8 percent. "
        "Headline inflation moderated significantly to 4.5 percent from 5.4 percent in the previous quarter. "
        "The industrial output index registered a positive expansion driven by manufacturing and capital goods. "
    )
    units = [
        {"text": paragraph * 3, "page_number": 1, "section_title": "Macro Overview"},
        {"text": paragraph * 2, "page_number": 2, "section_title": "Sector Analysis"},
    ]

    chunks = chunker.chunk_semantic_units(units, "report.pdf")
    assert len(chunks) >= 2

    for c in chunks:
        assert c.token_count <= DEFAULT_MAX_CHUNK_TOKENS
        # Ensure words are not sliced abruptly in middle
        assert not c.text.endswith("-")
        assert len(c.text.strip()) > 0


def test_extract_semantic_chunks_via_groq_mock():
    """Verify FactExtractor.extract_semantic_chunks_via_groq invokes paced chunk requests and reports progress."""
    from unittest.mock import MagicMock, patch
    from src.extraction.fact_extractor import FactExtractor
    from src.models.corpus import PageContent

    extractor = FactExtractor(use_llm_if_available=True)
    extractor.groq_key = "test_groq_key"

    page1 = PageContent(
        doc_id="chad.pdf",
        page_number=9,
        raw_text="In Chad, oil GDP growth averaged 1.7 percent from 2020-23.",
        tables=[],
        images=[],
    )

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "facts": [
            {
                "entity": "Republic of Chad",
                "metric": "Oil GDP Growth",
                "value": "1.7%",
                "temporal_period": "2020-23",
                "unit": "%",
                "verbatim_quote": "oil GDP growth averaged 1.7 percent from 2020-23"
            }
        ]
    })
    mock_chat_completion.choices = [mock_choice]

    progress_steps = []

    with patch("openai.OpenAI") as mock_openai_cls, patch("src.extraction.rate_limiter.groq_rate_limiter.acquire"):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_chat_completion
        mock_openai_cls.return_value = mock_client

        facts = extractor.extract_semantic_chunks_via_groq(
            [page1],
            "chad.pdf",
            progress_callback=lambda cur, tot, pg: progress_steps.append((cur, tot, pg)),
        )

        assert len(facts) >= 1
        assert any(f.metric == "Oil GDP Growth" and f.value == "1.7%" for f in facts)
        assert len(progress_steps) >= 1
        # Check that OpenAI was called with GROQ_CHUNK_SYSTEM_PROMPT
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "Extract atomic facts" in system_msg



"""Rule-based and regex extractor for high-precision numerical & semantic facts."""

import re
from typing import List, Optional, Tuple
from src.models.corpus import PageContent, ExtractedTable
from src.models.fact import Fact, FactCategory, SourceEvidence, ContextDimensions
from src.extraction.noise_filter import NoiseFilter
from src.extraction.entity_detector import EntityDetector

# Common metrics to recognize in financial and operational filings
METRIC_PATTERNS = [
    (r"\b(revenue\s+from\s+operations|operating\s+revenue|revenue\s+from\s+contract|revenue\s+from\s+services)\b", "Revenue from Operations", "INR Cr"),
    (r"\b(adjusted\s+ebitda|ebitda\s+margin|service\s+ebitda|ebitda)\b", "Adjusted EBITDA", "INR Cr / %"),
    (r"\b(express\s+parcel\s+(?:shipments)?(?:\s+volume)?|express\s+parcels\s+shipped)\b", "Express Parcel Shipments", "Million / Billion"),
    (r"\b(pin\s+codes?\s+covered|postal\s+index\s+number\s*\(?[\"“]?pin[\"”]?\)?\s*codes?)\b", "Pin Codes Covered", "Count"),
    (r"\b(last-mile\s+delivery\s+centres?|delivery\s+centres?|sortation\s+centres?)\b", "Last-Mile Delivery Centres", "Count"),
    (r"\b(workforce\s+strength|total\s+employees|headcount)\b", "Workforce Strength", "Count"),
    (r"\b(part-truckload\s+freight|ptl\s+freight)\b", "PTL Freight Volume", "Mn Tonnes"),
    # Qualified GDP & Sector Metrics (strictly prioritized before headline GDP)
    (r"\b(oil\s+(?:sector\s+)?gdp\s+growth|oil\s+gdp|growth\s+in\s+the\s+oil\s+sector|oil\s+sector\s+growth|growth\s+in\s+oil\s+production)\b", "Oil GDP Growth", "%"),
    (r"\b(non-oil\s+(?:sector\s+)?gdp\s+growth|non-oil\s+growth|non-oil\s+gdp|growth\s+in\s+the\s+non-oil\s+sector|non-oil\s+sector\s+growth|non-oil\s+activity)\b", "Non-Oil GDP Growth", "%"),
    (r"\b(compound\s+average\s+gdp\s+growth(?:\s+rate)?|compound\s+annual\s+gdp\s+growth(?:\s+rate)?|compound\s+gdp\s+growth|gdp\s+cagr|cagr\s+of\s+gdp)\b", "Compound Average GDP Growth", "%"),
    (r"\b(nominal\s+gdp\s+growth|growth\s+in\s+nominal\s+gdp|nominal\s+gross\s+domestic\s+product\s+growth)\b", "Nominal GDP Growth", "%"),
    (r"\b(potential\s+gdp\s+growth|potential\s+growth)\b", "Potential GDP Growth", "%"),
    # Regional & Multilateral Criteria / Convergence Targets
    (r"\b(waemu(?:\s+fiscal\s+deficit)?\s+convergence\s+criterion|beac\s+(?:target\s+rate|inflation\s+target|target)|cemac\s+convergence\s+criteria|deficit\s+convergence\s+criterion|convergence\s+criterion\s+of\s+[\d.]+\s*(?:%|percent))\b", "Regional Convergence Target", "%"),
    # Headline Inflation & Deficits
    (r"\b(headline\s+cpi(?:\s+inflation)?|cpi\s+inflation|headline\s+inflation|food\s+inflation|inflation\s+rate|inflation\s+moderated|inflation\s+eased|inflation\s+is\s+projected\s+at|inflationary\s+pressures)\b", "Headline Inflation", "%"),
    (r"\b(gross\s+fiscal\s+deficit|fiscal\s+deficit|gfd)\b", "Gross Fiscal Deficit", "%"),
    # Headline Real GDP Growth (Strict negative lookbehind to avoid qualifier drop)
    (r"\b(?<!oil\s)(?<!non-oil\s)(?<!nominal\s)(?<!potential\s)(?<!compound\s)(?<!average\s)(real\s+gross\s+domestic\s+product\s*\(?gdp\)?\s+growth|gross\s+domestic\s+product\s*\(?gdp\)?\s+growth|real\s+gdp\s+growth|(?<!oil\s)(?<!non-oil\s)gdp\s+growth|grow\s+by\s+[\d.]+\s*(?:%|per\s+cent)|growth\s+in\s+gdp|projects\s+growth\s+at)\b", "Real GDP Growth", "%"),
    (r"\b(global\s+economic\s+growth|global\s+growth)\b", "Global Economic Growth", "%"),
]

TEMPORAL_PATTERNS = [
    r"(\bFY\s*(?:20)?\d{2}(?:[–\-/]\d{2,4})?\b)",
    r"(\bQ[1-4]\s*FY\s*(?:20)?\d{2}\b)",
    r"(\b20\d{2}[–\-]\d{2,4}\b)",
    r"(\b(?:since\s+inception|inception)\b)",
    r"(\b20\d{2}\b)",
]

SCOPE_PATTERNS = [
    (r"\bconsolidated\b", "Consolidated"),
    (r"\bstandalone\b", "Standalone"),
    (r"\brestated\b", "Restated"),
]

PERCENT_METRICS = {
    "Real GDP Growth",
    "Oil GDP Growth",
    "Non-Oil GDP Growth",
    "Compound Average GDP Growth",
    "Nominal GDP Growth",
    "Potential GDP Growth",
    "Regional Convergence Target",
    "Headline Inflation",
    "Global Economic Growth",
    "Gross Fiscal Deficit",
}

KPI_METRICS = {
    "Pin Codes Covered",
    "Express Parcel Shipments",
    "Last-Mile Delivery Centres",
    "Workforce Strength",
    "PTL Freight Volume",
}


def detect_entity_from_doc(doc_filename: str, sample_text: str = "") -> str:
    """Infer the primary entity dynamically without hardcoded Indian bias."""
    return EntityDetector.detect_entity(doc_filename, sample_text)


class RuleBasedExtractor:
    """Extracts facts deterministically from text blocks and structured tables with zero noise."""

    @classmethod
    def extract_from_page(cls, page: PageContent, doc_filename: str, doc_entity: Optional[str] = None) -> List[Fact]:
        facts: List[Fact] = []
        entity = doc_entity or detect_entity_from_doc(doc_filename, page.raw_text[:500])

        # 1. Extract from Structured Tables
        for table in page.tables:
            table_facts = cls._extract_from_table(table, page.page_number, doc_filename, entity)
            facts.extend(table_facts)

        # 2. Extract from Page Text Blocks
        text_facts = cls._extract_from_text(page.raw_text, page.page_number, doc_filename, entity)
        facts.extend(text_facts)

        # Deduplicate facts on (entity, metric, value, temporal_period)
        seen = set()
        deduped = []
        for f in facts:
            key = (f.entity, f.metric.lower(), f.value.lower(), f.context.temporal_period)
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return deduped

    @classmethod
    def extract_pages_concurrently(cls, pages: List[PageContent], doc_filename: str, max_workers: int = 6, progress_callback=None) -> List[Fact]:
        """Extract facts across all pages with consistent document-level entity anchoring."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Anchor entity at the document level from first page headers
        first_pages_text = " ".join(p.raw_text for p in pages[:3]) if pages else ""
        doc_entity = detect_entity_from_doc(doc_filename, first_pages_text)

        all_facts = []
        total = len(pages)
        done = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(cls.extract_from_page, p, doc_filename, doc_entity): p.page_number for p in pages}
            for f in as_completed(futs):
                done += 1
                pg = futs[f]
                try:
                    all_facts.extend(f.result())
                except Exception:
                    pass
                if progress_callback:
                    progress_callback(done, total, pg)

        all_facts.sort(key=lambda f: (f.evidence.page_number, f.metric))
        return all_facts

    @classmethod
    def _extract_metric_from_segment(
        cls, segment: str, metric_name: str, default_unit: str, pat: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """Extract a rigorously validated numerical value and unit for a metric from clean text."""
        clean_seg = NoiseFilter.clean_text_for_extraction(segment)

        # Disqualify exchange rate changes or percentiles from Real GDP Growth
        if metric_name == "Real GDP Growth":
            lowered = clean_seg.lower()
            if any(term in lowered for term in ["exchange rate", "reer", "percentile", "contribution from", "debt dynamics", "climate change"]):
                return None
            if re.search(r"\b(?:oil|non-oil|compound|nominal|potential)\s+gdp\b", lowered):
                return None

        # Disqualify unsegmented table strings with collapsed multiple decimal numbers
        # e.g., '0.5 -0.3 0.2 -0.2 Domestic expenditure 12.4'
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", clean_seg)
        if len(nums) > 3 and not ("%" in clean_seg or "percent" in clean_seg.lower() or "cr" in clean_seg.lower()):
            return None

        # Build prioritized search targets
        search_targets = [clean_seg]
        if pat:
            m_term = re.search(pat, clean_seg, re.IGNORECASE)
            if m_term:
                post_win = clean_seg[m_term.start():min(len(clean_seg), m_term.end() + 150)]
                pre_win = clean_seg[max(0, m_term.start() - 60):m_term.start()]
                if metric_name in KPI_METRICS:
                    search_targets = [pre_win, post_win, clean_seg]
                else:
                    search_targets = [post_win, clean_seg]

        for target in search_targets:
            # 1. Percentage Metrics: Require explicit percent, bps, or decimal rate
            if metric_name in PERCENT_METRICS:
                if metric_name == "Real GDP Growth":
                    pct_pat = r"([><]?\s*[-+]?\d+(?:\.\d+)?)\s*(%|per\s+cent|percent|bps|basis\s+points)(?!\s+of\s+gdp)(?!\w)"
                else:
                    pct_pat = r"([><]?\s*[-+]?\d+(?:\.\d+)?)\s*(%|per\s+cent|percent|bps|basis\s+points)(?!\w)"

                m = re.search(pct_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).replace(" ", "")
                    unit = m.group(2).lower()
                    if "cent" in unit or "%" in unit:
                        return f"{num}%", num, "%"
                    else:
                        return f"{num} {unit}", num, unit

            # 2. Financial / Currency Metrics (Revenue, EBITDA)
            elif metric_name in ("Revenue from Operations", "Adjusted EBITDA"):
                curr_pat = (
                    r"(?:(?:[₹$]|Rs\.?|INR)\s*([><]?\s*[\d,]+(?:\.\d+)?)\s*\b(Cr|Crore|Crores|Mn|Million|Millions|Bn|Billion|Billions|Lakh|Lakhs)?\b"
                    r"|([><]?\s*[\d,]+(?:\.\d+)?)\s*\b(Cr|Crore|Crores|Mn|Million|Millions|Bn|Billion|Billions|Lakh|Lakhs)\b)"
                )
                m = re.search(curr_pat, target, re.IGNORECASE)
                if m:
                    num = (m.group(1) or m.group(3)).strip().rstrip(",")
                    scale = (m.group(2) or m.group(4) or "").strip()
                    if not scale and NoiseFilter.is_bare_year(num):
                        continue
                    val = f"{num} {scale}".strip() if scale else f"{num} {default_unit}"
                    return val, num, scale or default_unit

                if metric_name == "Adjusted EBITDA":
                    pct_pat = r"([><]?\s*[-+]?\d+(?:\.\d+)?)\s*(%|per\s+cent|percent|bps)(?!\w)"
                    m2 = re.search(pct_pat, target, re.IGNORECASE)
                    if m2:
                        num = m2.group(1).replace(" ", "")
                        return f"{num}%", num, "%"

            # 3. Express Parcel Shipments
            elif metric_name == "Express Parcel Shipments":
                vol_pat = r"([><]?\s*[\d,]+(?:\.\d+)?)\s*(Bn|Billion|Billions|Mn|Million|Millions)\b"
                m = re.search(vol_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).strip()
                    scale = m.group(2).strip()
                    return f"{num}{scale}", num, scale

            # 4. Pin Codes Covered
            elif metric_name == "Pin Codes Covered":
                pin_pat = r"([><]?\s*[\d,]+)\s*(?:pin\s*codes?)?"
                m = re.search(pin_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).strip().rstrip(",")
                    digits = re.sub(r"\D", "", num)
                    if digits and int(digits) >= 1000 and not NoiseFilter.is_bare_year(digits):
                        return num, num, "Count"

            # 5. Workforce Strength
            elif metric_name == "Workforce Strength":
                wf_pat = r"([><]?\s*[\d,]+)\s*(?:employees|workforce|people|headcount)?"
                m = re.search(wf_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).strip().rstrip(",")
                    digits = re.sub(r"\D", "", num)
                    if digits and int(digits) >= 100 and not NoiseFilter.is_bare_year(digits):
                        return num, num, "Count"

            # 6. Last-Mile Delivery Centres
            elif metric_name == "Last-Mile Delivery Centres":
                dc_pat = r"([><]?\s*[\d,]+)\s*(?:last-mile\s+delivery\s+centres?|delivery\s+centres?|sortation\s+centres?|centres?|centers?)"
                m = re.search(dc_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).strip().rstrip(",")
                    digits = re.sub(r"\D", "", num)
                    if digits and not NoiseFilter.is_bare_year(digits):
                        return num, num, "Count"

            # 7. PTL Freight Volume
            elif metric_name == "PTL Freight Volume":
                ptl_pat = r"([><]?\s*[\d,]+(?:\.\d+)?)\s*\b(Mn\s+Tonnes|Million\s+Tonnes|Tonnes|MT)\b"
                m = re.search(ptl_pat, target, re.IGNORECASE)
                if m:
                    num = m.group(1).strip().rstrip(",")
                    u = m.group(2).strip()
                    return f"{num} {u}", num, u

        return None

    @classmethod
    def _extract_from_text(
        cls, text: str, page_number: int, doc_filename: str, entity: str
    ) -> List[Fact]:
        facts: List[Fact] = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # PASS 1: Line-by-line & KPI Block Detection
        for i, line in enumerate(lines):
            if NoiseFilter.is_boilerplate(line):
                continue

            # Reject unsegmented table rows in text
            if len(re.findall(r"[-+]?\d+(?:\.\d+)?", line)) > 3 and not ("%" in line or "percent" in line.lower() or "cr" in line.lower()):
                continue

            # Qualifier check across line breaks: if line starts with 'GDP growth', inspect end of previous line
            prefix = ""
            if i > 0 and re.match(r"^(?:real\s+)?gdp\s+growth\b", line, re.IGNORECASE):
                prev_end = lines[i - 1].lower()
                for mod in ["oil", "non-oil", "compound average", "nominal", "potential"]:
                    if prev_end.endswith(mod) or prev_end.endswith(f"saw {mod}"):
                        prefix = f"{mod} "
                        break

            for pat, metric_name, default_unit in METRIC_PATTERNS:
                scan_line = f"{prefix}{line}" if prefix else line
                if re.search(pat, scan_line, re.IGNORECASE):
                    res = cls._extract_metric_from_segment(scan_line, metric_name, default_unit, pat=pat)
                    best_line = scan_line

                    # Backward check ONLY for KPI presentation blocks (numbers on line above)
                    if not res and i > 0 and metric_name in KPI_METRICS:
                        for prev_idx in range(i - 1, max(-1, i - 4), -1):
                            prev_line = lines[prev_idx]
                            if re.fullmatch(r"\(\d+(?:,\s*\d+)*\)", prev_line) or re.fullmatch(r"\(?\d+\)?", prev_line):
                                continue
                            candidate_res = cls._extract_metric_from_segment(
                                f"{prev_line} {line}", metric_name, default_unit, pat=pat
                            )
                            if candidate_res:
                                res = candidate_res
                                best_line = f"{prev_line} {line}"
                                break

                    # Forward check for line-wrapped prose (e.g. 'growth of 0.1' + 'percent')
                    if not res and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        candidate_res = cls._extract_metric_from_segment(
                            f"{scan_line} {next_line}", metric_name, default_unit, pat=pat
                        )
                        if candidate_res:
                            res = candidate_res
                            best_line = f"{scan_line} {next_line}"

                    if res:
                        full_val, raw_num, unit = res
                        if NoiseFilter.is_bare_year(full_val):
                            continue

                        temporal = cls._detect_temporal(best_line) or cls._detect_temporal(text[:300])
                        scope = cls._detect_scope(best_line) or cls._detect_scope(text[:300])

                        evidence = SourceEvidence(
                            document_id=doc_filename,
                            page_number=page_number,
                            section_title=f"Page {page_number}",
                            content_type="text",
                            verbatim_quote=best_line[:300],
                        )

                        context = ContextDimensions(
                            temporal_period=temporal,
                            reporting_scope=scope,
                            unit=unit or default_unit,
                        )

                        facts.append(
                            Fact(
                                category=FactCategory.NUMERICAL,
                                entity=entity,
                                metric=metric_name,
                                value=full_val,
                                raw_value=raw_num,
                                context=context,
                                evidence=evidence,
                                confidence_score=0.92,
                            )
                        )
                        break

        # PASS 2: Prose Sentence-Level Extraction on Normalized Paragraphs
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            cleaned_para = NoiseFilter.clean_text_for_extraction(para)
            single_line_para = " ".join(l.strip() for l in cleaned_para.splitlines() if l.strip())
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", single_line_para)
            for sent in sentences:
                if len(sent.strip()) < 10 or NoiseFilter.is_boilerplate(sent):
                    continue

                for pat, metric_name, default_unit in METRIC_PATTERNS:
                    if re.search(pat, sent, re.IGNORECASE):
                        res = cls._extract_metric_from_segment(sent, metric_name, default_unit, pat=pat)
                        if res:
                            full_val, raw_num, unit = res
                            if NoiseFilter.is_bare_year(full_val):
                                continue

                            temporal = cls._detect_temporal(sent)
                            scope = cls._detect_scope(sent)

                            evidence = SourceEvidence(
                                document_id=doc_filename,
                                page_number=page_number,
                                section_title=f"Page {page_number}",
                                content_type="text",
                                verbatim_quote=sent[:300],
                            )

                            context = ContextDimensions(
                                temporal_period=temporal,
                                reporting_scope=scope,
                                unit=unit or default_unit,
                            )

                            facts.append(
                                Fact(
                                category=FactCategory.NUMERICAL,
                                entity=entity,
                                metric=metric_name,
                                value=full_val,
                                raw_value=raw_num,
                                context=context,
                                evidence=evidence,
                                confidence_score=0.90,
                            )
                        )
                        break
        return facts

    @classmethod
    def _extract_from_table(
        cls, table: ExtractedTable, page_number: int, doc_filename: str, entity: str
    ) -> List[Fact]:
        facts: List[Fact] = []
        if not table.headers or not table.rows:
            return facts

        for row in table.rows:
            if not row or len(row) < 2:
                continue
            row_label = row[0].strip()
            if not row_label or len(row_label) > 100 or len(row_label) < 3 or NoiseFilter.is_boilerplate(row_label):
                continue

            # Disqualify debt dynamics contribution rows from being treated as Real GDP Growth
            if "contribution from" in row_label.lower() or "debt dynamics" in row_label.lower():
                continue

            for pat, metric_name, default_unit in METRIC_PATTERNS:
                if re.search(pat, row_label, re.IGNORECASE):
                    for col_idx in range(1, len(row)):
                        cell_val = NoiseFilter.clean_punctuation(row[col_idx].strip())
                        if not cell_val or len(cell_val) > 30 or cell_val in ("-", "NA", "N/A", "--", "…", "..."):
                            continue
                        if NoiseFilter.is_bare_year(cell_val):
                            continue

                        # Reject collapsed multi-column cells (e.g. '5.4 6.5 1.5 1.1')
                        if re.search(r"[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?", cell_val):
                            continue

                        header_name = table.headers[col_idx] if col_idx < len(table.headers) else ""
                        temporal = cls._detect_temporal(header_name) or cls._detect_temporal(table.markdown_repr)

                        unit = default_unit
                        if "%" in row_label or "(in percent)" in row_label.lower():
                            unit = "%"
                            if not cell_val.endswith("%") and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cell_val):
                                cell_val = f"{cell_val}%"
                        elif "cr" in row_label.lower():
                            unit = "INR Cr"
                        elif "million" in row_label.lower():
                            unit = "INR Mn"

                        evidence = SourceEvidence(
                            document_id=doc_filename,
                            page_number=page_number,
                            section_title=table.caption,
                            content_type="table",
                            verbatim_quote=f"Row: {row_label} | Col: {header_name} = {cell_val}",
                        )

                        fact = Fact(
                            category=FactCategory.NUMERICAL,
                            entity=entity,
                            metric=metric_name,
                            value=cell_val,
                            raw_value=re.sub(r"[^\d.\-+]", "", cell_val),
                            context=ContextDimensions(
                                temporal_period=temporal,
                                reporting_scope="Consolidated" if "consolidated" in table.markdown_repr.lower() else None,
                                unit=unit,
                            ),
                            evidence=evidence,
                            confidence_score=0.95,
                        )
                        facts.append(fact)
                    break
        return facts

    @staticmethod
    def _detect_temporal(text: str) -> Optional[str]:
        for pat in TEMPORAL_PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _detect_scope(text: str) -> Optional[str]:
        for pat, scope in SCOPE_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return scope
        return None

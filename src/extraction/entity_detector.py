"""Dynamic sovereign and enterprise entity detection module."""

import os
import re
from typing import Optional, Dict

# Comprehensive sovereign gazetteer mapping aliases, ISO terms, and adjective forms
SOVEREIGN_GAZETTEER: Dict[str, str] = {
    # Chad
    "chad": "Republic of Chad",
    "tchad": "Republic of Chad",
    "chadian": "Republic of Chad",
    # Niger
    "niger": "Republic of Niger",
    "nigerien": "Republic of Niger",
    # Nigeria (distinguished from Niger)
    "nigeria": "Federal Republic of Nigeria",
    "nigerian": "Federal Republic of Nigeria",
    # India
    "india": "Indian Economy",
    "indea": "Indian Economy",
    "indian": "Indian Economy",
    "rbi": "Indian Economy",
    "reserve bank of india": "Indian Economy",
    # Regional Central Banks & Unions
    "beac": "CEMAC Region",
    "bceao": "WAEMU Region",
    "cemac": "CEMAC Region",
    "waemu": "WAEMU Region",
    "uemoa": "WAEMU Region",
    # Corporate Entities
    "delhivery": "Delhivery Limited",
    # Other Sovereigns
    "ghana": "Republic of Ghana",
    "kenya": "Republic of Kenya",
    "senegal": "Republic of Senegal",
    "cameroon": "Republic of Cameroon",
    "cameroun": "Republic of Cameroon",
    "gabon": "Gabonese Republic",
    "ivory coast": "Republic of Côte d'Ivoire",
    "côte d'ivoire": "Republic of Côte d'Ivoire",
    "cote d'ivoire": "Republic of Côte d'Ivoire",
    "south africa": "Republic of South Africa",
    "egypt": "Arab Republic of Egypt",
    "bangladesh": "People's Republic of Bangladesh",
    "pakistan": "Islamic Republic of Pakistan",
    "sri lanka": "Democratic Socialist Republic of Sri Lanka",
    "brazil": "Federative Republic of Brazil",
    "mexico": "United Mexican States",
    "indonesia": "Republic of Indonesia",
    "vietnam": "Socialist Republic of Vietnam",
}

MULTILATERAL_METRICS = {
    "Regional Convergence Target",
    "Multilateral Convergence Criterion",
    "Global Economic Growth",
    "Global Growth",
    "Brent Crude Oil Price",
    "Commodity Price Volatility",
}


class EntityDetector:
    """Detects and resolves sovereign and corporate entities from documents with zero state leakage."""

    @classmethod
    def detect_entity(cls, doc_filename: str, sample_text: str = "") -> str:
        """
        Infer the primary sovereign or enterprise entity from document filename and header text.
        Never defaults generic IMF or international headers to 'Indian Economy'.
        """
        fn_clean = os.path.basename(doc_filename).lower()

        # 1. Corporate check on filename
        if "delhivery" in fn_clean:
            return "Delhivery Limited"

        # 2. Sovereign check on filename tokens
        sorted_keys = sorted(SOVEREIGN_GAZETTEER.keys(), key=lambda k: len(k), reverse=True)
        for k in sorted_keys:
            if re.search(r"(?:^|[\-_.\d])" + re.escape(k) + r"(?:$|[\-_.\d])", fn_clean):
                return SOVEREIGN_GAZETTEER[k]

        # 3. Header inspection: Check IMF Country Report sovereign header
        if sample_text:
            imf_match = re.search(
                r"IMF\s+Country\s+Report(?:\s+No\.\s*[\d/]+)?\s*[\r\n]+([A-Z\s]{3,30})\b",
                sample_text,
            )
            if imf_match:
                cand = imf_match.group(1).strip().lower()
                for k in sorted_keys:
                    if k in cand:
                        return SOVEREIGN_GAZETTEER[k]

            art_match = re.search(
                r"Article\s+IV\s+Consultation\s+(?:with\s+)?([A-Za-z\s]+)",
                sample_text,
                re.IGNORECASE,
            )
            if art_match:
                cand = art_match.group(1).strip().lower()
                for k in sorted_keys:
                    if k in cand:
                        return SOVEREIGN_GAZETTEER[k]

            if "delhivery" in sample_text.lower():
                return "Delhivery Limited"

            sample_lower = sample_text[:5000].lower()
            for k in sorted_keys:
                if re.search(r"\b" + re.escape(k) + r"\b", sample_lower):
                    return SOVEREIGN_GAZETTEER[k]

        return "Enterprise / Document Subject"

    @classmethod
    def is_multilateral_metric(cls, metric_name: str) -> bool:
        """Check if metric refers to a shared regional / multilateral rule or benchmark."""
        m_lower = metric_name.lower()
        if any(term in m_lower for term in ["convergence target", "convergence criterion", "global economic growth", "global growth", "brent"]):
            return True
        return metric_name in MULTILATERAL_METRICS

    @classmethod
    def are_entities_compatible(cls, entity_a: str, entity_b: str, metric_name: Optional[str] = None) -> bool:
        """
        Check whether two entities are legally and logically compatible for comparison.
        Cross-sovereign comparison of national accounts (e.g. Chad vs Niger GDP) is strictly forbidden.
        """
        ea = (entity_a or "").strip().lower()
        eb = (entity_b or "").strip().lower()

        if ea == eb:
            return True

        if metric_name and cls.is_multilateral_metric(metric_name):
            return True

        generic_terms = {"enterprise / document subject", "document subject", "enterprise", "all"}
        if (ea in generic_terms or eb in generic_terms) and not (cls._is_known_sovereign(ea) or cls._is_known_sovereign(eb)):
            return True

        return False

    @classmethod
    def _is_known_sovereign(cls, entity_name: str) -> bool:
        name = entity_name.lower()
        return any(sov.lower() in name for sov in [
            "chad", "niger", "india", "ghana", "kenya", "senegal", "cameroon",
            "gabon", "egypt", "bangladesh", "pakistan", "republic", "kingdom",
            "sovereign"
        ])

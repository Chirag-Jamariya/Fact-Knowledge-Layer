"""Unit normalization and numerical equivalence engine."""

import re
from typing import Tuple, Optional


def clean_number_str(s: str) -> Optional[float]:
    """Strip currency symbols, commas, and extract float value."""
    if not s or s.strip() in (",", "-", "NA", "N/A", "None", ""):
        return None
    # Remove currency symbols and non-numeric chars except decimal point and minus
    cleaned = re.sub(r"[^\d.\-+]", "", s.replace(",", ""))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


class UnitConverter:
    """Normalizes numbers across Indian (Lakh, Crore) and International (Million, Billion) scales."""

    SCALE_FACTORS = {
        "cr": 10_000_000.0,
        "crore": 10_000_000.0,
        "crores": 10_000_000.0,
        "lakh": 100_000.0,
        "lakhs": 100_000.0,
        "mn": 1_000_000.0,
        "million": 1_000_000.0,
        "millions": 1_000_000.0,
        "bn": 1_000_000_000.0,
        "billion": 1_000_000_000.0,
        "billions": 1_000_000_000.0,
        "k": 1_000.0,
        "thousand": 1_000.0,
    }

    @classmethod
    def get_canonical_scale(cls, text: str) -> float:
        """Determine multiplying scale from string unit or value text."""
        lower = text.lower()
        for key, factor in cls.SCALE_FACTORS.items():
            if re.search(rf"\b{key}\b", lower):
                return factor
        return 1.0

    @classmethod
    def to_canonical_number(cls, val_str: str, unit_str: Optional[str] = "") -> Optional[float]:
        """Convert any formatted number string with scale into an absolute baseline scalar float."""
        num = clean_number_str(val_str)
        if num is None:
            return None

        unit_text = f"{val_str} {unit_str or ''}"
        scale = cls.get_canonical_scale(unit_text)
        return num * scale

    @classmethod
    def are_values_equivalent(
        cls, val_a: str, unit_a: Optional[str], val_b: str, unit_b: Optional[str], tolerance_pct: float = 0.02
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if two values are mathematically equivalent within a given percentage tolerance.
        Returns (is_equivalent, explanation).
        """
        # Exact string match
        if val_a.strip().lower() == val_b.strip().lower():
            return True, "Identical verbatim values."

        canon_a = cls.to_canonical_number(val_a, unit_a)
        canon_b = cls.to_canonical_number(val_b, unit_b)

        if canon_a is not None and canon_b is not None and canon_a > 0:
            diff_ratio = abs(canon_a - canon_b) / canon_a
            if diff_ratio <= tolerance_pct:
                scale_a = cls.get_canonical_scale(f"{val_a} {unit_a or ''}")
                scale_b = cls.get_canonical_scale(f"{val_b} {unit_b or ''}")
                if scale_a != scale_b:
                    return (
                        True,
                        f"Unit conversion equivalence applied: {val_a} ({unit_a or ''}) normalizes to {canon_a:,.0f} base units, "
                        f"which matches {val_b} ({unit_b or ''}) at {canon_b:,.0f} base units (within {diff_ratio*100:.2f}% rounding variance)."
                    )
                return True, f"Normalized numerical equivalence (within {diff_ratio*100:.1f}% rounding variance)."

        return False, None

    @classmethod
    def explain_unit_normalization(
        cls, val_a: str, unit_a: Optional[str], val_b: str, unit_b: Optional[str]
    ) -> Optional[str]:
        """Generate human-readable audit text explaining the unit conversion between two values."""
        canon_a = cls.to_canonical_number(val_a, unit_a)
        canon_b = cls.to_canonical_number(val_b, unit_b)
        if canon_a is not None and canon_b is not None and canon_a > 0:
            diff_pct = abs(canon_a - canon_b) / canon_a * 100
            scale_a = cls.get_canonical_scale(f"{val_a} {unit_a or ''}")
            scale_b = cls.get_canonical_scale(f"{val_b} {unit_b or ''}")
            if scale_a != scale_b:
                return (
                    f"Converted '{val_a}' ({unit_a or 'unscaled'}) and '{val_b}' ({unit_b or 'unscaled'}) "
                    f"to canonical scalar: {canon_a:,.0f} vs {canon_b:,.0f} ({diff_pct:.2f}% variance)."
                )
            if diff_pct > 0:
                return f"Numerical comparison: {canon_a:,.0f} vs {canon_b:,.0f} ({diff_pct:.2f}% rounding variance)."
            return "Values are identically matched."
        return None

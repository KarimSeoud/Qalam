"""
Module 1 — Text Normalization
==============================
Handles Unicode normalization, character unification, diacritic handling,
and punctuation standardization for Arabic text.

Arabic-specific challenges addressed:
- Multiple Unicode representations of Alef (أ إ آ ا ٱ)
- Hamza forms (ء ؤ ئ)
- Alef maksura ↔ yeh (ى ↔ ي) — most common Arabic normalization step
- Teh marbuta ↔ heh (ة ↔ ه) — optional collapse
- Persian/Urdu letters appearing in scraped Arabic (ی ک ہ ۀ)
- Tatweel / kashida (ـ) elongation character
- Diacritics (tashkeel) — configurable keep/strip, includes superscript alef
- Eastern vs Western Arabic numerals (٠١٢ and ۰۱۲ vs 012)
- Presentation forms (FB50–FDFF, FE70–FEFF) — decomposed via NFKC, not deleted,
  so lam-alef ligatures (ﻻ ﻷ ﻹ ﻵ) survive as ل+ا instead of being dropped
- Bidi control characters and ZWNJ/ZWJ — strip by default
- Repeated-character emphasis ("ممتاااز") — optional collapse
- Inconsistent punctuation (Arabic vs Latin question/comma marks)
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizerConfig:
    """Configuration for the Arabic normalizer."""

    # Alef normalization: collapse أ إ آ ٱ → ا
    normalize_alef: bool = True

    # Hamza normalization: collapse ؤ ئ → ء (use for pretraining; disable for classical)
    normalize_hamza: bool = False

    # Alef maksura → yeh: ى → ي (standard Arabic normalization; default on)
    normalize_alef_maksura: bool = True

    # Teh marbuta → heh: ة → ه (loses grammatical info; default off,
    # but useful for retrieval / fuzzy dedup)
    normalize_teh_marbuta: bool = False

    # Persian/Urdu letters → Arabic equivalents: ی→ي, ک→ك, ہ→ه, ۀ→ه
    # (letters with no Arabic equivalent — گ چ پ ژ — are left alone)
    normalize_persian: bool = True

    # Remove tatweel / kashida (ـ)
    remove_tatweel: bool = True

    # Diacritics (tashkeel): "strip", "keep", or "keep_classical"
    # "strip"           — remove all diacritics (best for most LLM pretraining)
    # "keep"            — preserve all diacritics
    # "keep_classical"  — keep diacritics only if text contains Quranic markers
    #                    (U+06D6–U+06ED); otherwise strip
    diacritics: str = "strip"

    # Convert Eastern (٠-٩) and Extended Arabic-Indic (۰-۹) numerals to Western (0-9)
    normalize_numerals: bool = True

    # Convert Arabic punctuation to standard Unicode equivalents
    normalize_punctuation: bool = True

    # Decompose Unicode presentation forms (FB50–FDFF, FE70–FEFF) via NFKC.
    # Critical: lam-alef ligatures (ﻻ ﻷ ﻹ ﻵ) decompose to ل+ا rather than being dropped.
    remove_presentation_forms: bool = True

    # Strip bidi control characters and ZWNJ/ZWJ (common in scraped data)
    remove_bidi_controls: bool = True

    # Collapse N+ repeats of the same character to this many (e.g. ممتاااز → ممتاز).
    # None disables. 2 is typical. Applied after diacritic stripping.
    collapse_repeated_chars: Optional[int] = None

    # Strip leading/trailing whitespace and collapse internal whitespace
    normalize_whitespace: bool = True

    # Remove URLs
    remove_urls: bool = False

    # Remove email addresses
    remove_emails: bool = False

    # Remove HTML tags
    remove_html: bool = True

    # Minimum Arabic character ratio to consider text "Arabic" (0.0 = no filter)
    min_arabic_ratio: float = 0.0


class QalamNormalizer:
    """
    Normalizes Arabic text for downstream NLP / LLM use.

    Usage:
        normalizer = QalamNormalizer()
        clean = normalizer.normalize("مرحبًا بِكُم")
        # → "مرحبا بكم"  (diacritics stripped, default config)

        # Keep diacritics (e.g. for Quranic / classical data)
        normalizer = QalamNormalizer(NormalizerConfig(diacritics="keep"))
    """

    # ------------------------------------------------------------------ #
    # Unicode ranges & character sets
    # ------------------------------------------------------------------ #
    ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

    # All Arabic diacritics including superscript alef (U+0670), which is the
    # most common "diacritic" in printed Quran and was missing from the original.
    DIACRITICS = re.compile(
        r"[ً-ٟ"   # fathatan..wavy hamza below
        r"ٰ"           # superscript alef ٰ
        r"ؐ-ؚ"    # Arabic sign sallallahou..small waw
        r"ۖ-ۜ"    # Quranic annotation signs
        r"۟-ۤ"    # Quranic annotation signs
        r"ۧۨ"     # Quranic small high yeh / noon
        r"۪-ۭ]"   # Quranic marks
    )

    # Quranic annotation markers — presence implies classical/Quranic text.
    QURANIC_MARKERS = re.compile(r"[ۖ-ۭؐ-ؚ]")

    TATWEEL = re.compile(r"ـ+")

    # Combined Eastern Arabic-Indic (٠-٩) + Extended Arabic-Indic / Persian (۰-۹)
    EASTERN_NUMERALS = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    )

    # Range covering all Arabic presentation forms
    _PRES_FORM_RANGE = re.compile(r"[ﭐ-﷿ﹰ-﻿]+")

    # Alef variants → bare alef ا. Includes ٱ (alef wasla).
    ALEF_MAP = str.maketrans(
        "آأإٱ",  # آ أ إ ٱ
        "اااا",  # ا ا ا ا
    )

    # Hamza variants → standalone hamza ء
    HAMZA_MAP = str.maketrans(
        "ؤئ",  # ؤ ئ
        "ءء",  # ء ء
    )

    # Alef maksura → yeh
    ALEF_MAKSURA_MAP = str.maketrans("ى", "ي")  # ى → ي

    # Teh marbuta → heh
    TEH_MARBUTA_MAP = str.maketrans("ة", "ه")  # ة → ه

    # Persian/Urdu → Arabic (only chars that *have* an Arabic equivalent).
    # Persian-only letters (گ چ پ ژ) are intentionally left untouched.
    PERSIAN_MAP = str.maketrans(
        "یکہۀ",  # ی ک ہ ۀ
        "يكهه",  # ي ك ه ه
    )

    # ZWNJ, ZWJ, LRM, RLM, ALM, LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI
    BIDI_CONTROLS = re.compile(r"[​-‏‪-‮⁦-⁩؜]")

    ARABIC_PUNCTUATION_MAP = str.maketrans(
        "،؛؟٪٫٬",
        ",;?%.,",
    )

    URL_RE = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
    HTML_RE = re.compile(r"<[^>]+>")
    WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def normalize(self, text: str) -> str:
        """Normalize a single Arabic string."""
        cfg = self.config

        # 1. Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # 2. Remove HTML
        if cfg.remove_html:
            text = self.HTML_RE.sub(" ", text)

        # 3. Remove URLs / emails
        if cfg.remove_urls:
            text = self.URL_RE.sub(" ", text)
        if cfg.remove_emails:
            text = self.EMAIL_RE.sub(" ", text)

        # 4. Presentation forms — decompose via NFKC so ligatures like ﻻ become ل+ا
        #    instead of being silently deleted.
        if cfg.remove_presentation_forms:
            text = self._decompose_presentation_forms(text)

        # 5. Bidi controls / ZWNJ / ZWJ
        if cfg.remove_bidi_controls:
            text = self.BIDI_CONTROLS.sub("", text)

        # 6. Tatweel
        if cfg.remove_tatweel:
            text = self.TATWEEL.sub("", text)

        # 7. Diacritics
        if cfg.diacritics == "strip":
            text = self.DIACRITICS.sub("", text)
        elif cfg.diacritics == "keep_classical":
            if not self.QURANIC_MARKERS.search(text):
                text = self.DIACRITICS.sub("", text)
        # "keep" → no-op

        # 8. Alef unification
        if cfg.normalize_alef:
            text = text.translate(self.ALEF_MAP)

        # 9. Hamza
        if cfg.normalize_hamza:
            text = text.translate(self.HAMZA_MAP)

        # 10. Persian/Urdu → Arabic (runs before alef-maksura so Persian yeh ی,
        #     which would otherwise stay distinct from ي, gets collapsed properly)
        if cfg.normalize_persian:
            text = text.translate(self.PERSIAN_MAP)

        # 11. Alef maksura → yeh
        if cfg.normalize_alef_maksura:
            text = text.translate(self.ALEF_MAKSURA_MAP)

        # 12. Teh marbuta → heh
        if cfg.normalize_teh_marbuta:
            text = text.translate(self.TEH_MARBUTA_MAP)

        # 13. Numerals
        if cfg.normalize_numerals:
            text = text.translate(self.EASTERN_NUMERALS)

        # 14. Punctuation
        if cfg.normalize_punctuation:
            text = text.translate(self.ARABIC_PUNCTUATION_MAP)

        # 15. Collapse repeated characters (after diacritics so "اااا" collapses
        #     regardless of intervening shaddah/fatha)
        if cfg.collapse_repeated_chars and cfg.collapse_repeated_chars >= 1:
            text = self._collapse_repeats(text, cfg.collapse_repeated_chars)

        # 16. Whitespace
        if cfg.normalize_whitespace:
            text = self.WHITESPACE_RE.sub(" ", text).strip()

        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a list of strings."""
        return [self.normalize(t) for t in texts]

    def arabic_ratio(self, text: str) -> float:
        """Return the fraction of characters that are Arabic script."""
        if not text:
            return 0.0
        arabic_chars = len(self.ARABIC_RANGE.findall(text))
        return arabic_chars / len(text)

    def is_arabic(self, text: str, threshold: float = 0.5) -> bool:
        """Return True if text is predominantly Arabic."""
        return self.arabic_ratio(text) >= threshold

    def diff(self, original: str, normalized: str) -> dict:
        """
        Return a human-readable diff showing what changed.
        Useful for the AI rewriter feature and debugging.
        """
        cfg = self.config
        changes = []
        if original != normalized:
            if len(original) != len(normalized):
                changes.append(
                    f"length: {len(original)} → {len(normalized)} chars "
                    f"({len(original) - len(normalized):+d})"
                )
            if cfg.diacritics == "strip" and self.DIACRITICS.search(original):
                changes.append("diacritics stripped")
            if cfg.remove_tatweel and "ـ" in original:
                changes.append("tatweel removed")
            if cfg.normalize_alef:
                alefs = sum(1 for c in original if c in "آأإٱ")
                if alefs:
                    changes.append(f"{alefs} alef variant(s) unified")
            if cfg.normalize_hamza:
                hamzas = sum(1 for c in original if c in "ؤئ")
                if hamzas:
                    changes.append(f"{hamzas} hamza variant(s) unified")
            if cfg.normalize_alef_maksura and "ى" in original:
                changes.append("alef maksura → yeh")
            if cfg.normalize_teh_marbuta and "ة" in original:
                changes.append("teh marbuta → heh")
            if cfg.normalize_persian and any(c in original for c in "یکہۀ"):
                changes.append("persian letters unified")
            if cfg.remove_presentation_forms and self._PRES_FORM_RANGE.search(original):
                changes.append("presentation forms decomposed")
            if cfg.remove_bidi_controls and self.BIDI_CONTROLS.search(original):
                changes.append("bidi controls stripped")
        return {
            "original": original,
            "normalized": normalized,
            "changed": original != normalized,
            "changes": changes,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @classmethod
    def _decompose_presentation_forms(cls, text: str) -> str:
        """
        NFKC-decompose only characters in the Arabic presentation form ranges,
        leaving the rest of the string untouched. This turns lam-alef ligatures
        and isolated/medial/final glyphs back into their base letters, instead
        of deleting them outright.
        """
        def _sub(match: re.Match) -> str:
            return unicodedata.normalize("NFKC", match.group(0))
        return cls._PRES_FORM_RANGE.sub(_sub, text)

    @staticmethod
    def _collapse_repeats(text: str, keep: int) -> str:
        if keep < 1:
            return text
        # Collapse runs of the same character longer than `keep` down to `keep`.
        # Excludes whitespace/newlines.
        pattern = re.compile(r"(\S)\1{" + str(keep) + r",}")
        return pattern.sub(lambda m: m.group(1) * keep, text)

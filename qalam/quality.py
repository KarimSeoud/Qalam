"""
Module 3 — Quality Scoring & Filtering
========================================
Scores Arabic text samples on multiple quality dimensions and filters
low-quality examples before training.

Scoring dimensions:
  1. Arabic ratio       — fraction of Arabic script characters
  2. Length score       — penalizes very short or very long texts
  3. Repetition score   — detects copy-pasted / boilerplate text
  4. Noise score        — excessive punctuation, symbols, non-text characters
  5. Toxicity flag      — keyword-based toxic content detection (OFF by default)
  6. Boilerplate flag   — common web boilerplate phrases
  7. Language mix score — penalizes heavy code-switching

Final quality score: weighted average of all dimensions (0.0 – 1.0).
Texts below the threshold are flagged for removal.

Design note: This is intentionally a heuristic/rule-based scorer.
A perplexity-based scorer (using a small Arabic LM) can be plugged in
via the optional `perplexity_scorer` argument.

The toxicity check is OFF by default. The keyword list is intentionally
small and will produce both false positives (news mentioning "إرهاب" or
"داعش" gets flagged) and false negatives (most actual harmful content slips
through). Enable only if you understand these limits and pair with a real
classifier downstream.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class QualityConfig:
    # Minimum Arabic character ratio (0–1)
    min_arabic_ratio: float = 0.5

    # Text length bounds (in characters)
    min_length: int = 50
    max_length: int = 100_000

    # Maximum fraction of duplicate n-grams (repetition detection)
    max_repetition_ratio: float = 0.3

    # Maximum allowed noise ratio (non-Arabic, non-ASCII, non-Arabic-supplement chars)
    max_noise_ratio: float = 0.15

    # Minimum quality score to pass the filter (0.0 – 1.0)
    min_quality_score: float = 0.5

    # Enable toxicity keyword flagging. OFF by default — the lexicon is
    # narrow and naive (see module docstring). Wire to a real classifier
    # before relying on this for safety.
    check_toxicity: bool = False

    # If toxicity check is enabled, should a toxic flag block passing?
    # Off by default since the heuristic is unreliable — flagging without
    # blocking lets you triage rather than silently drop.
    toxic_blocks_pass: bool = False

    # Enable boilerplate detection
    check_boilerplate: bool = True

    # If boilerplate is detected, should it block passing?
    boilerplate_blocks_pass: bool = False

    # Weights for the composite score
    weights: Optional[dict] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "arabic_ratio": 0.25,
                "length": 0.15,
                "repetition": 0.25,
                "noise": 0.20,
                "language_mix": 0.15,
            }


# ---------------------------------------------------------------------------
# Flag codes — machine-readable identifiers paired with the human-readable
# strings in QualityResult.flags. Use these when filtering or analytics-ing
# downstream rather than string-matching the human messages.
# ---------------------------------------------------------------------------
FLAG_TOXIC = "toxic_content"
FLAG_BOILERPLATE = "boilerplate"
FLAG_LOW_ARABIC = "low_arabic_ratio"
FLAG_TOO_SHORT = "too_short"
FLAG_TOO_LONG = "too_long"
FLAG_HIGH_REPETITION = "high_repetition"
FLAG_HIGH_NOISE = "high_noise"


@dataclass
class QualityResult:
    text: str
    score: float                    # Final composite score 0–1
    passed: bool                    # True if score >= threshold
    arabic_ratio: float
    length_score: float
    repetition_score: float
    noise_score: float
    language_mix_score: float
    is_toxic: bool
    is_boilerplate: bool
    flags: list[str]                # Human-readable reasons for filtering
    flag_codes: list[str] = field(default_factory=list)  # Machine-readable codes


# ---------------------------------------------------------------------------
# Toxicity keyword list (Arabic) — extend as needed.
# WARNING: This is a deliberately small, fragile heuristic. It will mislabel
# legitimate news ("الحرب على الإرهاب", reports about "داعش") as toxic and
# completely miss the most important categories (self-harm: "انتحر", "قتل نفسه";
# violence: most slurs and threats; sexual content; CSAM). For any production
# safety use case, replace with a real classifier (e.g. Detoxify-Arabic,
# Perspective API, or a fine-tuned model). OFF by default; see QualityConfig.
# ---------------------------------------------------------------------------
_TOXICITY_PATTERNS = re.compile(
    r"(إرهاب|تفجير|اغتصاب|انتحار|انتحر|قتل نفس[هكي]|داعش|حرق|ذبح)",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Boilerplate patterns — common web cruft, both English and Arabic.
# Compiled with IGNORECASE so case variants of Latin terms (Cookie/cookie)
# are handled by the same pattern.
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in [
        # Legal / footer
        r"جميع الحقوق محفوظة",
        r"حقوق الطبع والنشر",
        r"سياسة الخصوصية",
        r"شروط الاستخدام",
        r"اتفاقية الاستخدام",
        r"©\s*\d{4}",
        # Auth / signup
        r"تسجيل الدخول",
        r"تسجيل الخروج",
        r"إنشاء حساب",
        r"اشترك الآن",
        r"اشترك في النشرة",
        # Navigation / CTAs
        r"للمزيد من المعلومات",
        r"انقر هنا",
        r"اضغط هنا",
        r"اقرأ المزيد",
        r"شارك المقال",
        r"شارك على",
        r"أضف تعليقا?ً?",
        r"اترك تعليقا?ً?",
        r"\d+ تعليق",
        # About / contact
        r"من نحن",
        r"تواصل معنا",
        r"اتصل بنا",
        r"أرسل رسالة",
        r"تابعنا على",
        # Search / pagination
        r"لا توجد نتائج",
        r"نتائج البحث",
        r"صفحة \d+ من \d+",
        r"الصفحة (السابقة|التالية)",
        # Errors
        r"الصفحة غير موجودة",
        r"خطأ \d{3}",
        # Cookies / GDPR
        r"cookie|كوكيز",
        r"powered by",
        r"all rights reserved",
    ]
]

# Arabic script ranges including:
#  - Arabic block U+0600-U+06FF
#  - Arabic Supplement U+0750-U+077F  (extended letters for non-Arabic languages)
#  - Arabic Extended-A U+08A0-U+08FF  (Quranic / Koranic + further extensions)
#  - Arabic Presentation Forms (handled separately; normalization should remove)
# Plus ASCII printable + whitespace + ASCII digits as legitimate non-noise.
# (Previous version permitted Latin-1 Supplement / Latin Extended which are
# rarely "Arabic text"; it also excluded the Arabic Supplement/Extended-A.)
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_NOISE_RE = re.compile(r"[^؀-ۿݐ-ݿࢠ-ࣿ -~\s]")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


class QualityScorer:
    """
    Scores and filters Arabic text for training quality.

    Usage:
        scorer = QualityScorer()
        result = scorer.score("هذا نص عربي جيد للتدريب.")
        print(result.score)    # e.g. 0.87
        print(result.passed)   # True
        print(result.flag_codes)  # e.g. ["too_short"]

        # Filter a dataset
        clean = scorer.filter_batch(texts)
    """

    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        perplexity_scorer: Optional[Callable[[str], float]] = None,
    ):
        """
        Args:
            config: QualityConfig instance.
            perplexity_scorer: Optional callable(text) → float.
                Plug in a language-model-based perplexity scorer here.
                Lower perplexity = higher quality text.
                If provided, adds a perplexity dimension to the score.
        """
        self.config = config or QualityConfig()
        self.perplexity_scorer = perplexity_scorer

    # ------------------------------------------------------------------ #
    # Individual scorers (each returns 0.0 – 1.0, higher = better)
    # ------------------------------------------------------------------ #

    def _arabic_ratio_score(self, text: str) -> float:
        if not text:
            return 0.0
        arabic = len(_ARABIC_RE.findall(text))
        return arabic / len(text)

    def _length_score(self, text: str) -> float:
        n = len(text)
        cfg = self.config
        if n < cfg.min_length:
            return n / cfg.min_length
        if n > cfg.max_length:
            return cfg.max_length / n
        return 1.0

    def _repetition_score(self, text: str) -> float:
        """
        Detect repetitive text using word n-gram overlap.
        High repetition → low score.

        We use word trigrams when there are enough tokens, falling back to
        word bigrams for shorter texts (so an 8-word string that repeats the
        same phrase 4 times still gets flagged — previously, any text under
        10 words got a free pass).
        """
        words = text.split()
        if len(words) < 4:
            return 1.0

        k = 3 if len(words) >= 10 else 2
        ngrams = [tuple(words[i : i + k]) for i in range(len(words) - k + 1)]
        if not ngrams:
            return 1.0

        counter = Counter(ngrams)
        # Total over-count: how many ngram occurrences beyond the first.
        duplicate_count = sum(v - 1 for v in counter.values() if v > 1)
        repetition_ratio = duplicate_count / len(ngrams)

        if repetition_ratio > self.config.max_repetition_ratio:
            return max(0.0, 1.0 - repetition_ratio)
        return 1.0 - (repetition_ratio * 0.5)

    def _noise_score(self, text: str) -> float:
        """Penalizes excessive non-text characters."""
        if not text:
            return 0.0
        noise_chars = len(_NOISE_RE.findall(text))
        ratio = noise_chars / len(text)
        if ratio > self.config.max_noise_ratio:
            return max(0.0, 1.0 - ratio * 3)
        return 1.0 - ratio

    def _language_mix_score(self, text: str) -> float:
        """
        Score based on Arabic vs Latin character mix.
        Pure Arabic = 1.0. Heavy Latin = lower score.
        Code-switched text gets partial credit — it's not necessarily bad,
        just needs to be tracked.
        """
        if not text:
            return 0.0
        arabic = len(_ARABIC_RE.findall(text))
        latin = len(_LATIN_RE.findall(text))
        total = arabic + latin
        if total == 0:
            return 0.5
        return arabic / total

    def _is_toxic(self, text: str) -> bool:
        return bool(_TOXICITY_PATTERNS.search(text))

    def _is_boilerplate(self, text: str) -> bool:
        return any(p.search(text) for p in _BOILERPLATE_PATTERNS)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(self, text: str) -> QualityResult:
        """Score a single text sample."""
        cfg = self.config
        flags: list[str] = []
        flag_codes: list[str] = []

        arabic_ratio = self._arabic_ratio_score(text)
        length_score = self._length_score(text)
        repetition_score = self._repetition_score(text)
        noise_score = self._noise_score(text)
        language_mix_score = self._language_mix_score(text)
        is_toxic = cfg.check_toxicity and self._is_toxic(text)
        is_boilerplate = cfg.check_boilerplate and self._is_boilerplate(text)

        # Composite score
        w = cfg.weights
        composite = (
            w["arabic_ratio"] * arabic_ratio
            + w["length"] * length_score
            + w["repetition"] * repetition_score
            + w["noise"] * noise_score
            + w["language_mix"] * language_mix_score
        )

        # Hard penalties
        if is_toxic:
            composite *= 0.1
            flags.append("toxic_content")
            flag_codes.append(FLAG_TOXIC)
        if is_boilerplate:
            composite *= 0.6
            flags.append("boilerplate")
            flag_codes.append(FLAG_BOILERPLATE)
        if arabic_ratio < cfg.min_arabic_ratio:
            flags.append(f"low_arabic_ratio ({arabic_ratio:.2f})")
            flag_codes.append(FLAG_LOW_ARABIC)
        if len(text) < cfg.min_length:
            flags.append(f"too_short ({len(text)} chars)")
            flag_codes.append(FLAG_TOO_SHORT)
        if len(text) > cfg.max_length:
            flags.append(f"too_long ({len(text)} chars)")
            flag_codes.append(FLAG_TOO_LONG)
        if repetition_score < 0.5:
            flags.append("high_repetition")
            flag_codes.append(FLAG_HIGH_REPETITION)
        if noise_score < 0.5:
            flags.append("high_noise")
            flag_codes.append(FLAG_HIGH_NOISE)

        # Optional perplexity scoring
        if self.perplexity_scorer is not None:
            try:
                ppl = self.perplexity_scorer(text)
                # Normalize: ppl > 1000 = bad, < 100 = good
                ppl_score = max(0.0, 1.0 - (ppl / 1000))
                composite = composite * 0.8 + ppl_score * 0.2
            except Exception:
                pass

        composite = round(min(max(composite, 0.0), 1.0), 4)
        passed = composite >= cfg.min_quality_score
        if is_toxic and cfg.toxic_blocks_pass:
            passed = False
        if is_boilerplate and cfg.boilerplate_blocks_pass:
            passed = False

        return QualityResult(
            text=text,
            score=composite,
            passed=passed,
            arabic_ratio=round(arabic_ratio, 4),
            length_score=round(length_score, 4),
            repetition_score=round(repetition_score, 4),
            noise_score=round(noise_score, 4),
            language_mix_score=round(language_mix_score, 4),
            is_toxic=is_toxic,
            is_boilerplate=is_boilerplate,
            flags=flags,
            flag_codes=flag_codes,
        )

    def score_batch(self, texts: list[str]) -> list[QualityResult]:
        """Score a list of texts."""
        return [self.score(t) for t in texts]

    def filter_batch(self, texts: list[str]) -> list[str]:
        """Return only texts that pass the quality threshold."""
        return [t for t in texts if self.score(t).passed]

    def summary(self, texts: list[str]) -> dict:
        """Return quality statistics for a corpus."""
        results = self.score_batch(texts)
        scores = [r.score for r in results]
        passed = [r for r in results if r.passed]

        flag_counts: dict[str, int] = {}
        for r in results:
            for code in r.flag_codes:
                flag_counts[code] = flag_counts.get(code, 0) + 1

        return {
            "total": len(results),
            "passed": len(passed),
            "filtered": len(results) - len(passed),
            "pass_rate": round(len(passed) / max(len(results), 1) * 100, 1),
            "mean_score": round(sum(scores) / max(len(scores), 1), 4),
            "min_score": round(min(scores, default=0), 4),
            "max_score": round(max(scores, default=0), 4),
            "flag_breakdown": flag_counts,
        }

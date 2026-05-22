"""
Module 2 — Dialect Detection & Routing
========================================
Identifies which variety of Arabic a text is written in and routes it
to the appropriate bucket.

Strategy: IDF-weighted lexicon matching. Each marker word is weighted by
how *exclusively* it identifies one dialect — markers shared across many
dialects (e.g. ``بس``, ``يلا``) get downweighted, single-dialect markers
(``بتاع``, ``زول``, ``اكو``) carry most of the signal. Fast, interpretable,
and requires no model downloads.

Arabic dialect landscape:
  - MSA  (Modern Standard Arabic) — formal writing, news, official documents
  - EGY  (Egyptian)               — most widely understood dialect; media
  - LEV  (Levantine)              — Syria, Lebanon, Palestine, Jordan
  - GULF (Gulf / Khaleeji)        — Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, Oman
  - MAG  (Maghrebi)               — Morocco, Algeria, Tunisia, Libya
  - IRQ  (Iraqi)
  - SDN  (Sudanese)               — Sudan (split from previous SDN+YEM bucket)
  - YEM  (Yemeni)                 — Yemen
  - UNK  (Unknown / ambiguous)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DialectResult:
    dialect: str          # e.g. "EGY"
    dialect_name: str     # e.g. "Egyptian"
    confidence: float     # 0.0 – 1.0
    scores: dict[str, float]
    country_hints: list[str]  # ISO-3 codes likely associated with this dialect


# ---------------------------------------------------------------------------
# Dialect lexicons — marker words unique or strongly associated with each dialect.
# These are high-precision markers, not comprehensive dictionaries.
# Overlap between lexicons is OK — the IDF weighting handles it.
# ---------------------------------------------------------------------------

_LEXICONS: dict[str, list[str]] = {
    # ------------------------------------------------------------------ #
    # MSA — Modern Standard Arabic
    # ------------------------------------------------------------------ #
    "MSA": [
        # Classical interrogative particles & pronouns (high-precision MSA
        # markers — these almost never appear in colloquial dialects)
        "هل", "أن", "إن", "أريد", "أين", "هذه", "ماذا", "لماذا", "كيف",
        "متى", "إلى",
        # Relative & subordinating pronouns
        "الذي", "التي", "الذين", "اللواتي", "اللذان", "اللتان",
        "مما", "فيما", "عندما", "حيثما",
        # Connectives & discourse markers
        "حيث", "إذ", "إذا", "لكن", "بينما", "إلا أن",
        "على الرغم من", "بالإضافة إلى", "في حين", "من ثم",
        "وبالتالي", "وعليه", "ومن هنا", "في المقابل",
        "فضلاً عن", "علاوة على", "تجدر الإشارة",
        # Attribution verbs (journalistic)
        "وفقا", "وفقاً", "وفق", "أشار", "أوضح", "أكد",
        "أعلن", "صرح", "طالب", "أفاد", "كشف", "نوّه",
        "أضاف", "ذكر", "لفت", "نبّه", "حذّر", "استنكر",
        "أبدى", "أعرب", "أسف", "رحّب",
        # Formal/institutional vocabulary
        "وزارة", "حكومة", "البرلمان", "المجلس", "المملكة",
        "الجمهورية", "لجنة", "هيئة", "مؤسسة", "منظمة",
        "السلطة", "الدولة", "الرئاسة", "الأمانة",
        "الاتحاد", "التحالف", "المبادرة", "الاتفاقية", "البروتوكول",
        # Formal connectives & prepositions
        "استناداً", "في إطار", "في سياق", "في ضوء", "بموجب",
        "إزاء", "حيال", "تجاه", "للتصدي", "بشأن",
        "يتضمن", "يشمل", "ومن المقرر", "من المتوقع", "يُتوقع",
        # Passive & formal verb forms
        "صدر", "أُقرّ", "اعتُمد", "نُفّذ", "أُعلن", "جرى",
        "تمّ", "يُعقد", "انعقد", "استُؤنف", "أُرجئ",
    ],

    # ------------------------------------------------------------------ #
    # EGY — Egyptian Arabic
    # ------------------------------------------------------------------ #
    "EGY": [
        # Wanting / desire (uniquely Egyptian)
        "عايز", "عاوز", "عايزة", "عاوزة", "عوزين",
        # Possessive particle بتاع family — uniquely Egyptian
        "بتاع", "بتاعة", "بتوع", "بتاعتي", "بتاعتك", "بتاعته",
        # Question words (specific to EGY)
        "إيه", "ايه", "ليه", "امتى", "إمتى", "فين",
        "إزيك", "ازيك", "ازيكو", "إزيكم",
        # Negation patterns
        "مش عارف", "مش قادر", "مش هينفع", "مش كده",
        # Demonstratives (Egyptian forms)
        "ده", "دي", "دول", "دا", "ديه",
        # Manner / filler words
        "كده", "كدا", "كدة", "بقى", "بقا", "بقي",
        "اهو", "اهي", "اهوه", "زي ما",
        # Common verbs / expressions
        "بصيت", "اتفضل", "عدّى",
        "خلّى", "اشمعنى", "معملش", "مفيش", "فيه إيه",
        "بيعمل", "بتعمل", "هيعمل", "هتعمل",
        # Adjectives & intensifiers
        "اوي", "قوي",
        "جامد", "جامدة",
        # Everyday words
        "وانت", "وانتي", "وانتو", "النهارده", "إمبارح", "دلوقتي",
        "اللي", "ساعتها",
        "يا ربي", "يا عيني",
        "معلش", "يلا بينا", "عال العال",
    ],

    # ------------------------------------------------------------------ #
    # LEV — Levantine Arabic (Syria, Lebanon, Jordan, Palestine)
    # ------------------------------------------------------------------ #
    "LEV": [
        # Core question words
        "شو", "شو بدك", "شو بدي", "شو صار",
        # Travel-domain b-imperfect modals (uniquely Levantine — Beirut)
        "بتريد", "فيك", "فيي", "إزا", "عمول", "معروف",
        # Demonstratives & place (Levantine)
        "هيك", "هيكي", "هون", "هونيك", "هوني", "هيدا", "هيدي",
        # Time markers (Levantine)
        "هلق", "هلأ", "هلأة", "هلّق", "هلقيت",
        # Want / desire (بدّ family — uniquely Levantine)
        "بدي", "بده", "بدها", "بدنا", "بدكم", "بدهم", "بدك",
        # Progressive marker عم (uniquely Levantine, when used as prefix)
        "عمبحكي", "عم بشوف", "عم يحكي", "عم تحكي",
        "عم بدرس", "عم يشتغل",
        # Future marker رح
        "رح روح", "رح شوف", "رح يجي", "رح نعمل",
        # Negation
        "ما بعرفش", "ما شفتش",
        # Subordinating conjunctions
        "لأنو", "لأنها", "لأنهم", "متل", "متل ما",
        # Common Levantine adjectives & nouns
        "منيح", "منيحة", "كتير", "كتيرين",
        "يا زلمة", "زلمة", "ستي", "عمّو",
        # Expressions
        "اشي", "حكي", "بحكي",
        "مزبوط",
        # Verb conjugations typical of Levantine (b-imperfect)
        "بروح", "بشوف", "بعرف", "بحب", "بكتب", "بقرأ",
    ],

    # ------------------------------------------------------------------ #
    # GULF — Gulf / Khaleeji Arabic (Saudi, UAE, Kuwait, Qatar, Bahrain, Oman)
    # ------------------------------------------------------------------ #
    "GULF": [
        # Question words
        "وش", "ايش تبي", "ايش تقول",
        "ليش", "وليش", "شلون", "اشلون",
        # Desire / want (يبغى family — uniquely Gulf)
        "يبغى", "تبغى", "أبغى", "ابغى", "ابغي", "بغيت", "يبي", "ابي",
        "تبين", "يبون", "ما ابغى",
        # Gulf possessive particle حق (also Yemeni — IDF handles overlap)
        "حق",
        # Demonstratives (Gulf-specific)
        "جذي", "جدي", "جذا", "ذيك", "هذاك", "هذي", "هني",
        # Discourse & filler (Gulf-specific)
        "ترا", "تراه",
        "زين", "زينه", "زينة",
        "صج", "كذا",
        # Negation
        "موب", "مب", "مب زين", "مافي",
        # Wish / blessing forms
        "عساك", "عساكم", "عسى",
        "يزاك الله خير", "الله يوفقك",
        # Kinship / address
        "يهال", "يهل", "عيل", "خوي", "اخوي",
        "دشداشة", "غترة",
        # Verbs
        "ودّيت", "ودّ", "يودّي",
        # Intensifiers & time particles (Gulf-specific)
        "الحين", "توّه", "من توّه",
        # Unique Khaleeji vocabulary
        "خلني", "خله",
    ],

    # ------------------------------------------------------------------ #
    # MAG — Moroccan / Maghrebi Arabic (Darija)
    # ------------------------------------------------------------------ #
    "MAG": [
        # Moroccan polite forms & travel markers (Rabat)
        "عافاك", "واخا", "غادي", "ليا", "ليك",
        # Tunisian polite forms & modals (uniquely Tunisian — all collapse to MAG)
        "يعيشك", "عيشك", "نحب", "تنجم", "نجم", "متاع", "فما", "بش",
        "قداش", "هاذي",
        # Question / state words
        "واش", "واشني", "كيفاش", "كيفاش داير", "علاش", "فاش",
        "منين", "وقتاش", "شكون", "اشنو",
        # Existence / state
        "كاين", "كاينة", "كاينين", "مكاينش", "ما فيهش",
        # Progressive / continuous (Maghrebi-specific)
        "راك", "راني", "راهو", "راهي", "راهم", "رانا",
        "داير", "دايرة", "دايرين",
        # Intensifier / quantifier
        "بزاف", "شوية", "قليل ديال",
        # Discourse & filler (Maghrebi-specific)
        "دابا", "دابا دابا", "هاك",
        "هاد", "هادي", "هادو", "هاداك", "هاداكي",
        "كيما", "بحال", "بحال بحال",
        # Pronouns (Maghrebi-specific)
        "نتا", "نتي", "نتوما", "هوما",
        # Prepositions & connectives
        "باش", "بلا", "من غير",
        "ديال", "ديالي", "ديالك", "ديالو", "ديالنا",
        # Common verbs
        "سيري", "سيرو", "اجي", "اجيو",
        "ما بغاش",
        # Address forms & everyday expressions
        "خويا", "وليدي", "أ خويا",
        "بسلامة",
        "فالحقيقة", "بصح", "والو", "مزيان", "مزيانة",
        # French-origin words common in Darija
        "بيصا", "كاميو", "طوموبيل",
    ],

    # ------------------------------------------------------------------ #
    # IRQ — Iraqi Arabic
    # ------------------------------------------------------------------ #
    "IRQ": [
        # Existence (uniquely Iraqi)
        "اكو", "أكو", "ماكو", "مو اكو",
        # Question words (Iraqi-specific)
        "شكو", "شنو", "شنهو", "شبيك", "شبيها",
        "منو", "منهو", "وين", "وينك",
        "شكد", "شقد", "شقدر",
        # Intensifier (Iraqi)
        "هواية", "هواي", "كلش", "كلش زين", "مو كلش",
        # Address & kinship
        "يمعود", "يبه", "يمه",
        "يا حجي",
        # Verb stems with گ (unique Iraqi feature)
        "گلت", "گال", "گالوا", "يگول", "تگول",
        "گعد", "يگعد", "گاعد",
        # چ sound words
        "چا", "چاي", "چيكن", "چفت",
        # Greeting / politeness
        "شلونك", "شلونج", "شلونكم", "هلا وغلا",
        # Everyday vocabulary
        "يابه", "حسافة",
        "عدل", "مو عدل",
        "دكة", "باجة", "تشريب",
        # Time markers (Iraqi-specific)
        "هسه", "هسة", "هسعة", "باچر", "لهسه",
        # Pronouns (Iraqi)
        "آني", "احنه",
    ],

    # ------------------------------------------------------------------ #
    # SDN — Sudanese Arabic
    # ------------------------------------------------------------------ #
    "SDN": [
        # Uniquely Sudanese
        "زول", "أزول", "ناس زول", "زول كويس", "يا زول",
        "زاتو", "زاتها", "زاتهم",
        "ياخ", "ياخوي",
        # Sudanese food / culture words (high precision)
        "عصيدة", "كسرة", "مريسة", "ساي",
        # Sudanese b-imperfect verbs (shared w/ LEV but Sudanese context differs)
        "بقول", "بكلم", "بمشي",
        # Sudanese demonstrative / time
        "دلو", "دحين",
        # Demonstratives shared with EGY but heavily used in Sudanese too
        # (IDF downweights them so they don't override truly distinctive markers)
        "دا", "دي", "دول",
        # Question words shared with IRQ / others (IDF handles overlap)
        "شنو", "منو", "وين", "كيف الحال",
        # Sudanese discourse
        "عشان كده", "طيب كده", "طيب وبعدين",
        # Sudanese politeness
        "الله يحفظك", "يعافيك",
    ],

    # ------------------------------------------------------------------ #
    # YEM — Yemeni Arabic (split from the prior SDN+YEM bucket)
    # ------------------------------------------------------------------ #
    "YEM": [
        # Uniquely Yemeni want verbs (بشتي family)
        "بشتي", "أبشتي", "اشتي", "تشتي", "يشتي",
        # Yemeni question / negation
        "وش تبي", "فيش", "ما فيش",
        # Yemeni possessive حق (used as separate possessive particle)
        "حقي", "حقك", "حقه", "حقها", "حقنا", "حقهم",
        # Yemeni demonstratives
        "كذاك", "هكذاك", "ذاك", "ذيا",
        # Yemeni address
        "يا خوي", "يا أخي العزيز",
        # Yemeni-specific
        "قاعد أشتي", "قاعد يشتي",
        "تواً", "تو",
        # Yemeni connectives / fillers
        "عاد", "عادي عاد",
    ],
}

# Map dialects to likely countries (ISO-3)
_DIALECT_COUNTRIES: dict[str, list[str]] = {
    "MSA":  ["EGY", "SAU", "ARE", "MAR", "DZA", "TUN", "LBY", "SDN",
             "SYR", "LBN", "JOR", "IRQ", "KWT", "QAT", "BHR", "OMN", "YEM"],
    "EGY":  ["EGY"],
    "LEV":  ["SYR", "LBN", "JOR", "PSE"],
    "GULF": ["SAU", "ARE", "KWT", "QAT", "BHR", "OMN"],
    "MAG":  ["MAR", "DZA", "TUN", "LBY"],
    "IRQ":  ["IRQ"],
    "SDN":  ["SDN"],
    "YEM":  ["YEM"],
    "UNK":  [],
}

_DIALECT_NAMES: dict[str, str] = {
    "MSA":  "Modern Standard Arabic",
    "EGY":  "Egyptian",
    "LEV":  "Levantine",
    "GULF": "Gulf / Khaleeji",
    "MAG":  "Moroccan / Maghrebi",
    "IRQ":  "Iraqi",
    "SDN":  "Sudanese",
    "YEM":  "Yemeni",
    "UNK":  "Unknown",
}


# ---------------------------------------------------------------------------
# Pre-normalization of input text — applied inside detect() so that markers
# match correctly even on raw text containing diacritics or tatweel.
# ---------------------------------------------------------------------------
_PRE_NORM_RE = re.compile(
    r"[ً-ٟ"           # diacritics
    r"ٰ"                   # superscript alef
    r"ؐ-ؚ"            # Arabic sign sallallahou..small waw
    r"ۖ-ۭ"            # Quranic annotation signs
    r"ـ"                   # tatweel
    r"​-‏"            # ZWSP..RLM
    r"‪-‮"            # bidi embedding/override
    r"⁦-⁩"            # bidi isolates
    r"؜]"                  # Arabic letter mark
)
_WS_RE = re.compile(r"\s+")


def _pre_normalize(text: str) -> str:
    """Strip diacritics/tatweel/bidi marks; collapse whitespace. Idempotent."""
    text = _PRE_NORM_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# IDF weighting
# ---------------------------------------------------------------------------
# Markers shared across multiple dialects are downweighted; uniquely-identifying
# markers (e.g. "بتاع" → EGY only, "زول" → SDN only) carry most of the signal.
# Without this, common verbs in the MSA list like "قال" would dominate any text
# that happens to contain quoted speech.

def _build_idf_weights() -> dict[str, float]:
    df: dict[str, set[str]] = {}
    for dialect, words in _LEXICONS.items():
        for w in set(words):  # dedupe within a lexicon
            df.setdefault(w, set()).add(dialect)
    n = len(_LEXICONS)
    # Smoothed IDF: log(N / df) + 1.0 — adds a floor so pan-dialect markers
    # still contribute a little (otherwise their weight would be 0).
    return {w: math.log(n / len(dialects)) + 1.0 for w, dialects in df.items()}


_IDF: dict[str, float] = _build_idf_weights()


# ---------------------------------------------------------------------------
# Pattern compilation
# ---------------------------------------------------------------------------
# We compile *one regex per dialect-entry pair* rather than one giant alternation,
# because we need to know which entry matched in order to apply its IDF weight.
# Each pattern matches the entry surrounded by non-Arabic-word boundaries.
# For single-token entries that don't already begin with the definite article
# "ال", we generate two variants (with and without leading "ال") so that
# "حكومة" in the lexicon also matches "الحكومة" in real text.

_BOUNDARY_BEFORE = r"(?<![؀-ۿݐ-ݿࢠ-ࣿ])"
_BOUNDARY_AFTER = r"(?![؀-ۿݐ-ݿࢠ-ࣿ])"


def _compile_entry(entry: str) -> re.Pattern:
    # Multi-word entries: match as-is (allowing flexible whitespace).
    parts = entry.split()
    if len(parts) > 1:
        body = r"\s+".join(re.escape(p) for p in parts)
        return re.compile(_BOUNDARY_BEFORE + body + _BOUNDARY_AFTER, re.UNICODE)
    # Single-token entry: optionally match a leading "ال" — but only if the
    # entry itself does not already start with "ال" (avoids "(?:ال)?الذي" which
    # would otherwise also try to match the nonsensical "الالذي").
    if entry.startswith("ال"):
        body = re.escape(entry)
    else:
        body = r"(?:ال)?" + re.escape(entry)
    return re.compile(_BOUNDARY_BEFORE + body + _BOUNDARY_AFTER, re.UNICODE)


_DIALECT_ENTRIES: dict[str, list[tuple[str, re.Pattern]]] = {
    dialect: [(entry, _compile_entry(entry)) for entry in set(words)]
    for dialect, words in _LEXICONS.items()
}


_PROFILE_DIR = Path(__file__).resolve().parent / "data"
_PROFILE_CACHE: dict[str, dict] = {}


def _load_profile(name: str) -> dict:
    """Load a learned dialect-detection profile (cached)."""
    if name in _PROFILE_CACHE:
        return _PROFILE_CACHE[name]
    path = _PROFILE_DIR / f"dialect_profile_{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No dialect profile named {name!r} found at {path}. "
            f"Available profiles: {[p.stem.replace('dialect_profile_', '') for p in _PROFILE_DIR.glob('dialect_profile_*.json')] if _PROFILE_DIR.exists() else []}"
        )
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    # Pre-compile patterns for this profile's markers (cached on the profile dict).
    profile["_patterns"] = [_compile_entry(m) for m in profile["markers"]]
    _PROFILE_CACHE[name] = profile
    return profile


class DialectDetector:
    """
    Detects the Arabic dialect of a text sample.

    Two operating modes:

      1. **Default (lexicon + IDF)** — no profile required. Pure rule-based
         scoring using qalam's curated regional lexicons weighted by inverse
         document frequency. Good general-purpose recall on web/news/social
         Arabic. ~50-60% on MADAR-6 (travel-domain is out-of-distribution).

      2. **Profile mode** — `DialectDetector(profile="madar")` — loads a
         learned per-(marker, dialect) weight matrix trained on MADAR-6 train
         and applies it via pure-Python linear scoring. No sklearn or numpy
         needed at inference; the JSON profile (~100KB) ships with qalam.
         ~79% accuracy on MADAR-6 dev — competitive with MARBERT-DialectID.

    Usage:
        detector = DialectDetector()                    # default IDF mode
        detector = DialectDetector(profile="madar")     # learned weights

        result = detector.detect("عايز أروح السينما")
        print(result.dialect)       # "EGY"
        print(result.confidence)
    """

    def __init__(
        self,
        min_tokens: int = 3,
        min_signal: float = 1.5,
        min_distinct_markers: int = 2,
        profile: Optional[str] = None,
    ):
        """
        Args:
            min_tokens: Minimum word count before attempting detection.
                        Shorter texts return UNK.
            min_signal: Minimum IDF-weighted score required to commit to a label
                        (default mode only — unused in profile mode).
            min_distinct_markers: Require at least this many *distinct* marker
                        words to commit to a non-MSA label. Prevents a single
                        repeated keyword from producing a confident result.
                        (Default mode only.)
            profile: Optional name of a learned weight profile shipped with
                        qalam (e.g. "madar"). When set, switches to linear
                        scoring over learned weights instead of IDF heuristic.
        """
        self.min_tokens = min_tokens
        self.min_signal = min_signal
        self.min_distinct_markers = min_distinct_markers
        self.profile_name = profile
        self.profile = _load_profile(profile) if profile else None

    def detect(self, text: str) -> DialectResult:
        """Detect dialect of a single text."""
        # Pre-normalize: strip diacritics/tatweel/bidi so raw input matches
        # the lexicon entries (which are stored in their bare form).
        normalized = _pre_normalize(text)
        tokens = normalized.split()
        if len(tokens) < self.min_tokens:
            return self._unk(tokens)

        if self.profile is not None:
            return self._detect_profile(normalized, tokens)

        scores: dict[str, float] = {}
        distinct_markers: dict[str, int] = {}
        max_marker_idf: dict[str, float] = {}

        for dialect, entries in _DIALECT_ENTRIES.items():
            total_weight = 0.0
            n_distinct = 0
            max_idf = 0.0
            for entry, pattern in entries:
                hits = len(pattern.findall(normalized))
                if hits:
                    w = _IDF[entry]
                    total_weight += hits * w
                    n_distinct += 1
                    if w > max_idf:
                        max_idf = w
            # Normalize by sqrt(token count) — full length-normalization
            # over-penalizes long texts and under-penalizes short ones.
            scores[dialect] = total_weight / math.sqrt(len(tokens))
            distinct_markers[dialect] = n_distinct
            max_marker_idf[dialect] = max_idf

        # Pick the best-scoring dialect.
        best = max(scores, key=scores.get)
        best_score = scores[best]

        # No useful signal at all → UNK.
        if best_score < self.min_signal:
            return self._unk(tokens, scores)

        # Require multiple distinct markers for non-MSA labels, UNLESS the
        # single marker that did fire is uniquely identifying for that dialect
        # (e.g. "زول" → SDN, "اكو" → IRQ, "بتاع" → EGY). Such markers have an
        # IDF of log(N)+1.0 (the maximum possible).
        if best != "MSA" and distinct_markers[best] < self.min_distinct_markers:
            unique_marker_idf = math.log(len(_LEXICONS)) + 1.0
            has_unique = max_marker_idf[best] >= unique_marker_idf - 1e-9
            if not has_unique:
                # Fall back to MSA if it had a meaningful signal of its own,
                # otherwise UNK.
                if (
                    scores.get("MSA", 0.0) >= self.min_signal
                    and distinct_markers["MSA"] >= 1
                ):
                    best = "MSA"
                    best_score = scores["MSA"]
                else:
                    return self._unk(tokens, scores)

        # Confidence: ratio of best score to (best + runner-up). High when the
        # winner clearly dominates; near 0.5 when two dialects tie.
        sorted_scores = sorted(scores.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        if best_score + runner_up == 0:
            confidence = 0.0
        else:
            confidence = best_score / (best_score + runner_up)

        return DialectResult(
            dialect=best,
            dialect_name=_DIALECT_NAMES[best],
            confidence=round(confidence, 3),
            scores={k: round(v, 4) for k, v in scores.items()},
            country_hints=_DIALECT_COUNTRIES.get(best, []),
        )

    def _detect_profile(self, normalized: str, tokens: list[str]) -> DialectResult:
        """
        Linear-scoring inference using a learned weight profile.
        Pure Python — no sklearn / numpy dependency at runtime.
        """
        profile = self.profile
        classes: list[str] = profile["classes"]
        markers: list[str] = profile["markers"]
        patterns: list = profile["_patterns"]
        coef: list[list[float]] = profile["coef"]            # [C][F]
        intercept: list[float] = profile["intercept"]         # [C]
        clip = profile.get("feature_clip", 3)

        # Compute feature vector inline (sparse loop instead of materializing).
        # For each marker that fires, accumulate its weight into each class score.
        # Score[c] = intercept[c] + sum over fired markers j: coef[c][j] * count(j)
        scores_list = list(intercept)  # copy
        n_fired = 0
        for j, pat in enumerate(patterns):
            hits = len(pat.findall(normalized))
            if not hits:
                continue
            if hits > clip:
                hits = clip
            n_fired += 1
            for c in range(len(classes)):
                scores_list[c] += coef[c][j] * hits

        # Softmax-style confidence (without exp overflow)
        max_score = max(scores_list)
        exps = [math.exp(s - max_score) for s in scores_list]
        total = sum(exps)
        probs = [e / total for e in exps]
        best_idx = max(range(len(classes)), key=lambda i: probs[i])
        best = classes[best_idx]

        # When zero markers fired, the prediction is purely the prior (intercept).
        # We surface this through a low confidence value rather than abstaining,
        # to match the multinomial-logistic-regression baseline behavior. Callers
        # who want a hard abstention threshold can check `result.confidence`.
        scores_dict = {c: round(scores_list[i], 4) for i, c in enumerate(classes)}
        return DialectResult(
            dialect=best,
            dialect_name=_DIALECT_NAMES.get(best, "Unknown"),
            confidence=round(probs[best_idx], 3),
            scores=scores_dict,
            country_hints=_DIALECT_COUNTRIES.get(best, []),
        )

    def _unk(
        self,
        tokens: list[str],
        scores: Optional[dict[str, float]] = None,
    ) -> DialectResult:
        return DialectResult(
            dialect="UNK",
            dialect_name=_DIALECT_NAMES["UNK"],
            confidence=0.0,
            scores=(
                {k: round(v, 4) for k, v in scores.items()}
                if scores is not None
                else {d: 0.0 for d in _LEXICONS}
            ),
            country_hints=[],
        )

    def detect_batch(self, texts: list[str]) -> list[DialectResult]:
        """Detect dialects for a list of texts."""
        return [self.detect(t) for t in texts]

    def tag_batch(self, texts: list[str]) -> list[dict]:
        """Return texts with dialect metadata attached."""
        results = []
        for text in texts:
            r = self.detect(text)
            results.append({
                "text": text,
                "dialect": r.dialect,
                "dialect_name": r.dialect_name,
                "confidence": r.confidence,
                "country_hints": r.country_hints,
            })
        return results

    def route(
        self,
        texts: list[str],
        keep: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        tag: bool = False,
    ) -> list[str] | list[dict]:
        """
        Route texts by dialect.

        Args:
            texts:   Input texts.
            keep:    Dialect codes to KEEP (e.g. ["MSA", "EGY"]).
                     If None, keeps all.
            exclude: Dialect codes to EXCLUDE.
            tag:     If True, return dicts with metadata instead of raw strings.
        """
        results: list = []
        for text in texts:
            r = self.detect(text)
            if keep and r.dialect not in keep:
                continue
            if exclude and r.dialect in exclude:
                continue
            if tag:
                results.append({
                    "text": text,
                    "dialect": r.dialect,
                    "dialect_name": r.dialect_name,
                    "confidence": r.confidence,
                })
            else:
                results.append(text)
        return results

    def distribution(self, texts: list[str]) -> dict[str, dict]:
        """
        Compute dialect distribution statistics for a corpus.
        Returns per-dialect counts, percentages, and country breakdown.
        """
        results = self.detect_batch(texts)
        counts: dict[str, int] = {d: 0 for d in _DIALECT_NAMES}
        country_counts: dict[str, int] = {}

        for r in results:
            counts[r.dialect] = counts.get(r.dialect, 0) + 1
            for country in r.country_hints:
                country_counts[country] = country_counts.get(country, 0) + 1

        total = max(len(texts), 1)
        distribution = {}
        for dialect, count in counts.items():
            if count == 0:
                continue
            distribution[dialect] = {
                "count": count,
                "percentage": round(count / total * 100, 1),
                "dialect_name": _DIALECT_NAMES[dialect],
                "country_hints": _DIALECT_COUNTRIES.get(dialect, []),
            }

        # Dominant dialect: max by count, but if no text was labelled at all,
        # return UNK rather than whichever key happened to be first.
        if any(counts.values()):
            dominant = max(counts, key=counts.get)
        else:
            dominant = "UNK"

        return {
            "total": total,
            "dialects": distribution,
            "country_counts": country_counts,
            "dominant_dialect": dominant,
        }

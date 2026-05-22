"""
Module 4 — Deduplication
==========================
Removes duplicate and near-duplicate Arabic text samples from datasets.

Two strategies:
  1. Exact dedup  — MD5 hash of normalized text.
                    Catches perfect duplicates and near-identical texts
                    that differ only in whitespace/punctuation/diacritics.

  2. Near dedup   — MinHash + Locality Sensitive Hashing (LSH).
                    Catches paraphrases, slightly edited copies,
                    and crawl-introduced duplicates.
                    Does NOT require any model — pure string similarity.

Cross-dataset dedup is supported: deduplicate one dataset against another
(e.g. remove training examples that appear in a test set).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# MinHash implementation (pure Python, no datasketch required)
# ---------------------------------------------------------------------------

import random


# 64-bit non-cryptographic hash. Replaces the previous 32-bit MD5 mask, which
# hit birthday-paradox collisions at ~65K shingles and silently inflated
# Jaccard estimates on large documents.
def _hash64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


class _MinHasher:
    """
    Lightweight MinHash implementation using universal hash functions.
    Estimates Jaccard similarity between sets of shingles.
    """

    _MERSENNE_PRIME = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 42):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self._a = [rng.randint(1, self._MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, self._MERSENNE_PRIME - 1) for _ in range(num_perm)]

    def signature(self, shingles: set[int]) -> list[int]:
        # Initial sentinel is the prime itself — any real hash value is strictly
        # less than _prime, so the first shingle always replaces the sentinel.
        # The old code initialized to (1<<32)-1, which is *smaller* than typical
        # 61-bit hashes — the sentinel never got replaced for some permutations.
        prime = self._MERSENNE_PRIME
        sig = [prime] * self.num_perm
        a = self._a
        b = self._b
        for shingle in shingles:
            for i in range(self.num_perm):
                h = (a[i] * shingle + b[i]) % prime
                if h < sig[i]:
                    sig[i] = h
        return sig

    @staticmethod
    def jaccard(sig1: list[int], sig2: list[int]) -> float:
        matches = sum(a == b for a, b in zip(sig1, sig2))
        return matches / len(sig1)


def _shingles(
    text: str,
    k: int = 5,
    mode: Literal["char", "word"] = "char",
) -> set[int]:
    """
    k-shingles of a text, hashed to 64-bit ints.

    mode="char" — character k-shingles (good for short texts, robust to typos).
    mode="word" — word k-shingles (usually better for Arabic at corpus scale,
                  since char shingles after diacritic stripping conflate
                  morphologically distinct words).
    """
    text = re.sub(r"\s+", " ", text.strip())
    if mode == "word":
        words = text.split()
        if len(words) < k:
            return set()
        return {_hash64(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)}
    if len(text) < k:
        return set()
    return {_hash64(text[i : i + k]) for i in range(len(text) - k + 1)}


class _LSHIndex:
    """
    Band-based LSH for approximate nearest-neighbor lookup using MinHash.
    Splits the signature into `bands` bands of `rows` rows each.
    Two signatures that are identical in at least one band become candidates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.7):
        # Choose bands/rows to approximate the threshold
        # Using the standard formula: threshold ≈ (1/bands)^(1/rows)
        self.num_perm = num_perm
        self.threshold = threshold
        self.bands, self.rows = self._optimal_params(num_perm, threshold)
        self._buckets: dict[tuple, list[int]] = {}

    @staticmethod
    def _optimal_params(n: int, t: float) -> tuple[int, int]:
        best = (1, n)
        best_err = float("inf")
        for b in range(1, n + 1):
            if n % b != 0:
                continue
            r = n // b
            err = abs((1 / b) ** (1 / r) - t)
            if err < best_err:
                best_err = err
                best = (b, r)
        return best

    def add(self, idx: int, sig: list[int]) -> None:
        for band in range(self.bands):
            start = band * self.rows
            band_key = (band, tuple(sig[start : start + self.rows]))
            self._buckets.setdefault(band_key, []).append(idx)

    def query(self, sig: list[int]) -> set[int]:
        """Return candidate near-duplicate indices."""
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows
            band_key = (band, tuple(sig[start : start + self.rows]))
            candidates.update(self._buckets.get(band_key, []))
        return candidates


@dataclass
class DedupConfig:
    # Enable exact deduplication
    exact_dedup: bool = True

    # Enable near-deduplication
    near_dedup: bool = True

    # Jaccard similarity threshold for near-dedup (0–1)
    near_dedup_threshold: float = 0.8

    # Number of MinHash permutations (higher = more accurate, slower)
    num_perm: int = 128

    # Shingle size for text fingerprinting
    shingle_size: int = 5

    # "char" — character k-shingles (default; robust to typos, good for short texts)
    # "word" — word k-shingles (recommended for long-form Arabic at corpus scale)
    shingle_mode: Literal["char", "word"] = "char"

    # Normalize text before hashing (strip diacritics, whitespace)
    normalize_before_hash: bool = True

    # Length-ratio prefilter: skip Jaccard comparison when the shorter doc is
    # less than this fraction of the longer one. Disable with 0.0.
    length_ratio_threshold: float = 0.5

    # Minimum text length (chars) eligible for near-dedup. Shorter texts are
    # kept as-is — they don't produce reliable MinHash signatures.
    min_near_dedup_length: int = 10


@dataclass
class DedupResult:
    original_count: int
    unique_count: int
    removed_count: int
    exact_removed: int
    near_removed: int
    duplicate_rate: float
    unique_texts: list[str]


class Deduplicator:
    """
    Deduplicates Arabic text datasets using exact and near-duplicate detection.

    Usage:
        deduper = Deduplicator()
        result = deduper.dedup(texts)
        print(f"Removed {result.removed_count} duplicates")
        clean_texts = result.unique_texts

        # Cross-dataset: remove texts from dataset A that appear in dataset B
        clean_train = deduper.dedup_against(train_texts, test_texts)
    """

    # Strip-to-nothing: Arabic diacritics, tatweel, ZWNJ/ZWJ, bidi controls.
    # These never carry information for dedup purposes — removing them lets two
    # visually identical strings collapse to the same fingerprint.
    _STRIP_RE = re.compile(
        r"[ً-ٟ"   # fathatan..wavy hamza below
        r"ٰ"           # superscript alef
        r"ؐ-ؚ"    # Arabic sign sallallahou..small waw
        r"ۖ-ۭ"    # Quranic annotation signs
        r"ـ"           # tatweel
        r"​-‏"    # ZWSP, ZWNJ, ZWJ, LRM, RLM
        r"‪-‮"    # bidi embedding/override
        r"⁦-⁩"    # bidi isolates
        r"؜]+"         # Arabic letter mark
    )
    # Whitespace runs collapse to a single space (handled separately so that
    # stripping diacritics between letters does NOT insert spurious spaces).
    _WS_RE = re.compile(r"\s+")

    def __init__(self, config: Optional[DedupConfig] = None):
        self.config = config or DedupConfig()
        self._hasher = _MinHasher(num_perm=self.config.num_perm)

    def _normalize_for_hash(self, text: str) -> str:
        if not self.config.normalize_before_hash:
            return text
        text = self._STRIP_RE.sub("", text)
        text = self._WS_RE.sub(" ", text).strip()
        return text.lower()

    def _fingerprint(self, text: str) -> str:
        """Exact fingerprint (MD5 of normalized text)."""
        normalized = self._normalize_for_hash(text)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _sig_for(self, text: str) -> Optional[list[int]]:
        """
        Build a MinHash signature for `text`, or None if the text is too short
        / produces no shingles. Callers must treat None as "not comparable" —
        do NOT use a zero-filled placeholder (every empty-shingle text would
        then collide with every other).
        """
        cfg = self.config
        if len(text) < cfg.min_near_dedup_length:
            return None
        normalized = self._normalize_for_hash(text)
        shingles = _shingles(normalized, k=cfg.shingle_size, mode=cfg.shingle_mode)
        if not shingles:
            return None
        return self._hasher.signature(shingles)

    @staticmethod
    def _length_ok(a: str, b: str, threshold: float) -> bool:
        if threshold <= 0:
            return True
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return False
        return min(la, lb) / max(la, lb) >= threshold

    def dedup(self, texts: list[str]) -> DedupResult:
        """
        Deduplicate a list of texts.
        Returns a DedupResult with unique texts and statistics.
        """
        cfg = self.config
        original_count = len(texts)
        exact_removed = 0
        near_removed = 0

        # --- Phase 1: Exact dedup ---
        seen_hashes: set[str] = set()
        after_exact: list[str] = []
        if cfg.exact_dedup:
            for text in texts:
                h = self._fingerprint(text)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    after_exact.append(text)
                else:
                    exact_removed += 1
        else:
            after_exact = list(texts)

        # --- Phase 2: Near dedup (MinHash LSH) ---
        unique: list[str] = []
        if cfg.near_dedup and len(after_exact) > 1:
            lsh = _LSHIndex(
                num_perm=cfg.num_perm,
                threshold=cfg.near_dedup_threshold,
            )
            sigs: list[Optional[list[int]]] = [self._sig_for(t) for t in after_exact]
            removed_indices: set[int] = set()

            for i, sig in enumerate(sigs):
                if sig is None:
                    # Too short / unshingled — keep, don't index, don't compare.
                    continue
                candidates = lsh.query(sig)
                is_dup = False
                for j in candidates:
                    if j in removed_indices:
                        continue
                    other_sig = sigs[j]
                    if other_sig is None:
                        continue
                    if not self._length_ok(
                        after_exact[i], after_exact[j], cfg.length_ratio_threshold
                    ):
                        continue
                    if _MinHasher.jaccard(sig, other_sig) >= cfg.near_dedup_threshold:
                        removed_indices.add(i)
                        near_removed += 1
                        is_dup = True
                        break
                if not is_dup:
                    lsh.add(i, sig)

            unique = [t for i, t in enumerate(after_exact) if i not in removed_indices]
        else:
            unique = after_exact

        removed_count = original_count - len(unique)
        return DedupResult(
            original_count=original_count,
            unique_count=len(unique),
            removed_count=removed_count,
            exact_removed=exact_removed,
            near_removed=near_removed,
            duplicate_rate=round(removed_count / max(original_count, 1) * 100, 2),
            unique_texts=unique,
        )

    def dedup_against(
        self,
        source: list[str],
        reference: list[str],
    ) -> list[str]:
        """
        Remove from `source` any texts that are near-duplicates of `reference`.
        Useful for train/test contamination detection.

        Args:
            source:    The dataset to clean (e.g. training set).
            reference: The reference dataset to check against (e.g. test set).

        Returns:
            Filtered source texts with cross-contamination removed.
        """
        cfg = self.config

        lsh = _LSHIndex(num_perm=cfg.num_perm, threshold=cfg.near_dedup_threshold)
        ref_sigs: list[Optional[list[int]]] = []
        for i, text in enumerate(reference):
            sig = self._sig_for(text)
            ref_sigs.append(sig)
            if sig is not None:
                lsh.add(i, sig)

        clean: list[str] = []
        for text in source:
            sig = self._sig_for(text)
            if sig is None:
                # Can't compare — pass through (don't risk false-positive contamination).
                clean.append(text)
                continue
            candidates = lsh.query(sig)
            is_contaminated = False
            for j in candidates:
                ref_sig = ref_sigs[j]
                if ref_sig is None:
                    continue
                if not self._length_ok(text, reference[j], cfg.length_ratio_threshold):
                    continue
                if _MinHasher.jaccard(sig, ref_sig) >= cfg.near_dedup_threshold:
                    is_contaminated = True
                    break
            if not is_contaminated:
                clean.append(text)

        return clean

    def contamination_report(
        self,
        train: list[str],
        test: list[str],
    ) -> dict:
        """
        Report on train/test contamination.
        Returns counts and percentage of contaminated training examples.
        """
        clean_train = self.dedup_against(train, test)
        contaminated = len(train) - len(clean_train)
        return {
            "train_size": len(train),
            "test_size": len(test),
            "contaminated_train_examples": contaminated,
            "contamination_rate": round(contaminated / max(len(train), 1) * 100, 2),
            "clean_train_size": len(clean_train),
        }

"""
Pipeline — Orchestrator
========================
Chains all modules together into a single configurable pipeline.
The pipeline processes Arabic data end-to-end:

  Raw text → Normalize → Quality Filter → Dedup → Format → Report

Each stage is optional and configurable via PipelineConfig.

Usage:
    from qalam import Pipeline

    # Simple one-liner
    result = Pipeline().run(texts)
    clean_texts = result.texts          # ← originals (kept), not normalized
    norm_texts  = result.normalized_texts  # ← normalized versions of kept texts

    # Full control
    from qalam.normalize import NormalizerConfig
    from qalam.quality import QualityConfig

    pipeline = Pipeline(PipelineConfig(
        normalizer=NormalizerConfig(diacritics="keep"),
        quality=QualityConfig(min_quality_score=0.6),
        dialect_filter=["MSA", "EGY"],
        run_dedup=True,
    ))
    result = pipeline.run(texts)
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

from .normalize import QalamNormalizer, NormalizerConfig
from .dialect import DialectDetector
from .quality import QualityScorer, QualityConfig
from .dedup import Deduplicator, DedupConfig
from .formats import FormatConverter
from .report import ReportGenerator, DatasetReport

logger = logging.getLogger("qalam")


@dataclass
class PipelineConfig:
    # --- Normalization ---
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)

    # --- Dialect filtering ---
    # List of dialect codes to KEEP. None = keep all.
    dialect_filter: Optional[list[str]] = None

    # Minimum dialect detection confidence (0–1). Below this, behavior is
    # governed by `dialect_keep_uncertain`.
    dialect_min_confidence: float = 0.4

    # When dialect detection confidence is below threshold, keep the text
    # (True, default) or drop it (False). Keeping uncertain texts is safer
    # for high-recall pipelines but can let mislabeled foreign text through.
    dialect_keep_uncertain: bool = True

    # --- Quality ---
    quality: QualityConfig = field(default_factory=QualityConfig)

    # --- Deduplication ---
    run_dedup: bool = True
    dedup: DedupConfig = field(default_factory=DedupConfig)

    # --- Output format (informational; the formatter is exposed on the
    # pipeline as `.formatter` if you need to apply a chat template).
    chat_template: str = "chatml"

    # --- Report ---
    generate_report: bool = True
    report_path: Optional[str] = None    # If set, saves HTML report here
    report_json_path: Optional[str] = None

    # --- Verbosity ---
    # When True, the qalam logger is configured to emit INFO-level messages
    # to stderr (unless a handler is already attached). Disable to silence
    # the pipeline when embedding qalam in a larger application.
    verbose: bool = True


@dataclass
class PipelineResult:
    """
    Output of a pipeline run.

    Attributes:
        texts:            The kept *original* input texts (not normalized).
                          This makes round-tripping back to source datasets safe.
        normalized_texts: The normalized versions of `texts`, in the same order.
                          Use these for training; use `texts` for traceability.
        kept_indices:     Indices into the original input list that survived.
        report:           Optional DatasetReport (None if generate_report=False).
        stats:            Per-stage drop counts and timing.
    """
    texts: list[str]
    normalized_texts: list[str]
    kept_indices: list[int]
    report: Optional[DatasetReport]
    stats: dict


@dataclass
class StreamItem:
    """One record yielded by Pipeline.run_stream."""
    index: int                          # Position in the original input
    original: str
    normalized: str
    kept: bool
    drop_reason: Optional[str] = None   # "empty", "dialect", "quality", "duplicate"
    dialect: Optional[str] = None
    quality_score: Optional[float] = None


class Pipeline:
    """
    End-to-end Arabic data quality pipeline.

    Example:
        from qalam import Pipeline

        pipeline = Pipeline()
        result = pipeline.run(my_texts)
        print(result.stats)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        cfg = self.config

        self.normalizer = QalamNormalizer(cfg.normalizer)
        self.detector = DialectDetector()
        self.scorer = QualityScorer(cfg.quality)
        self.deduper = Deduplicator(cfg.dedup)
        self.formatter = FormatConverter(cfg.chat_template)
        self.reporter = ReportGenerator(self.detector, self.scorer)

        if cfg.verbose:
            _ensure_handler(logger, logging.INFO)

    def run(self, texts: list[str]) -> PipelineResult:
        """
        Run the full pipeline on a list of Arabic texts.

        Args:
            texts: Raw Arabic text strings.

        Returns:
            PipelineResult with kept-original texts, normalized texts,
            kept indices, optional report, and stats.
        """
        cfg = self.config
        stats: dict = {"input_count": len(texts)}
        t0 = time.time()

        logger.info("Starting pipeline — %s examples", f"{len(texts):,}")

        # Track which original indices survive each stage. Maintaining this
        # mapping lets us return the user's *original* strings at the end
        # (the previous version returned normalized strings, losing traceability).
        live_indices: list[int] = list(range(len(texts)))

        # ── Stage 1: Normalize ───────────────────────────────────────────
        logger.info("Stage 1/4: Normalizing text...")
        normalized_all: list[str] = self.normalizer.normalize_batch(texts)
        # Drop indices whose normalized form is empty (often happens when
        # input was just HTML/whitespace/punctuation).
        kept = [i for i in live_indices if normalized_all[i].strip()]
        stats["empty_after_normalize"] = len(live_indices) - len(kept)
        stats["after_normalize"] = len(kept)
        logger.info("  %s empty after normalization", f"{stats['empty_after_normalize']:,}")
        live_indices = kept

        # ── Stage 2: Dialect filter ──────────────────────────────────────
        if cfg.dialect_filter:
            logger.info("Stage 2/4: Dialect filtering → keep %s...", cfg.dialect_filter)
            dialect_removed = 0
            kept = []
            for i in live_indices:
                result = self.detector.detect(normalized_all[i])
                if result.confidence < cfg.dialect_min_confidence:
                    if cfg.dialect_keep_uncertain:
                        kept.append(i)
                    else:
                        dialect_removed += 1
                elif result.dialect in cfg.dialect_filter:
                    kept.append(i)
                else:
                    dialect_removed += 1
            stats["dialect_removed"] = dialect_removed
            stats["after_dialect_filter"] = len(kept)
            logger.info("  Removed %s examples by dialect", f"{dialect_removed:,}")
            live_indices = kept
        else:
            logger.info("Stage 2/4: Dialect filtering — skipped (no filter set)")
            stats["dialect_removed"] = 0

        # ── Stage 3: Quality filter ──────────────────────────────────────
        logger.info("Stage 3/4: Quality scoring & filtering...")
        before = len(live_indices)
        kept = []
        for i in live_indices:
            if self.scorer.score(normalized_all[i]).passed:
                kept.append(i)
        quality_removed = before - len(kept)
        stats["quality_removed"] = quality_removed
        stats["after_quality"] = len(kept)
        stats["quality_pass_rate"] = round(
            len(kept) / max(before, 1) * 100, 1
        )
        logger.info(
            "  Removed %s low-quality examples (pass rate: %s%%)",
            f"{quality_removed:,}", stats["quality_pass_rate"],
        )
        live_indices = kept

        # ── Stage 4: Deduplication ───────────────────────────────────────
        if cfg.run_dedup and len(live_indices) > 1:
            logger.info("Stage 4/4: Deduplicating...")
            # Dedup on normalized text; map back to original indices.
            normalized_live = [normalized_all[i] for i in live_indices]
            dedup_result = self.deduper.dedup(normalized_live)
            # Reconstruct the kept indices by matching position in the dedup
            # output. Since Deduplicator.dedup preserves first-occurrence
            # order, we can re-walk the input to find which positions survived.
            kept_positions = _surviving_positions(normalized_live, dedup_result.unique_texts)
            live_indices = [live_indices[p] for p in kept_positions]

            stats["exact_dupes_removed"] = dedup_result.exact_removed
            stats["near_dupes_removed"] = dedup_result.near_removed
            stats["after_dedup"] = len(live_indices)
            stats["duplicate_rate"] = dedup_result.duplicate_rate
            logger.info(
                "  Removed %s duplicates (exact: %s, near: %s)",
                f"{dedup_result.removed_count:,}",
                dedup_result.exact_removed, dedup_result.near_removed,
            )
        else:
            logger.info("Stage 4/4: Deduplication — skipped")
            stats["after_dedup"] = len(live_indices)

        # ── Summary stats ────────────────────────────────────────────────
        elapsed = round(time.time() - t0, 2)
        final_texts = [texts[i] for i in live_indices]
        final_normalized = [normalized_all[i] for i in live_indices]
        stats["output_count"] = len(final_texts)
        stats["total_removed"] = len(texts) - len(final_texts)
        stats["retention_rate"] = round(
            len(final_texts) / max(len(texts), 1) * 100, 1
        )
        stats["elapsed_seconds"] = elapsed

        logger.info(
            "Done in %ss — %s / %s examples retained (%s%%)",
            elapsed, f"{len(final_texts):,}", f"{len(texts):,}",
            stats["retention_rate"],
        )

        # ── Generate report ───────────────────────────────────────────────
        report = None
        if cfg.generate_report and final_normalized:
            logger.info("Generating report card...")
            report = self.reporter.generate(final_normalized)
            if cfg.report_json_path:
                self.reporter.save_json(report, cfg.report_json_path)
                logger.info("  Saved JSON report → %s", cfg.report_json_path)
            if cfg.report_path:
                self.reporter.save_html(report, cfg.report_path)
                logger.info("  Saved HTML report → %s", cfg.report_path)
            if cfg.verbose:
                self.reporter.print_summary(report)

        return PipelineResult(
            texts=final_texts,
            normalized_texts=final_normalized,
            kept_indices=live_indices,
            report=report,
            stats=stats,
        )

    def run_stream(self, texts: Iterator[str]) -> Iterator[StreamItem]:
        """
        Streaming variant. Yields one StreamItem per input text, in input order.

        Memory: O(stream so far) for the dedup index. Set `run_dedup=False`
        in config for truly constant-memory streaming.

        Unlike `run()`, this does NOT generate a report (reports require the
        full corpus). Aggregate StreamItems yourself if you need stats.
        """
        cfg = self.config
        dedup_lsh = None
        dedup_sigs: list = []
        seen_exact: set[str] = set()
        if cfg.run_dedup:
            from .dedup import _LSHIndex
            dedup_lsh = _LSHIndex(
                num_perm=cfg.dedup.num_perm,
                threshold=cfg.dedup.near_dedup_threshold,
            )

        for idx, text in enumerate(texts):
            normalized = self.normalizer.normalize(text)
            if not normalized.strip():
                yield StreamItem(idx, text, normalized, False, "empty")
                continue

            dialect_code = None
            if cfg.dialect_filter:
                d = self.detector.detect(normalized)
                dialect_code = d.dialect
                if d.confidence < cfg.dialect_min_confidence:
                    if not cfg.dialect_keep_uncertain:
                        yield StreamItem(idx, text, normalized, False, "dialect", dialect_code)
                        continue
                elif d.dialect not in cfg.dialect_filter:
                    yield StreamItem(idx, text, normalized, False, "dialect", dialect_code)
                    continue

            q = self.scorer.score(normalized)
            if not q.passed:
                yield StreamItem(
                    idx, text, normalized, False, "quality",
                    dialect_code, q.score,
                )
                continue

            if cfg.run_dedup:
                fp = self.deduper._fingerprint(text)
                if fp in seen_exact:
                    yield StreamItem(
                        idx, text, normalized, False, "duplicate",
                        dialect_code, q.score,
                    )
                    continue
                seen_exact.add(fp)

                sig = self.deduper._sig_for(text)
                if sig is not None and dedup_lsh is not None:
                    candidates = dedup_lsh.query(sig)
                    is_dup = False
                    from .dedup import _MinHasher
                    for j in candidates:
                        if _MinHasher.jaccard(sig, dedup_sigs[j]) >= cfg.dedup.near_dedup_threshold:
                            is_dup = True
                            break
                    if is_dup:
                        yield StreamItem(
                            idx, text, normalized, False, "duplicate",
                            dialect_code, q.score,
                        )
                        continue
                    dedup_lsh.add(len(dedup_sigs), sig)
                    dedup_sigs.append(sig)

            yield StreamItem(
                idx, text, normalized, True, None,
                dialect_code, q.score,
            )

    def run_from_file(
        self,
        path: Union[str, Path],
        column: Optional[str] = None,
    ) -> PipelineResult:
        """
        Load texts from a JSONL / JSONL.GZ / NDJSON / TXT file and run the pipeline.

        Args:
            path:   Path to file. Recognized extensions: .jsonl, .jsonl.gz,
                    .ndjson, .ndjson.gz, .txt. Anything else is read as plain text
                    (one record per line). UTF-8 BOMs are tolerated.
            column: For JSON-line formats, the key containing the text field.
                    Defaults to "text". Rows missing this key are skipped (with
                    a logged warning).
        """
        path = Path(path)
        name = path.name.lower()
        is_jsonl = name.endswith((".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz"))
        is_gz = name.endswith(".gz")

        open_fn = (
            (lambda p: gzip.open(p, "rt", encoding="utf-8-sig"))
            if is_gz
            else (lambda p: open(p, "r", encoding="utf-8-sig"))
        )

        texts: list[str] = []
        skipped = 0
        key = column or "text"

        with open_fn(path) as f:
            if is_jsonl:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        logger.warning("Skipping malformed JSON at line %s", lineno)
                        continue
                    if isinstance(obj, str):
                        texts.append(obj)
                    elif isinstance(obj, dict) and key in obj:
                        texts.append(obj[key])
                    else:
                        skipped += 1
            else:
                texts = [line.strip() for line in f if line.strip()]

        if skipped:
            logger.warning(
                "Skipped %s rows (malformed or missing '%s' key)", skipped, key
            )

        return self.run(texts)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _surviving_positions(input_list: list[str], output_list: list[str]) -> list[int]:
    """
    Map each item in `output_list` back to its first matching position in
    `input_list`. Dedup preserves first-occurrence order, so this walk is
    deterministic.
    """
    positions: list[int] = []
    cursor = 0
    for out in output_list:
        while cursor < len(input_list) and input_list[cursor] != out:
            cursor += 1
        if cursor >= len(input_list):
            break
        positions.append(cursor)
        cursor += 1
    return positions


def _ensure_handler(lg: logging.Logger, level: int) -> None:
    """Attach a stderr handler to the logger if none is configured yet."""
    if lg.handlers:
        lg.setLevel(min(lg.level or level, level))
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(handler)
    lg.setLevel(level)
    lg.propagate = False

"""
qalam (qalam)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A pipeline for preparing high-quality Arabic datasets for LLM training and fine-tuning.

Modules:
    normalize   — Unicode & script normalization
    dialect     — Dialect detection & routing
    quality     — Quality scoring & filtering
    dedup       — Exact & near-deduplication
    formats     — Output format converters (SFT, DPO, chat templates)
    report      — Dataset statistics & report card
    pipeline    — Orchestrator
"""

from .normalize import QalamNormalizer
from .dialect import DialectDetector
from .quality import QualityScorer
from .dedup import Deduplicator
from .formats import FormatConverter
from .report import ReportGenerator
from .pipeline import Pipeline

__version__ = "0.1.0"
__all__ = [
    "QalamNormalizer",
    "DialectDetector",
    "QualityScorer",
    "Deduplicator",
    "FormatConverter",
    "ReportGenerator",
    "Pipeline",
]

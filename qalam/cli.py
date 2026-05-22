#!/usr/bin/env python3
"""
qalam — CLI for Arabic data quality pipeline

Usage:
    qalam clean data.jsonl --output clean.jsonl
    qalam clean data.jsonl --dialect msa --format sft --template llama3
    qalam report data.jsonl --out report.html
    qalam report data.jsonl --json report.json
    qalam dedup data.jsonl --threshold 0.8 --output deduped.jsonl
    qalam info
"""

import argparse
import json
import sys
from pathlib import Path


def _force_utf8_output() -> None:
    """
    Ensure stdout/stderr can render Arabic text and box-drawing characters.

    On Windows the console defaults to a legacy code page (cp1252), which
    raises UnicodeEncodeError the moment we print Arabic or the report's
    box-drawing separators. Python 3.7+ lets us reconfigure the streams to
    UTF-8 in place; this is a no-op on platforms that are already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Stream may be redirected to something non-reconfigurable;
                # printing will still work for ASCII in that case.
                pass


def cmd_clean(args):
    from qalam.pipeline import Pipeline, PipelineConfig
    from qalam.normalize import NormalizerConfig
    from qalam.quality import QualityConfig
    from qalam.dedup import DedupConfig

    dialect_filter = None
    if args.dialect:
        dialect_map = {
            "msa": "MSA", "egy": "EGY", "lev": "LEV",
            "gulf": "GULF", "mag": "MAG", "irq": "IRQ", "sdn": "SDN", "yem": "YEM",
        }
        dialect_filter = [dialect_map.get(d.lower(), d.upper()) for d in args.dialect.split(",")]

    cfg = PipelineConfig(
        normalizer=NormalizerConfig(
            diacritics=args.diacritics,
            normalize_alef=not args.keep_alef,
        ),
        quality=QualityConfig(
            min_quality_score=args.quality_threshold,
            min_length=args.min_length,
        ),
        dedup=DedupConfig(
            near_dedup_threshold=args.dedup_threshold,
            near_dedup=not args.exact_only,
        ),
        dialect_filter=dialect_filter,
        run_dedup=not args.no_dedup,
        chat_template=args.template,
        generate_report=True,
        report_path=args.report_html,
        report_json_path=args.report_json,
        verbose=not args.quiet,
    )

    pipeline = Pipeline(cfg)
    result = pipeline.run_from_file(args.input, column=args.column)

    if args.output:
        out_path = args.output
        if args.format == "pretraining":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(result.texts))
        else:
            # "raw" and "sft" both emit one JSON object per line. (SFT from raw
            # text has no instruction/output structure to add — kept as alias.)
            with open(out_path, "w", encoding="utf-8") as f:
                for text in result.texts:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(result.texts):,} examples → {out_path}")

    print("\nPipeline stats:")
    for k, v in result.stats.items():
        print(f"  {k:<28} {v}")


def cmd_report(args):
    from qalam.report import ReportGenerator

    texts = []
    input_path = args.input
    column = args.column or "text"

    if input_path.endswith(".jsonl"):
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if column in obj:
                    texts.append(obj[column])
    else:
        with open(input_path, encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]

    if not texts:
        print("No texts found in input file.", file=sys.stderr)
        sys.exit(1)

    gen = ReportGenerator()
    report = gen.generate(texts)
    gen.print_summary(report)

    if args.out:
        gen.save_html(report, args.out, dataset_name=Path(input_path).stem)
        print(f"HTML report saved → {args.out}")

    if getattr(args, "json", None):
        gen.save_json(report, args.json)
        print(f"JSON report saved → {args.json}")

    if args.markdown:
        md = gen.to_markdown(report, dataset_name=Path(input_path).stem)
        print("\n" + md)


def cmd_dedup(args):
    from qalam.dedup import Deduplicator, DedupConfig

    texts = []
    column = args.column or "text"
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.input.endswith(".jsonl"):
                obj = json.loads(line)
                texts.append(obj.get(column, ""))
            else:
                texts.append(line)

    cfg = DedupConfig(
        near_dedup_threshold=args.threshold,
        near_dedup=not args.exact_only,
    )
    deduper = Deduplicator(cfg)
    result = deduper.dedup(texts)

    print(f"Original : {result.original_count:,}")
    print(f"Unique   : {result.unique_count:,}")
    print(f"Removed  : {result.removed_count:,} ({result.duplicate_rate:.1f}%)")
    print(f"  Exact  : {result.exact_removed:,}")
    print(f"  Near   : {result.near_removed:,}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for text in result.unique_texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"\nSaved → {args.output}")


def cmd_info(args):
    import qalam
    from qalam.formats import CHAT_TEMPLATES

    print(f"\nqalam v{qalam.__version__}")
    print("\nModules:")
    print("  normalize  — Unicode normalization, alef/hamza unification, diacritics")
    print("  dialect    — Dialect detection (MSA/EGY/LEV/GULF/MAG/IRQ/SDN/YEM)")
    print("  quality    — Quality scoring & filtering")
    print("  dedup      — Exact + MinHash near-deduplication")
    print("  formats    — SFT, DPO, chat template converters")
    print("  report     — Dataset stats & HTML report card")
    print("\nAvailable chat templates:")
    for key, tmpl in CHAT_TEMPLATES.items():
        print(f"  {key:<12} — {tmpl['name']}")
    print("\nDialect codes:")
    dialects = {
        "MSA": "Modern Standard Arabic",
        "EGY": "Egyptian",
        "LEV": "Levantine (Syria/Lebanon/Jordan/Palestine)",
        "GULF": "Gulf / Khaleeji",
        "MAG": "Moroccan / Maghrebi",
        "IRQ": "Iraqi",
        "SDN": "Sudanese",
        "YEM": "Yemeni",
    }
    for code, name in dialects.items():
        print(f"  {code:<8} — {name}")


def main():
    _force_utf8_output()
    parser = argparse.ArgumentParser(
        prog="qalam",
        description="Qalam — Arabic data quality pipeline for LLM training",
    )
    sub = parser.add_subparsers(dest="command")

    # ── clean ──────────────────────────────────────────────────────────
    p_clean = sub.add_parser("clean", help="Run the full cleaning pipeline")
    p_clean.add_argument("input", help="Input file (.jsonl or .txt)")
    p_clean.add_argument("--output", "-o", help="Output JSONL path")
    p_clean.add_argument("--column", default="text", help="Text column name in JSONL")
    p_clean.add_argument("--dialect", help="Comma-separated dialects to keep (e.g. msa,egy)")
    p_clean.add_argument("--diacritics", choices=["strip", "keep", "keep_classical"], default="strip")
    p_clean.add_argument("--keep-alef", action="store_true", help="Don't unify alef variants")
    p_clean.add_argument("--quality-threshold", type=float, default=0.5, metavar="0-1")
    p_clean.add_argument("--min-length", type=int, default=50)
    p_clean.add_argument("--dedup-threshold", type=float, default=0.8, metavar="0-1")
    p_clean.add_argument("--exact-only", action="store_true", help="Only exact dedup (faster)")
    p_clean.add_argument("--no-dedup", action="store_true")
    p_clean.add_argument("--format", choices=["raw", "sft", "pretraining"], default="raw")
    p_clean.add_argument(
        "--template",
        choices=[
            "chatml", "llama3", "llama2", "mistral", "jais",
            "gemma4", "gemma2", "command-r", "phi3", "deepseek",
            "alpaca", "vicuna",
        ],
        default="chatml",
        help="Chat template for --format sft. See `qalam info` for the full list.",
    )
    p_clean.add_argument("--report-html", metavar="PATH", help="Save HTML report")
    p_clean.add_argument("--report-json", metavar="PATH", help="Save JSON report")
    p_clean.add_argument("--quiet", "-q", action="store_true")

    # ── report ─────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Generate dataset report card")
    p_report.add_argument("input", help="Input file (.jsonl or .txt)")
    p_report.add_argument("--out", help="Save interactive HTML report")
    p_report.add_argument("--json", help="Save JSON report")
    p_report.add_argument("--markdown", action="store_true", help="Print markdown to stdout")
    p_report.add_argument("--column", default="text")

    # ── dedup ──────────────────────────────────────────────────────────
    p_dedup = sub.add_parser("dedup", help="Deduplicate a dataset")
    p_dedup.add_argument("input", help="Input file (.jsonl or .txt)")
    p_dedup.add_argument("--output", "-o")
    p_dedup.add_argument("--threshold", type=float, default=0.8)
    p_dedup.add_argument("--exact-only", action="store_true")
    p_dedup.add_argument("--column", default="text")

    # ── info ───────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show module info and available options")

    args = parser.parse_args()

    if args.command == "clean":
        cmd_clean(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

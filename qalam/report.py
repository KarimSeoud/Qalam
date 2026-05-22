"""
Module 6 — Dataset Statistics & Report Card
=============================================
Generates comprehensive statistics and a visual report card for an
Arabic dataset. The report is suitable for inclusion in model cards,
research papers, and dataset documentation.

Output formats:
  - JSON  — machine-readable, embeddable in model cards
  - HTML  — the visual dialect map + charts for README / GitHub Pages
  - Markdown — for HuggingFace dataset cards
  - Text  — CLI-friendly summary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional

from .dialect import DialectDetector
from .quality import QualityScorer, QualityConfig


@dataclass
class DatasetReport:
    """Full report card for an Arabic dataset."""

    # Basic statistics
    total_examples: int
    total_chars: int
    total_words: int
    avg_length_chars: float
    avg_length_words: float
    median_length_chars: float

    # Quality stats
    quality_pass_rate: float
    mean_quality_score: float
    quality_distribution: dict   # {"0.0-0.2": N, "0.2-0.4": N, ...}

    # Dialect stats
    dialect_distribution: dict   # {"MSA": {"count": N, "pct": X}, ...}
    dominant_dialect: str
    dialect_diversity_score: float   # Shannon entropy of dialect distribution

    # Content stats
    arabic_ratio_mean: float
    unique_char_count: int
    estimated_tokens: int            # rough estimate at ~3.5 chars/token for Arabic

    # Dedup stats
    exact_duplicate_rate: float
    near_duplicate_rate: float

    # Flags
    flag_breakdown: dict

    # Country-level data (for the map visualizer)
    country_counts: dict


class ReportGenerator:
    """
    Generates a full report card for an Arabic dataset.

    Usage:
        gen = ReportGenerator()
        report = gen.generate(texts)

        # Save as JSON
        gen.save_json(report, "report.json")

        # Save as HTML (includes dialect map)
        gen.save_html(report, "report.html")

        # Print summary to terminal
        gen.print_summary(report)
    """

    def __init__(
        self,
        dialect_detector: Optional[DialectDetector] = None,
        quality_scorer: Optional[QualityScorer] = None,
    ):
        self.detector = dialect_detector or DialectDetector()
        self.scorer = quality_scorer or QualityScorer()

    def generate(self, texts: list[str]) -> DatasetReport:
        """Generate a full report card for the dataset."""
        if not texts:
            raise ValueError("Cannot generate report for empty dataset.")

        n = len(texts)

        # --- Basic stats ---
        char_lengths = [len(t) for t in texts]
        word_lengths = [len(t.split()) for t in texts]
        sorted_chars = sorted(char_lengths)
        median_chars = sorted_chars[n // 2]

        total_chars = sum(char_lengths)
        total_words = sum(word_lengths)

        # --- Quality scoring ---
        quality_results = self.scorer.score_batch(texts)
        scores = [r.score for r in quality_results]
        passed = sum(1 for r in quality_results if r.passed)
        flag_counts: dict[str, int] = {}
        for r in quality_results:
            for flag in r.flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        quality_dist = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for s in scores:
            if s < 0.2:
                quality_dist["0.0-0.2"] += 1
            elif s < 0.4:
                quality_dist["0.2-0.4"] += 1
            elif s < 0.6:
                quality_dist["0.4-0.6"] += 1
            elif s < 0.8:
                quality_dist["0.6-0.8"] += 1
            else:
                quality_dist["0.8-1.0"] += 1

        arabic_ratios = [r.arabic_ratio for r in quality_results]
        mean_arabic_ratio = sum(arabic_ratios) / max(n, 1)

        # --- Dialect distribution ---
        dialect_dist = self.detector.distribution(texts)

        # Shannon entropy for dialect diversity
        import math
        entropy = 0.0
        for info in dialect_dist["dialects"].values():
            p = info["percentage"] / 100
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(max(len(dialect_dist["dialects"]), 1))
        diversity_score = round(entropy / max_entropy, 3) if max_entropy > 0 else 0.0

        # --- Character stats ---
        all_chars = "".join(texts)
        unique_chars = len(set(all_chars))

        # Estimated token count (Arabic averages ~3.5 chars/token with common tokenizers)
        estimated_tokens = int(total_chars / 3.5)

        # --- Dedup stats (lightweight for report purposes) ---
        # Use Deduplicator._fingerprint so the reported exact-duplicate rate
        # exactly matches what `Pipeline` / `Deduplicator.dedup` actually find.
        # The previous version did raw MD5 of stripped text and disagreed with
        # the deduper for any input containing diacritics / tatweel / bidi marks.
        from .dedup import Deduplicator, DedupConfig
        deduper = Deduplicator(DedupConfig(near_dedup=False))  # exact only for speed
        fingerprints = {deduper._fingerprint(t) for t in texts}
        exact_dupes = n - len(fingerprints)
        exact_rate = round(exact_dupes / max(n, 1) * 100, 2)

        # Build per-dialect distribution suitable for report
        dialect_report = {}
        for dialect, info in dialect_dist["dialects"].items():
            dialect_report[dialect] = {
                "count": info["count"],
                "percentage": info["percentage"],
                "dialect_name": info["dialect_name"],
            }

        return DatasetReport(
            total_examples=n,
            total_chars=total_chars,
            total_words=total_words,
            avg_length_chars=round(total_chars / n, 1),
            avg_length_words=round(total_words / n, 1),
            median_length_chars=float(median_chars),
            quality_pass_rate=round(passed / n * 100, 1),
            mean_quality_score=round(sum(scores) / max(len(scores), 1), 3),
            quality_distribution=quality_dist,
            dialect_distribution=dialect_report,
            dominant_dialect=dialect_dist["dominant_dialect"],
            dialect_diversity_score=diversity_score,
            arabic_ratio_mean=round(mean_arabic_ratio, 3),
            unique_char_count=unique_chars,
            estimated_tokens=estimated_tokens,
            exact_duplicate_rate=exact_rate,
            near_duplicate_rate=0.0,  # requires near_dedup pass
            flag_breakdown=flag_counts,
            country_counts=dialect_dist["country_counts"],
        )

    # ------------------------------------------------------------------ #
    # Output formats
    # ------------------------------------------------------------------ #

    def save_json(self, report: DatasetReport, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    def to_markdown(self, report: DatasetReport, dataset_name: str = "Arabic Dataset") -> str:
        """Generate a HuggingFace-compatible dataset card section."""
        lines = [
            f"## Dataset Report: {dataset_name}",
            "",
            "### Basic Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total examples | {report.total_examples:,} |",
            f"| Total words | {report.total_words:,} |",
            f"| Estimated tokens | {report.estimated_tokens:,} |",
            f"| Avg length (chars) | {report.avg_length_chars} |",
            f"| Avg length (words) | {report.avg_length_words} |",
            "",
            "### Quality Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Quality pass rate | {report.quality_pass_rate}% |",
            f"| Mean quality score | {report.mean_quality_score} |",
            f"| Mean Arabic ratio | {report.arabic_ratio_mean} |",
            f"| Exact duplicate rate | {report.exact_duplicate_rate}% |",
            "",
            "### Dialect Distribution",
            "",
            f"| Dialect | Count | % |",
            f"|---------|-------|---|",
        ]
        for dialect, info in sorted(
            report.dialect_distribution.items(),
            key=lambda x: -x[1]["percentage"],
        ):
            lines.append(
                f"| {info['dialect_name']} ({dialect}) | "
                f"{info['count']:,} | {info['percentage']}% |"
            )

        lines += [
            "",
            f"**Dialect diversity score:** {report.dialect_diversity_score} "
            f"(0 = single dialect, 1 = perfectly balanced)",
            "",
            "### Quality Score Distribution",
            "",
        ]
        for bucket, count in report.quality_distribution.items():
            pct = round(count / max(report.total_examples, 1) * 100, 1)
            bar = "█" * int(pct / 2)
            lines.append(f"- `{bucket}`: {count:,} ({pct}%) {bar}")

        return "\n".join(lines)

    def print_summary(self, report: DatasetReport) -> None:
        """Print a formatted summary to the terminal."""
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  qalam — Dataset Report")
        print(sep)
        print(f"  Examples     : {report.total_examples:>10,}")
        print(f"  Words        : {report.total_words:>10,}")
        print(f"  Est. tokens  : {report.estimated_tokens:>10,}")
        print(f"  Avg length   : {report.avg_length_chars:>10.1f} chars")
        print(sep)
        print(f"  Quality pass : {report.quality_pass_rate:>9.1f}%")
        print(f"  Mean score   : {report.mean_quality_score:>10.3f}")
        print(f"  Arabic ratio : {report.arabic_ratio_mean:>10.3f}")
        print(f"  Exact dupes  : {report.exact_duplicate_rate:>9.2f}%")
        print(sep)
        print("  Dialect distribution:")
        for dialect, info in sorted(
            report.dialect_distribution.items(),
            key=lambda x: -x[1]["percentage"],
        ):
            bar = "▓" * int(info["percentage"] / 3)
            print(f"    {dialect:<6} {info['percentage']:>5.1f}%  {bar}")
        if report.flag_breakdown:
            print(sep)
            print("  Quality flags:")
            for flag, count in sorted(
                report.flag_breakdown.items(), key=lambda x: -x[1]
            ):
                print(f"    {flag:<30} {count:>6,}")
        print(sep)

    def save_html(self, report: DatasetReport, path: str, dataset_name: str = "Arabic Dataset") -> None:
        """
        Save an interactive HTML report with the dialect map visualizer.
        This is the sharable artifact for GitHub READMEs and papers.
        """
        dialect_data_js = json.dumps(report.dialect_distribution, ensure_ascii=False)
        country_data_js = json.dumps(report.country_counts, ensure_ascii=False)
        quality_data_js = json.dumps(report.quality_distribution, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{dataset_name} — qalam Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#f8f7f4;color:#1a1a18;direction:ltr}}
  .container{{max-width:900px;margin:0 auto;padding:2rem 1rem}}
  h1{{font-size:22px;font-weight:500;margin-bottom:4px}}
  .subtitle{{font-size:13px;color:#73726c;margin-bottom:2rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:2rem}}
  .card{{background:#fff;border-radius:10px;border:0.5px solid rgba(0,0,0,.1);padding:14px 16px}}
  .card-label{{font-size:11px;color:#73726c;margin-bottom:6px}}
  .card-val{{font-size:22px;font-weight:500}}
  .card-sub{{font-size:11px;color:#73726c;margin-top:2px}}
  .section{{background:#fff;border-radius:10px;border:0.5px solid rgba(0,0,0,.1);padding:1.25rem;margin-bottom:1.25rem}}
  .section h2{{font-size:14px;font-weight:500;margin-bottom:1rem;color:#444}}
  .bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:13px}}
  .bar-name{{min-width:200px;color:#1a1a18}}
  .bar-track{{flex:1;height:7px;background:#f0ede8;border-radius:4px;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:4px;background:#534AB7}}
  .bar-pct{{min-width:44px;text-align:right;color:#73726c}}
  #map{{width:100%;height:360px}}
  .flag-row{{display:flex;justify-content:space-between;font-size:12px;padding:5px 0;border-bottom:0.5px solid #f0ede8}}
  .flag-name{{color:#444}}
  .flag-val{{color:#888780}}
  footer{{font-size:11px;color:#888780;text-align:center;margin-top:2rem}}
</style>
</head>
<body>
<div class="container">
  <h1>{dataset_name}</h1>
  <p class="subtitle">Generated by qalam · github.com/KarimSeoud/Qalam</p>

  <div class="grid">
    <div class="card">
      <div class="card-label">Total examples</div>
      <div class="card-val">{report.total_examples:,}</div>
    </div>
    <div class="card">
      <div class="card-label">Estimated tokens</div>
      <div class="card-val">{report.estimated_tokens:,}</div>
    </div>
    <div class="card">
      <div class="card-label">Quality pass rate</div>
      <div class="card-val">{report.quality_pass_rate}%</div>
    </div>
    <div class="card">
      <div class="card-label">Dominant dialect</div>
      <div class="card-val">{report.dominant_dialect}</div>
      <div class="card-sub">Dialect diversity: {report.dialect_diversity_score}</div>
    </div>
    <div class="card">
      <div class="card-label">Avg length</div>
      <div class="card-val">{report.avg_length_chars}</div>
      <div class="card-sub">chars per example</div>
    </div>
    <div class="card">
      <div class="card-label">Exact duplicates</div>
      <div class="card-val">{report.exact_duplicate_rate}%</div>
    </div>
  </div>

  <div class="section">
    <h2>Dialect Distribution</h2>
    <div id="dialect-bars"></div>
  </div>

  <div class="section">
    <h2>Dialect Coverage Map</h2>
    <div id="map"></div>
  </div>

  <div class="section">
    <h2>Quality Score Distribution</h2>
    <div id="quality-bars"></div>
  </div>

  <div class="section">
    <h2>Quality Flags</h2>
    <div id="flags"></div>
  </div>

  <footer>qalam — open source Arabic LLM data preparation</footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/topojson/3.0.2/topojson.min.js"></script>
<script>
const dialectData = {dialect_data_js};
const countryData = {country_data_js};
const qualityData = {quality_data_js};

const COUNTRY_DIALECT = {{
  "EGY":"EGY","SAU":"GULF","ARE":"GULF","MAR":"MAG","DZA":"MAG","TUN":"MAG",
  "LBY":"MAG","SDN":"SDN","SYR":"LEV","LBN":"LEV","JOR":"LEV","IRQ":"IRQ",
  "KWT":"GULF","QAT":"GULF","BHR":"GULF","OMN":"GULF","YEM":"SDN","PSE":"LEV",
}};

const DIALECT_COLORS = {{
  "MSA":"#534AB7","EGY":"#1D9E75","LEV":"#D85A30","GULF":"#BA7517",
  "MAG":"#D4537E","IRQ":"#378ADD","SDN":"#639922","UNK":"#888780",
}};

const ISO3_TO_NUMERIC = {{
  "EGY":"818","SAU":"682","ARE":"784","MAR":"504","DZA":"012","TUN":"788",
  "LBY":"434","SDN":"729","SYR":"760","LBN":"422","JOR":"400","IRQ":"368",
  "KWT":"414","QAT":"634","BHR":"048","OMN":"512","YEM":"887","PSE":"275",
}};

// Render dialect bars
const dialectEl = document.getElementById('dialect-bars');
const sorted = Object.entries(dialectData).sort((a,b) => b[1].percentage - a[1].percentage);
sorted.forEach(([code, info]) => {{
  const color = DIALECT_COLORS[code] || '#888780';
  dialectEl.innerHTML += `<div class="bar-row">
    <div class="bar-name">${{info.dialect_name}} (${{code}})</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{info.percentage}}%;background:${{color}}"></div></div>
    <div class="bar-pct">${{info.percentage}}%</div>
  </div>`;
}});

// Render quality bars
const qualityEl = document.getElementById('quality-bars');
const totalEx = {report.total_examples};
Object.entries(qualityData).forEach(([bucket, count]) => {{
  const pct = (count/totalEx*100).toFixed(1);
  qualityEl.innerHTML += `<div class="bar-row">
    <div class="bar-name">${{bucket}}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:#1D9E75"></div></div>
    <div class="bar-pct">${{pct}}%</div>
  </div>`;
}});

// Render flags
const flagEl = document.getElementById('flags');
const flags = {json.dumps(report.flag_breakdown)};
if (Object.keys(flags).length === 0) {{
  flagEl.innerHTML = '<p style="font-size:13px;color:#73726c">No quality flags detected.</p>';
}} else {{
  Object.entries(flags).sort((a,b)=>b[1]-a[1]).forEach(([flag,count])=>{{
    flagEl.innerHTML += `<div class="flag-row"><span class="flag-name">${{flag}}</span><span class="flag-val">${{count.toLocaleString()}}</span></div>`;
  }});
}}

// Map
const maxCount = Math.max(1, ...Object.values(countryData));
d3.json('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json').then(world => {{
  const svg = d3.select('#map').append('svg')
    .attr('viewBox','0 0 900 360').attr('width','100%');
  const projection = d3.geoNaturalEarth1().scale(140).translate([450,190]);
  const path = d3.geoPath(projection);
  const countries = topojson.feature(world, world.objects.countries);

  const arabicNumerics = new Set(Object.values(ISO3_TO_NUMERIC));

  svg.selectAll('path').data(countries.features).join('path')
    .attr('d', path)
    .attr('fill', d => {{
      const numericId = String(d.id).padStart(3,'0');
      const iso3 = Object.entries(ISO3_TO_NUMERIC).find(([k,v])=>v===numericId)?.[0];
      if (!iso3) return '#e8e6e0';
      const dialect = COUNTRY_DIALECT[iso3];
      const count = countryData[iso3] || 0;
      if (!count) return '#d4d2c8';
      const intensity = 0.3 + (count / maxCount) * 0.7;
      const baseColor = DIALECT_COLORS[dialect] || '#534AB7';
      return baseColor;
    }})
    .attr('opacity', d => {{
      const numericId = String(d.id).padStart(3,'0');
      const iso3 = Object.entries(ISO3_TO_NUMERIC).find(([k,v])=>v===numericId)?.[0];
      const count = countryData[iso3] || 0;
      if (!iso3 || !count) return 0.15;
      return 0.35 + (count / maxCount) * 0.65;
    }})
    .attr('stroke','#fff')
    .attr('stroke-width', d => {{
      const numericId = String(d.id).padStart(3,'0');
      const iso3 = Object.entries(ISO3_TO_NUMERIC).find(([k,v])=>v===numericId)?.[0];
      return countryData[iso3] ? 0.8 : 0.3;
    }});
}});
</script>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

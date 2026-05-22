"""
Example: Clean a raw Arabic corpus for LLM pretraining.

This example shows how to take a raw JSONL file and produce
a clean, deduplicated, quality-filtered dataset ready for pretraining.
"""

from qalam.pipeline import Pipeline, PipelineConfig
from qalam.normalize import NormalizerConfig
from qalam.quality import QualityConfig
from qalam.dedup import DedupConfig

# Sample data (replace with your own corpus)
raw_texts = [
    "أعلن رئيس الوزراء عن خطة جديدة لتعزيز الاقتصاد الوطني وتحقيق التنمية المستدامة في المنطقة.",
    "أعلن رئيس الوزراء عن خطة جديدة لتعزيز الاقتصاد الوطني وتحقيق التنمية المستدامة في المنطقة.",  # duplicate
    "مرحبـــا بكـم",  # tatweel, too short
    "شو رأيك بهيك قرار؟ أنا بدي أروح هلق على الجامعة وبدي أحكيلك شي مهم.",
    "   ",  # empty
    "عايز أروح السينما النهارده بس مش عارف إيه الأفلام المتاحة دلوقتي في القاهرة.",
    "The quick brown fox jumps over the lazy dog.",  # non-Arabic
    "في عام 2024، شهدت المنطقة العربية تطورات اقتصادية مهمة أثّرت على المشهد الإقليمي بشكل كبير.",
    "في عام 2024، شهدت المنطقة العربية تطورات اقتصادية مهمة أثّرت على المشهد الإقليمي بشكل كبير.",  # dupe
    "التعليم هو أساس التنمية، وينبغي على كل دولة أن تستثمر في تعليم أبنائها لضمان مستقبل أفضل.",
]

# Configure the pipeline for pretraining
config = PipelineConfig(
    normalizer=NormalizerConfig(
        normalize_alef=True,
        remove_tatweel=True,
        diacritics="strip",
        normalize_numerals=True,
        remove_html=True,
        remove_urls=True,
    ),
    quality=QualityConfig(
        min_arabic_ratio=0.5,
        min_length=50,
        min_quality_score=0.45,
        check_toxicity=True,
        check_boilerplate=True,
    ),
    dedup=DedupConfig(
        exact_dedup=True,
        near_dedup=True,
        near_dedup_threshold=0.85,
    ),
    run_dedup=True,
    generate_report=True,
    verbose=True,
)

pipeline = Pipeline(config)
result = pipeline.run(raw_texts)

print(f"\n{'='*50}")
print(f"Clean texts ({len(result.texts)}):")
for i, text in enumerate(result.texts, 1):
    print(f"  {i}. {text[:80]}{'...' if len(text) > 80 else ''}")

# Save to JSONL
import json
with open("clean_pretraining.jsonl", "w", encoding="utf-8") as f:
    for text in result.texts:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"\nSaved to clean_pretraining.jsonl")

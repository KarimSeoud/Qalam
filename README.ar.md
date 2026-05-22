# قلم 🔤

> أداة مفتوحة المصدر لتحضير بيانات اللغة العربية وتحسين جودتها لتدريب النماذج اللغوية الكبيرة وضبطها الدقيق.

فرق تطوير النماذج العربية كانت تبني خطوط معالجة البيانات من الصفر في كل مرة. **قلم** يجمع ما تراكم من خبرة المجتمع في أداة واحدة متكاملة — بدون نماذج مطلوبة للتحميل.

---

## لماذا قلم؟

البيانات العربية تواجه تحديات فريدة لا توجد في الإنجليزية:

- **تعقيد الكتابة** — أشكال يونيكود متعددة لنفس الحرف (أ إ آ ا)، التطويل (ـ)، التشكيل، الحروف الفارسية المختلطة، وعلامات الاتجاه ثنائي الاتجاه
- **تشتت اللهجات** — الفصحى والمصرية والشامية والخليجية والمغاربية والعراقية والسودانية واليمنية تختلف لغوياً. نموذج مدرَّب على "عربية" تكون 70% منها مصرية سيضعف أداؤه على الفصحى
- **غياب خط معالجة موحد** — كل ورقة بحثية تصف خط معالجة خاص بها. قلم يحل هذه المشكلة

---

## المعيار — كشف اللهجة

كاشف اللهجات في قلم يحقق **دقة 78.6% على MADAR-6**، المعيار القياسي لكشف اللهجة العربية — بفارق **1.4 نقطة فقط عن MARBERT** (نموذج BERT عربي مضبوط بحجم 500 ميغابايت)، بينما يعمل بـ **Python خالص في 400 كيلوبايت وبدون GPU**.

| النظام | دقة MADAR-6 | حجم التثبيت | GPU |
|--------|:-----------:|:-----------:|:---:|
| الأساس العشوائي | 16.7% | — | — |
| Salameh et al. 2018 — أفضل معجم | ~67.0% | — | لا |
| **قلم** — `DialectDetector(profile="madar")` | **78.6%** | **400 كيلوبايت** | **لا** |
| MARBERT-DialectID | ~80.0% | ~500 ميغابايت | موصى به |
| CAMeL ADIDA — النموذج الرائد | ~85.0% | ~1.5 جيجابايت | مطلوب |

الكاشف الافتراضي (بدون ملف) يصل إلى **دقة 91%** عندما يُسمح له بالامتناع — استخدمه كمرشّح أولي عالي الثقة لأي خط معالجة عربي. المنهجية الكاملة والتفصيل لكل لهجة وخطوات إعادة الإنتاج في **[BENCHMARKS.md](BENCHMARKS.md)**.

---

## المميزات

| الوحدة | الوظيفة |
|--------|---------|
| `normalize` | توحيد يونيكود، الألف والهمزة، الألف المقصورة، الحروف الفارسية، إزالة التطويل، التشكيل، علامات ثنائي الاتجاه، الأرقام |
| `dialect` | كشف ٨ لهجات: فصحى / مصرية / شامية / خليجية / مغاربية / عراقية / سودانية / يمنية — مع **ملف MADAR** (دقة 78.6%، بدون GPU) |
| `quality` | تقييم الجودة: نسبة العربية، التكرار، الضوضاء، المحتوى الجاهز، السُّمية — مع رموز الإشارة التفصيلية |
| `dedup` | إزالة التكرار الحرفي (blake2b) والشبه-تكرار (MinHash LSH 64-بت) مع كشف التلوث بين المجموعات |
| `formats` | تحويل إلى SFT وDPO والتدريب المسبق — **١٢ قالب محادثة** لجميع عائلات النماذج العربية الكبرى |
| `report` | بطاقة إحصاءات كاملة مع **خريطة تفاعلية للهجات** قابلة للتصدير |

### 🗺️ مرئيات خريطة اللهجات

ارفع مجموعة بياناتك واكتشف فوراً كيف تتوزع على العالم العربي. تكشف الانحياز اللهجي قبل التدريب.

افتح `scripts/dialect_map_visualizer.html` في أي متصفح — لا يحتاج خادماً.

---

## التثبيت

```bash
pip install qalam
```

لا توجد مكتبات إلزامية للوظائف الأساسية. إضافات اختيارية:

```bash
pip install qalam[hub]   # رفع إلى HuggingFace Hub
pip install qalam[all]   # كل شيء
```

---

## البداية السريعة

### واجهة Python

```python
from qalam import Pipeline

# سطر واحد: تطبيع ← فلترة لهجة ← فلترة جودة ← إزالة تكرار
result = Pipeline().run(my_texts)
print(f"تم الاحتفاظ بـ {len(result.texts)} من أصل {len(my_texts)} مثال")
```

### خط معالجة كامل

```python
from qalam import Pipeline
from qalam.pipeline import PipelineConfig
from qalam.normalize import NormalizerConfig
from qalam.quality import QualityConfig

pipeline = Pipeline(PipelineConfig(
    normalizer=NormalizerConfig(diacritics="strip"),
    quality=QualityConfig(min_quality_score=0.6),
    dialect_filter=["MSA", "EGY"],    # فصحى ومصرية فقط
    run_dedup=True,
    generate_report=True,
    report_path="report.html",        # تقرير HTML تفاعلي
))

result = pipeline.run(texts)
```

### سطر الأوامر

```bash
# تنظيف مجموعة بيانات
qalam clean data.jsonl --output clean.jsonl

# فلترة الفصحى فقط مع تقرير
qalam clean data.jsonl --dialect msa --output msa.jsonl --report-html report.html

# تنسيق SFT مع قالب Llama-3
qalam clean data.jsonl --format sft --template llama3 --output train.jsonl

# إزالة التكرار فقط
qalam dedup data.jsonl --threshold 0.8 --output deduped.jsonl

# توليد بطاقة إحصاءات
qalam report data.jsonl --out report.html --json report.json

# عرض الخيارات المتاحة
qalam info
```

---

## أمثلة تفصيلية

### التطبيع

```python
from qalam import QalamNormalizer
from qalam.normalize import NormalizerConfig

# الإعداد الافتراضي: حذف التشكيل، توحيد الألف، إزالة التطويل
normalizer = QalamNormalizer()
clean = normalizer.normalize("مَرْحَبًا بِكُم في هَذا النَّص")
# → "مرحبا بكم في هذا النص"

# الحفاظ على التشكيل (للنصوص التراثية والقرآنية)
classical = QalamNormalizer(NormalizerConfig(diacritics="keep"))

# توحيد الحروف الفارسية (ی ک ہ) إلى مقابلاتها العربية
normalizer = QalamNormalizer(NormalizerConfig(normalize_persian=True))

# عرض ما تغيّر بالتفصيل
diff = normalizer.diff(original, clean)
# → {"changes": ["تم حذف التشكيل", "توحيد متغيرَي الألف"]}

# معالجة دفعية
clean_texts = normalizer.normalize_batch(texts)
```

| الخيار | القيم | الافتراضي |
|--------|-------|-----------|
| `diacritics` | `"strip"` / `"keep"` / `"keep_classical"` | `"strip"` |
| `normalize_alef` | bool — توحيد أ إ آ ٱ → ا | `True` |
| `normalize_alef_maksura` | bool — ى → ي | `True` |
| `normalize_teh_marbuta` | bool — ة → ه (يُغيّر المعنى، معطّل افتراضياً) | `False` |
| `normalize_persian` | bool — ی ک ہ → نظيراتها العربية | `True` |
| `remove_tatweel` | bool — حذف ـ | `True` |
| `remove_bidi_controls` | bool — حذف ZWNJ وZWJ وRLM وLRM | `True` |
| `normalize_numerals` | bool — ٠١٢ و۰۱۲ → 012 | `True` |
| `collapse_repeated_chars` | `None` أو عدد صحيح — تقليص التكرار الزائد عن N | `None` |

### كشف اللهجة

```python
from qalam import DialectDetector

# الوضع الافتراضي: معجم IDF-مرجّح — سريع وفوري، بدون نموذج
detector = DialectDetector()

# الوضع عالي الدقة: ملف MADAR (78.6% دقة على MADAR-6، 400 كيلوبايت، بدون GPU)
detector = DialectDetector(profile="madar")

result = detector.detect("عايز أروح السينما النهارده")
# DialectResult(dialect="EGY", dialect_name="Egyptian", confidence=0.85, ...)

# فلترة المدونة إلى فصحى فقط
msa_texts = detector.route(texts, keep=["MSA"])

# استثناء المغاربية
filtered = detector.route(texts, exclude=["MAG"])

# وسم كل مثال بلهجته
tagged = detector.tag_batch(texts)
# [{"text": "...", "dialect": "LEV", "confidence": 0.79}, ...]

# توزيع اللهجات (يُغذّي خريطة المرئيات)
dist = detector.distribution(texts)
# {"total": 1000, "dialects": {"EGY": {"count": 350, "percentage": 35.0}}}
```

### تقييم الجودة

```python
from qalam import QualityScorer
from qalam.quality import QualityConfig

scorer = QualityScorer(QualityConfig(
    min_arabic_ratio=0.5,
    min_length=50,
    min_quality_score=0.55,
))

result = scorer.score(text)
result.score          # 0.82
result.passed         # True
result.flag_codes     # [] أو ["too_short", "high_noise", ...]
result.arabic_ratio   # 0.89

# فلترة دفعية
clean = scorer.filter_batch(texts)

# ملخص إحصائي
summary = scorer.summary(texts)
# {"total": 1000, "passed": 847, "pass_rate": 84.7, "flag_breakdown": {...}}
```

### إزالة التكرار

```python
from qalam import Deduplicator
from qalam.dedup import DedupConfig

deduper = Deduplicator(DedupConfig(
    near_dedup_threshold=0.8,   # عتبة تشابه Jaccard
    num_perm=128,               # عدد تباديل MinHash
))

result = deduper.dedup(texts)
result.removed_count   # إجمالي المحذوفات
result.exact_removed   # التكرار الحرفي
result.near_removed    # الشبه-تكرار
result.duplicate_rate  # مثلاً: 12.4 (نسبة مئوية)
result.unique_texts    # القائمة النظيفة

# كشف التلوث: حذف أمثلة التدريب الموجودة في مجموعة الاختبار
clean_train = deduper.dedup_against(train_texts, test_texts)

# تقرير التلوث
report = deduper.contamination_report(train_texts, test_texts)
# {"contamination_rate": 2.3, "contaminated_train_examples": 230}
```

### تحويل الصيغ

```python
from qalam import FormatConverter
from qalam.formats import SFTExample, DPOExample

converter = FormatConverter(default_template="chatml")

example = SFTExample(
    instruction="ما هي عاصمة مصر؟",
    input="",
    output="عاصمة مصر هي القاهرة.",
)

# اختر أياً من ١٢ قالباً مدعوماً
converter.sft_to_formatted(example, template="llama3")    # Llama 3 / 3.1 / 3.2
converter.sft_to_formatted(example, template="llama2")    # Llama 2 Chat
converter.sft_to_formatted(example, template="mistral")   # Mistral / Mixtral / ALLaM
converter.sft_to_formatted(example, template="chatml")    # Qwen, Yi, AceGPT, جيس-v2
converter.sft_to_formatted(example, template="jais")      # جيس (نموذج عربي أولاً)
converter.sft_to_formatted(example, template="gemma4")    # Gemma 4
converter.sft_to_formatted(example, template="gemma2")    # Gemma 2 / 3
converter.sft_to_formatted(example, template="command-r") # Command-R / Aya / Aya-Expanse
converter.sft_to_formatted(example, template="phi3")      # Phi-3 / Phi-3.5
converter.sft_to_formatted(example, template="deepseek")  # DeepSeek V2 / V3 / R1
converter.sft_to_formatted(example, template="alpaca")    # Alpaca / الضبط الأكاديمي
converter.sft_to_formatted(example, template="vicuna")    # Vicuna / FastChat

# Gemma 4 مع وضع التفكير المتسلسل (chain-of-thought)
from qalam.formats import CHAT_TEMPLATES
CHAT_TEMPLATES["gemma4"]["enable_thinking"] = True
converter.sft_to_formatted(example, template="gemma4")

# حفظ كـ JSONL
converter.save_jsonl(examples, "train.jsonl")

# صيغة DPO (لـ TRL DPOTrainer)
dpo = DPOExample(
    prompt="كيف أتعلم البرمجة؟",
    chosen="ابدأ بتعلم Python عبر مشاريع عملية...",
    rejected="اقرأ كتب.",
)
converter.dpo_to_dict(dpo)

# تنسيق التدريب المسبق
converter.to_pretraining(texts, eos_token="</s>")

# رفع إلى HuggingFace Hub
converter.push_to_hub(examples, "your-org/arabic-sft-dataset")
```

### بطاقة الإحصاءات

```python
from qalam.report import ReportGenerator

gen = ReportGenerator()
report = gen.generate(texts)

# طباعة الملخص في الطرفية
gen.print_summary(report)

# حفظ تقرير HTML تفاعلي (مع خريطة اللهجات)
gen.save_html(report, "report.html", dataset_name="مجموعتي العربية")

# JSON للاستخدام البرمجي
gen.save_json(report, "report.json")

# قسم بطاقة مجموعة البيانات على HuggingFace (Markdown)
print(gen.to_markdown(report))
```

---

## اللهجات المدعومة

| الرمز | اللهجة | الدول |
|-------|--------|-------|
| MSA | العربية الفصحى | جميع الدول العربية |
| EGY | المصرية | مصر |
| LEV | الشامية | سوريا، لبنان، الأردن، فلسطين |
| GULF | الخليجية | السعودية، الإمارات، الكويت، قطر، البحرين، عُمان |
| MAG | المغاربية | المغرب، الجزائر، تونس، ليبيا |
| IRQ | العراقية | العراق |
| SDN | السودانية | السودان |
| YEM | اليمنية | اليمن |

---

## قوالب المحادثة المدعومة (١٢ قالباً)

| القالب | النماذج | ملاحظات |
|--------|---------|---------|
| `chatml` | Qwen 2.5/3، Yi، AceGPT، InternLM، Zephyr | الصيغة الأكثر شيوعاً للنماذج مفتوحة المصدر |
| `llama3` | Llama 3، 3.1، 3.2 | |
| `llama2` | Llama 2 Chat، إصدارات AceGPT القديمة | النظام مدمج في أول [INST] |
| `mistral` | Mistral، Mixtral، ALLaM (SDAIA)، Mistral-Arabic | |
| `jais` | جيس (MBZUAI) | نموذج عربي أولاً |
| `gemma4` | Gemma 4 بجميع أحجامه | `enable_thinking=True` للتفكير المتسلسل |
| `gemma2` | Gemma 2 / Gemma 3 | النظام يُدمج في رسالة المستخدم الأولى |
| `command-r` | Command-R، Command-R+، Aya-23، Aya-Expanse (Cohere) | أشمل تغطية متعددة اللغات |
| `phi3` | Phi-3، Phi-3.5 Mini / Small / Medium (Microsoft) | |
| `deepseek` | DeepSeek-V2، V3، R1 | يستخدم رموز الشريط الكامل ｜ |
| `alpaca` | Stanford Alpaca — المعيار الأكاديمي للضبط الدقيق | صيغة ### Instruction / ### Response |
| `vicuna` | Vicuna 1.1، FastChat | صيغة USER: / ASSISTANT: |

---

## الإعدادات المسبقة

```bash
qalam clean corpus.jsonl --config configs/pretraining.yaml  # التدريب المسبق: استرجاع عالٍ، جميع اللهجات
qalam clean data.jsonl   --config configs/sft.yaml          # الضبط الدقيق: دقة أعلى، إزالة تكرار صارمة
qalam clean quran.jsonl  --config configs/classical.yaml    # النصوص التراثية: الحفاظ على التشكيل
```

---

## التطوير

```bash
git clone https://github.com/KarimSeoud/Qalam qalam
cd qalam
pip install -e ".[dev]"

# تشغيل الاختبارات
pytest tests/ -v

# التنسيق
black qalam/ tests/
ruff check qalam/
```

---

## خارطة الطريق

- [x] معيار MADAR-6 (دقة 78.6%، Python خالص، بدون GPU)
- [x] ملف لهجة متعلَّم — أوزان خطية مستخرجة من الانحدار اللوجستي، 400 كيلوبايت JSON
- [x] ١٢ قالب محادثة يغطيان جميع عائلات النماذج العربية الكبرى
- [x] تقسيم SDN/YEM إلى رمزين مستقلين
- [ ] إعادة صياغة النصوص بالذكاء الاصطناعي (Claude/GPT — إصلاح النص بدلاً من فلترته)
- [ ] تقييم الارتباك (Perplexity) باستخدام نموذج لغوي عربي صغير (تكامل CAMeL Tools)
- [ ] دعم البث من HuggingFace Datasets للمدونات الضخمة
- [ ] واجهة ويب للخط الكامل (وليس فقط خريطة اللهجات)
- [ ] معيار: خط جودة البيانات مقارنةً بالبيانات الخام على مهام NLP العربية اللاحقة

---

## المساهمة

نرحب بالمساهمات في:
- **تحسين معاجم اللهجات** في `qalam/dialect.py`
- **قوالب محادثة جديدة** في `qalam/formats.py`
- **أنماط المحتوى غير المرغوب والقوالب الجاهزة** في `qalam/quality.py`
- **تقارير الأخطاء وحالات الاستخدام الخاصة**

يُرجى فتح issue قبل أي pull request كبير.

---

## الترخيص

MIT — انظر [LICENSE](LICENSE)

---

## الاستشهاد

```bibtex
@software{qalam,
  title  = {Qalam: A pipeline for preparing Arabic datasets for LLM training},
  year   = {2026},
  url    = {https://github.com/KarimSeoud/Qalam}
}
```

---

*قلم — أداة الكتابة العربية*

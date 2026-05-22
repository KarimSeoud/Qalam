"""
Example: Prepare Arabic SFT data with chat templates.

Shows how to convert Q&A pairs into multiple training formats
(Llama-3, Mistral, ChatML/Qwen, Jais) ready for fine-tuning.
"""

from qalam.formats import FormatConverter, SFTExample, DPOExample, ChatMessage

converter = FormatConverter(default_template="chatml")

# ── Supervised Fine-Tuning Examples ─────────────────────────────────────

sft_examples = [
    SFTExample(
        instruction="ما هي عاصمة المملكة العربية السعودية؟",
        input="",
        output="عاصمة المملكة العربية السعودية هي مدينة الرياض.",
        source="geography_qa",
    ),
    SFTExample(
        instruction="اشرح مفهوم التعلم الآلي بأسلوب بسيط.",
        input="",
        output=(
            "التعلم الآلي هو فرع من فروع الذكاء الاصطناعي يُمكّن الحواسيب من التعلم "
            "من البيانات وتحسين أدائها تلقائياً دون الحاجة إلى برمجة صريحة لكل مهمة. "
            "تخيل أنك تُعلّم طفلاً التمييز بين القطط والكلاب بعرض آلاف الصور عليه — "
            "هكذا يعمل التعلم الآلي تماماً."
        ),
        source="explanations",
    ),
    SFTExample(
        instruction="اكتب قصيدة قصيرة عن البحر.",
        input="في أسلوب الشعر الحديث",
        output=(
            "البحرُ مرآةٌ للسماء\n"
            "يحملُ أسرارَ الغياب\n"
            "موجةٌ تأتي وتمضي\n"
            "كالحلمِ في عينِ الصحاب"
        ),
        source="creative_writing",
    ),
]

# Show all available templates
print("Available chat templates:")
for key, name in converter.list_templates().items():
    print(f"  {key}: {name}")

print("\n" + "="*60)

# Apply different templates and compare
for template in ["chatml", "llama3", "jais"]:
    print(f"\n── Template: {template} ──")
    formatted = converter.sft_to_formatted(sft_examples[0], template=template)
    print(formatted[:300] + "..." if len(formatted) > 300 else formatted)

# ── DPO Examples ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("\nDPO Format:")

dpo_example = DPOExample(
    prompt="كيف يمكنني تحسين مستواي في اللغة العربية؟",
    chosen=(
        "لتحسين مستواك في اللغة العربية، أنصحك بالقراءة اليومية للكتب والمقالات، "
        "والاستماع إلى البرامج الإذاعية باللغة الفصحى، وممارسة الكتابة الإبداعية، "
        "والانخراط في محادثات مع متحدثين أصليين."
    ),
    rejected="اقرأ كتب واستمع لأشياء.",
)

dpo_dict = converter.dpo_to_dict(dpo_example, template="chatml")
import json
print(json.dumps(dpo_dict, ensure_ascii=False, indent=2)[:400])

# ── Save to files ─────────────────────────────────────────────────────────

# Save as JSONL
count = converter.save_jsonl(sft_examples, "arabic_sft_data.jsonl")
print(f"\nSaved {count} SFT examples to arabic_sft_data.jsonl")

# Print formatted example
print("\n" + "="*60)
print("Final formatted example (ChatML):")
print(converter.sft_to_formatted(sft_examples[1], template="chatml"))

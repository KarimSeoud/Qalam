"""
Module 5 — Format & Structure Converters
==========================================
Converts cleaned Arabic text into training-ready formats for:
  - SFT  (Supervised Fine-Tuning)  — instruction/response pairs
  - DPO  (Direct Preference Optimization) — chosen/rejected pairs
  - Chat templates: ChatML (Qwen/Yi), Llama-3, Llama-2, Mistral, Jais,
                    Gemma-2, Gemma-4, Command-R (Aya), Phi-3, DeepSeek,
                    Alpaca, Vicuna
  - HuggingFace datasets (JSONL and push to Hub)
  - Pretraining format — plain text concatenated with EOS tokens

Each converter preserves the original Arabic text while adding
the structural metadata required by different training frameworks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SFTExample:
    """Supervised fine-tuning example."""
    instruction: str
    input: str
    output: str
    system: Optional[str] = None
    dialect: Optional[str] = None
    source: Optional[str] = None


@dataclass
class DPOExample:
    """Direct Preference Optimization example."""
    prompt: str
    chosen: str
    rejected: str
    system: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ChatMessage:
    role: str   # "system", "user", "assistant"
    content: str


# ---------------------------------------------------------------------------
# Chat template definitions
# ---------------------------------------------------------------------------

CHAT_TEMPLATES = {
    "llama3": {
        "name": "Llama 3",
        "bos": "<|begin_of_text|>",
        "eos": "<|end_of_text|>",
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
    },
    "mistral": {
        "name": "Mistral / Mixtral",
        "bos": "<s>",
        "eos": "</s>",
        "system_start": "",
        "system_end": "",
        "user_start": "[INST] ",
        "user_end": " [/INST]",
        "assistant_start": " ",
        "assistant_end": "</s>",
    },
    "chatml": {
        "name": "ChatML (Qwen / Yi / InternLM)",
        "bos": "",
        "eos": "<|im_end|>",
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
    },
    "jais": {
        "name": "Jais (Arabic LLM)",
        "bos": "",
        "eos": "</s>",
        "system_start": "### System:\n",
        "system_end": "\n",
        "user_start": "### Human:\n",
        "user_end": "\n",
        "assistant_start": "### Assistant:\n",
        "assistant_end": "\n</s>",
    },
    "gemma4": {
        # Gemma 4 uses asymmetric <|turn>role ... <turn|> delimiters —
        # a completely different structure from all other supported templates.
        # Cannot be expressed with flat start/end keys, so apply_chat_template
        # handles it via the "_special": "gemma4" sentinel.
        #
        # Basic format (thinking disabled):
        #   <bos><|turn>system\n{content}<turn|>
        #   <|turn>user\n{content}<turn|>
        #   <|turn>model\n{content}<eos>
        #
        # Thinking mode: set enable_thinking=True to prepend <|think|> to
        # the system prompt, which activates chain-of-thought reasoning.
        # Docs: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
        "name": "Gemma 4 (Google DeepMind)",
        "bos": "<bos>",
        "eos": "<eos>",
        "_special": "gemma4",
        "enable_thinking": False,
    },
    "gemma2": {
        # Gemma 2 / 3 do not natively support a `system` role — system content
        # is merged into the first user turn. Multi-turn format:
        #   <bos><start_of_turn>user\n{user}<end_of_turn>
        #   <start_of_turn>model\n{assistant}<end_of_turn>
        # Handled via "_special": "gemma2" so system is folded into user.
        "name": "Gemma 2 / Gemma 3 (Google DeepMind)",
        "bos": "<bos>",
        "eos": "<end_of_turn>",
        "_special": "gemma2",
    },
    "llama2": {
        # Llama-2-Chat. System prompt is embedded inside the first [INST] block.
        # Each turn restarts with <s>:
        #   <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user1} [/INST] {ans1} </s>
        #   <s>[INST] {user2} [/INST] {ans2} </s>
        # Cannot be expressed with the flat schema.
        "name": "Llama 2 / Llama 2 Chat (Meta)",
        "bos": "<s>",
        "eos": "</s>",
        "_special": "llama2",
    },
    "command-r": {
        # Cohere Command-R / Command-R+ / Aya / Aya-Expanse. Distinct token
        # set; fits the flat schema cleanly.
        "name": "Command-R / Aya (Cohere)",
        "bos": "<BOS_TOKEN>",
        "eos": "<|END_OF_TURN_TOKEN|>",
        "system_start": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        "system_end": "<|END_OF_TURN_TOKEN|>",
        "user_start": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "user_end": "<|END_OF_TURN_TOKEN|>",
        "assistant_start": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "assistant_end": "<|END_OF_TURN_TOKEN|>",
    },
    "phi3": {
        # Microsoft Phi-3 / Phi-3.5 (mini, small, medium, MoE).
        # Format:  <|system|>\n{sys}<|end|>\n<|user|>\n{usr}<|end|>\n
        #          <|assistant|>\n{ans}<|end|>\n<|endoftext|>
        "name": "Phi-3 / Phi-3.5 (Microsoft)",
        "bos": "",
        "eos": "<|endoftext|>",
        "system_start": "<|system|>\n",
        "system_end": "<|end|>\n",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "<|end|>\n",
    },
    "deepseek": {
        # DeepSeek-V2 / V3 / R1 chat format. Uses fullwidth pipe (｜, U+FF5C).
        # System content is prepended raw, no header. Format:
        #   <｜begin▁of▁sentence｜>{sys}<｜User｜>{user}
        #   <｜Assistant｜>{assistant}<｜end▁of▁sentence｜>
        "name": "DeepSeek V2 / V3 / R1",
        "bos": "<｜begin▁of▁sentence｜>",
        "eos": "<｜end▁of▁sentence｜>",
        "system_start": "",
        "system_end": "",
        "user_start": "<｜User｜>",
        "user_end": "",
        "assistant_start": "<｜Assistant｜>",
        "assistant_end": "<｜end▁of▁sentence｜>",
    },
    "alpaca": {
        # Stanford Alpaca / academic-style SFT format. Single-turn only by
        # design; multi-turn conversations collapse with "\n\n" between turns.
        # The standard preamble is included in the BOS slot.
        "name": "Alpaca (Stanford / academic SFT)",
        "bos": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
        ),
        "eos": "",
        "system_start": "",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n\n",
    },
    "vicuna": {
        # Vicuna 1.1 / FastChat. Single-line turns separated by spaces.
        # Format:
        #   {sys} USER: {user} ASSISTANT: {assistant}</s>USER: ...
        "name": "Vicuna 1.1 / FastChat",
        "bos": "",
        "eos": "</s>",
        "system_start": "",
        "system_end": " ",
        "user_start": "USER: ",
        "user_end": " ",
        "assistant_start": "ASSISTANT: ",
        "assistant_end": "</s>",
    },
}

DEFAULT_ARABIC_SYSTEM_PROMPT = (
    "أنت مساعد ذكاء اصطناعي مفيد وأمين. "
    "أجب دائماً باللغة العربية الفصحى ما لم يطلب المستخدم خلاف ذلك."
)


class FormatConverter:
    """
    Converts Arabic text data into training-ready formats.

    Usage:
        converter = FormatConverter()

        # Convert raw Q&A pairs to SFT format
        examples = [
            SFTExample(
                instruction="ما هي عاصمة المملكة العربية السعودية؟",
                input="",
                output="عاصمة المملكة العربية السعودية هي مدينة الرياض.",
            )
        ]
        jsonl = converter.to_jsonl(examples)

        # Apply Llama-3 chat template
        formatted = converter.apply_chat_template(messages, template="llama3")

        # Save as HuggingFace-compatible JSONL
        converter.save_jsonl(examples, "train_sft.jsonl")
    """

    def __init__(self, default_template: str = "chatml"):
        if default_template not in CHAT_TEMPLATES:
            raise ValueError(
                f"Unknown template '{default_template}'. "
                f"Choose from: {list(CHAT_TEMPLATES.keys())}"
            )
        self.default_template = default_template

    # ------------------------------------------------------------------ #
    # Chat template application
    # ------------------------------------------------------------------ #

    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        template: Optional[str] = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Format a conversation using a chat template.

        Args:
            messages: List of ChatMessage objects.
            template: Template name (llama3, mistral, chatml, jais).
                      Defaults to self.default_template.
            add_generation_prompt: If True, append the assistant header
                                   (useful for inference prompts).
        """
        tmpl = CHAT_TEMPLATES[template or self.default_template]

        # Gemma 4 uses a unique asymmetric turn-token format that cannot be
        # expressed with the flat start/end key schema used by other templates.
        special = tmpl.get("_special")
        if special == "gemma4":
            return self._apply_gemma4_template(
                messages, tmpl, add_generation_prompt
            )
        if special == "gemma2":
            return self._apply_gemma2_template(
                messages, tmpl, add_generation_prompt
            )
        if special == "llama2":
            return self._apply_llama2_template(
                messages, tmpl, add_generation_prompt
            )

        parts = [tmpl["bos"]]

        for msg in messages:
            if msg.role == "system":
                parts.append(
                    tmpl["system_start"] + msg.content + tmpl["system_end"]
                )
            elif msg.role == "user":
                parts.append(
                    tmpl["user_start"] + msg.content + tmpl["user_end"]
                )
            elif msg.role == "assistant":
                parts.append(
                    tmpl["assistant_start"] + msg.content + tmpl["assistant_end"]
                )

        if add_generation_prompt:
            parts.append(tmpl["assistant_start"])

        return "".join(parts)

    def _apply_gemma4_template(
        self,
        messages: list,
        tmpl: dict,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Gemma 4 chat template.

        Format (thinking disabled):
            <bos><|turn>system
            {system_content}<turn|>
            <|turn>user
            {user_content}<turn|>
            <|turn>model
            {assistant_content}<eos>

        For multi-turn dialogs, only the *final* assistant turn closes with
        <eos> — intermediate assistant turns close with <turn|> like any other
        turn. (Previously every assistant turn emitted <eos>, which during
        training would teach the model to stop generation after the first reply.)

        Thinking mode (enable_thinking=True):
            System content is prefixed with <|think|> to activate
            chain-of-thought reasoning. The model outputs its internal
            reasoning wrapped in <|channel>thought\n...<channel|> before
            the final answer. Strip thought blocks from multi-turn history.
        """
        parts = [tmpl["bos"]]
        enable_thinking = tmpl.get("enable_thinking", False)
        last_idx = len(messages) - 1

        for i, msg in enumerate(messages):
            if msg.role == "system":
                content = msg.content
                if enable_thinking:
                    content = "<|think|>\n" + content
                parts.append(f"<|turn>system\n{content}<turn|>\n")
            elif msg.role == "user":
                parts.append(f"<|turn>user\n{msg.content}<turn|>\n")
            elif msg.role == "assistant":
                # Final assistant turn closes with <eos>; intermediate ones
                # close with <turn|> so generation continues into the next turn.
                is_final = (i == last_idx)
                closer = tmpl["eos"] if is_final else "<turn|>"
                parts.append(f"<|turn>model\n{msg.content}{closer}\n")

        if add_generation_prompt:
            parts.append("<|turn>model\n")

        return "".join(parts)

    def _apply_gemma2_template(
        self,
        messages: list,
        tmpl: dict,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Gemma 2 / Gemma 3 chat template.

        Gemma 2 has no native `system` role. We fold any system content into
        the first user turn, separated by a blank line. Format:

            <bos><start_of_turn>user
            {system}

            {user_1}<end_of_turn>
            <start_of_turn>model
            {assistant_1}<end_of_turn>
            <start_of_turn>user
            {user_2}<end_of_turn>
            ...
        """
        parts = [tmpl["bos"]]
        pending_system: list[str] = []
        first_user_seen = False

        for msg in messages:
            if msg.role == "system":
                # Hold system content until we see the first user turn.
                pending_system.append(msg.content)
            elif msg.role == "user":
                content = msg.content
                if pending_system and not first_user_seen:
                    content = "\n\n".join(pending_system) + "\n\n" + content
                    pending_system = []
                first_user_seen = True
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif msg.role == "assistant":
                parts.append(
                    f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"
                )

        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")

        return "".join(parts)

    def _apply_llama2_template(
        self,
        messages: list,
        tmpl: dict,
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Llama-2-Chat template.

        System prompt is embedded inside the FIRST [INST] block with <<SYS>>
        tags. Every turn after the first starts with a fresh <s>. Format:

            <s>[INST] <<SYS>>
            {system}
            <</SYS>>

            {user_1} [/INST] {assistant_1} </s><s>[INST] {user_2} [/INST] ...

        For inference (add_generation_prompt=True), trailing turn closes
        after `[/INST]` with no assistant content.
        """
        parts: list[str] = []
        system_content: Optional[str] = None
        first_user_seen = False
        last_was_user = False

        # Pull out system content (only one is honored — Llama 2's design).
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
                break

        # Walk through user/assistant turns in order.
        for msg in messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                parts.append("<s>[INST] ")
                if not first_user_seen and system_content:
                    parts.append(
                        f"<<SYS>>\n{system_content}\n<</SYS>>\n\n"
                    )
                parts.append(f"{msg.content} [/INST]")
                first_user_seen = True
                last_was_user = True
            elif msg.role == "assistant":
                parts.append(f" {msg.content} </s>")
                last_was_user = False

        # If the conversation ends on a user turn and the caller wants a
        # generation prompt, nothing more to append — `[/INST]` already
        # primes the model to respond.
        if add_generation_prompt and not last_was_user:
            # Caller wants to extend an assistant-final convo; rare, but
            # emit an empty user→[/INST] hand-off so the model speaks next.
            parts.append("<s>[INST]  [/INST]")

        return "".join(parts)

    # ------------------------------------------------------------------ #
    # SFT format
    # ------------------------------------------------------------------ #

    def sft_to_messages(
        self,
        example: SFTExample,
        system_prompt: Optional[str] = None,
    ) -> list[ChatMessage]:
        """Convert an SFTExample to a list of ChatMessages."""
        messages = []

        sys = system_prompt or example.system or DEFAULT_ARABIC_SYSTEM_PROMPT
        messages.append(ChatMessage(role="system", content=sys))

        user_content = example.instruction
        if example.input:
            user_content += f"\n\n{example.input}"
        messages.append(ChatMessage(role="user", content=user_content))
        messages.append(ChatMessage(role="assistant", content=example.output))
        return messages

    def sft_to_formatted(
        self,
        example: SFTExample,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Convert an SFTExample directly to a formatted string."""
        messages = self.sft_to_messages(example, system_prompt)
        return self.apply_chat_template(messages, template)

    # ------------------------------------------------------------------ #
    # DPO format
    # ------------------------------------------------------------------ #

    def dpo_to_dict(
        self,
        example: DPOExample,
        template: Optional[str] = None,
    ) -> dict:
        """
        Convert a DPOExample to the format expected by TRL's DPOTrainer.
        Output:  { "prompt": ..., "chosen": ..., "rejected": ... }
        """
        tmpl = template or self.default_template
        system = example.system or DEFAULT_ARABIC_SYSTEM_PROMPT

        prompt_messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=example.prompt),
        ]
        prompt_str = self.apply_chat_template(
            prompt_messages, tmpl, add_generation_prompt=True
        )

        return {
            "prompt": prompt_str,
            "chosen": example.chosen,
            "rejected": example.rejected,
            "source": example.source,
        }

    # ------------------------------------------------------------------ #
    # Pretraining format
    # ------------------------------------------------------------------ #

    def to_pretraining(
        self,
        texts: list[str],
        eos_token: str = "</s>",
        separator: str = "\n\n",
    ) -> str:
        """
        Concatenate texts for pretraining with EOS tokens.
        Returns a single large string ready for tokenization.
        """
        return (eos_token + separator).join(texts) + eos_token

    def to_pretraining_chunks(
        self,
        texts: list[str],
        max_chars: int = 50_000,
        eos_token: str = "</s>",
    ) -> Iterator[str]:
        """Yield pretraining chunks of approximately max_chars characters."""
        current_chunk: list[str] = []
        current_len = 0

        for text in texts:
            entry = text + eos_token
            if current_len + len(entry) > max_chars and current_chunk:
                yield "\n\n".join(current_chunk)
                current_chunk = []
                current_len = 0
            current_chunk.append(entry)
            current_len += len(entry)

        if current_chunk:
            yield "\n\n".join(current_chunk)

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #

    def to_jsonl(self, examples: list) -> str:
        """Serialize a list of examples (dataclasses or dicts) to JSONL."""
        lines = []
        for ex in examples:
            if hasattr(ex, "__dataclass_fields__"):
                d = {k: v for k, v in asdict(ex).items() if v is not None}
            else:
                d = ex
            lines.append(json.dumps(d, ensure_ascii=False))
        return "\n".join(lines)

    def save_jsonl(self, examples: list, path: str) -> int:
        """Save examples to a JSONL file. Returns number of examples written."""
        content = self.to_jsonl(examples)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return len(examples)

    def load_jsonl(self, path: str) -> list[dict]:
        """Load a JSONL file into a list of dicts."""
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def convert_alpaca_to_sft(self, path: str) -> list[SFTExample]:
        """
        Convert an Alpaca-format JSONL to SFTExample objects.
        Alpaca format: {"instruction": ..., "input": ..., "output": ...}
        """
        raw = self.load_jsonl(path)
        return [
            SFTExample(
                instruction=r.get("instruction", ""),
                input=r.get("input", ""),
                output=r.get("output", ""),
            )
            for r in raw
        ]

    def push_to_hub(
        self,
        examples: list,
        repo_id: str,
        split: str = "train",
        token: Optional[str] = None,
    ) -> None:
        """
        Push examples to HuggingFace Hub.
        Requires: pip install datasets huggingface_hub
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "Install the 'datasets' package: pip install datasets"
            )

        records = []
        for ex in examples:
            if hasattr(ex, "__dataclass_fields__"):
                records.append({k: v for k, v in asdict(ex).items() if v is not None})
            else:
                records.append(ex)

        dataset = Dataset.from_list(records)
        dataset.push_to_hub(repo_id, split=split, token=token)
        print(f"Pushed {len(records)} examples to {repo_id} (split: {split})")

    def list_templates(self) -> dict[str, str]:
        """Return available chat templates with their names."""
        return {k: v["name"] for k, v in CHAT_TEMPLATES.items()}

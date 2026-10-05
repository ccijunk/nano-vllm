"""T3b · char-level tokenizer（C1 侧车 vocab.json，`{"itos": [...]}` 格式，与 nano-model CharTokenizer 同构）。

char 词表无 BOS/EOS 特殊 token → eos_token_id = None；引擎 finish_reason 仅 "length"
可触发（doc/topics/t3_nano_vllm.md §6.1）。
"""

import json
from pathlib import Path


class CharTokenizer:

    eos_token_id = None

    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("duplicate chars in vocab")

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "CharTokenizer":
        itos = json.loads((Path(model_dir) / "vocab.json").read_text())["itos"]
        return cls(itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]  # 语料外字符 KeyError = 快速失败

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

"""C1 fixture 生成器：合成 nano 模型目录（单一事实源）。

两个消费方：
- pytest：tests/conftest.py 经 gen_weights/write_model_dir 建 fixture（LV1）
- LV2 loader Job：`python3 tests/c1_fixture.py /data/model`（裁决⑥：loader 写 PVC，幂等）

超参 = C1 基线（reference_model.CFG，单一事实源）；
权重 = 固定 seed 的 fp32 master（供参考实现）→ bf16 落盘（供引擎 bitwise 加载，E-B1）。
"""

import json
import os
import sys

import torch
from safetensors.torch import save_file

from reference_model import CFG

SEED = 20261004
MODEL_FILE = "model.safetensors"


def gen_weights() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(SEED)

    def rand(*shape, std=0.05):
        return torch.randn(*shape, generator=g) * std

    def norm_w(n):
        return 1.0 + torch.randn(n, generator=g) * 0.02

    V, H, KVH, D, I = CFG["vocab_size"], CFG["hidden_size"], CFG["n_kv_heads"], CFG["head_dim"], CFG["intermediate_size"]
    fp32 = {
        "model.embed_tokens.weight": rand(V, H, std=0.1),
        "model.norm.weight": norm_w(H),
    }
    for i in range(CFG["n_layers"]):
        p = f"model.layers.{i}."
        fp32.update({
            p + "input_layernorm.weight": norm_w(H),
            p + "self_attn.q_proj.weight": rand(CFG["n_heads"] * D, H),
            p + "self_attn.k_proj.weight": rand(KVH * D, H),
            p + "self_attn.v_proj.weight": rand(KVH * D, H),
            p + "self_attn.o_proj.weight": rand(H, CFG["n_heads"] * D),
            p + "self_attn.q_norm.weight": norm_w(D),
            p + "self_attn.k_norm.weight": norm_w(D),
            p + "post_attention_layernorm.weight": norm_w(H),
            p + "mlp.gate_proj.weight": rand(I, H),
            p + "mlp.up_proj.weight": rand(I, H),
            p + "mlp.down_proj.weight": rand(H, I),
        })
    bf16 = {k: v.to(torch.bfloat16) for k, v in fp32.items()}
    return fp32, bf16


def write_model_dir(target: str) -> str:
    """落盘 C1 侧车目录（幂等：模型文件已存在则跳过——确定性 seed 下重生成结果恒等）。"""
    if os.path.exists(os.path.join(target, MODEL_FILE)):
        return target
    os.makedirs(target, exist_ok=True)
    _, bf16 = gen_weights()
    save_file(bf16, os.path.join(target, MODEL_FILE))
    with open(os.path.join(target, "config.json"), "w") as f:
        json.dump(CFG, f)
    # vocab.json = {"itos": [...]}（与 nano-model CharTokenizer 同构）；42 个可打印 ASCII
    with open(os.path.join(target, "vocab.json"), "w") as f:
        json.dump({"itos": [chr(33 + i) for i in range(CFG["vocab_size"])]}, f)
    return target


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/data/model"
    action = "exists (skip)" if os.path.exists(os.path.join(target, MODEL_FILE)) else "written"
    print(f"C1 fixture {action}: {write_model_dir(target)}")

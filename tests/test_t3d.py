"""T3d · LV1-b 多池验证（doc/topics/t3d_swa_multipool.md §3）：E-D2 账面 / E-D3 带窗正确性 / E-D4 回收 / E-D5 graph 冒烟。

运行（C5 契约命令，容器内；同 test_t3b 头注）：
docker run --rm --gpus all --network none -v "$PWD":/w -w /w/nano-vllm \
  -e PYTHONPATH=/w/nano-vllm -v "$PWD/.cache":/opt/cache -e HF_HOME=/opt/cache/hf \
  nano-stack/engine:vllm python3 -u -m pytest tests -q

E-D3 方法 = E-B2 同源：decisive 步（fp32 参考 top1-top2 gap ≥ 5e-2）强制与带窗参考一致；
batch/serial 分叉只允许发生在 near-tie 步（bf16 噪声区，T2 两阶段判据同源教训）。
引擎生命周期：E-D3/E-D5 各自建引擎并在测末退出 + empty_cache（dist 端口与 VRAM 预算均需释放）。
"""

import gc

import pytest
import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

from reference_model import CFG
from reference_model import forward as ref_forward

GAP = 5e-2        # decisive 步阈值（E-B2 同源）
WINDOW = 256      # 与 conftest.swa_model_dir 一致（= kvcache_block_size）
SWA_LAYERS = (1,)
PROMPTS = ["ABCDEFGH" * 70, "!#%&" * 60, "0123456789()" * 30, "*+-/:;<=>?@" * 10]
# 长度 560（> W+bs=512，decode 触发窗口滑出）/ 240 / 360 / 110（< W，窗口不裁剪）

ED3_BATCH = None  # E-D3 产出 → E-D5 graph 冒烟基准


def _ref_top1_gap(fp32, ctx):
    logits = ref_forward(fp32, ctx, CFG, swa_layers=SWA_LAYERS, window=WINDOW)
    top2 = torch.topk(logits, 2)
    return top2.indices[0].item(), (top2.values[0] - top2.values[1]).item()


def _teacher_force(fp32, ctx: list[int], stream: list[int]) -> int:
    """从 ctx 出发逐步校验 stream 中每个 decisive 步的 argmax（分叉不级联）。"""
    ctx = list(ctx)
    checked = 0
    for tid in stream:
        top1, gap = _ref_top1_gap(fp32, ctx)
        if gap > GAP:
            assert top1 == tid, f"decisive 步: ref={top1} engine={tid} gap={gap:.4f}"
            checked += 1
        ctx.append(tid)
    return checked


def test_ed2_swa_pool_bookkeeping():
    """E-D2 账面：窗口数学 + 饱和恒定（≤ ceil(W/bs)+1）+ 全池单调增长 + 预算公式可复核。"""
    bs = 8
    Sequence.block_size = bs
    try:
        bm = BlockManager(num_blocks=64, block_size=bs, num_blocks_swa=16, window=24)
        seq = Sequence(list(range(100)))
        assert bm.can_allocate(seq) == 0          # 决策②扩展：SWA 模型不走 prefix cache
        bm.allocate(seq, 0)
        assert seq.swa_table_start == (100 - 24) // bs * bs == 72
        assert len(seq.swa_block_table) == 4      # ceil(28/8)
        full_free0 = len(bm.free_block_ids)
        max_blocks = -(-24 // bs) + 1             # ceil(W/bs)+1 = 4 = (W-1)//bs+2
        for _ in range(120):                      # decode 120 步：跨越 15 个块边界
            seq.append_token(1)
            bm.may_append(seq)
            assert len(seq.swa_block_table) <= max_blocks
        assert len(seq.swa_block_table) == max_blocks
        assert seq.swa_table_start == (seq.num_tokens - 24) // bs * bs
        assert len(bm.free_block_ids) < full_free0        # 全池占用单调增长
        assert len(bm.free_block_ids_swa) == 16 - max_blocks   # SWA 池饱和后恒定
        # 决策④预算公式的每-seq 下界：blocks_swa ≥ max_num_seqs × (W//bs + 2) ≥ 每-seq 峰值占用
        assert 16 >= 1 * (24 // bs + 2) >= max_blocks
    finally:
        Sequence.block_size = 256


def test_ed4_swa_block_recycle():
    """E-D4 回收：窗口饱和的 seq 退出后，SWA 块 ref_count 归零回 free list 并被新 seq 复用。"""
    bs = 8
    Sequence.block_size = bs
    try:
        bm = BlockManager(num_blocks=64, block_size=bs, num_blocks_swa=8, window=24)
        seq_a = Sequence(list(range(100)))
        assert bm.can_allocate(seq_a) == 0
        bm.allocate(seq_a, 0)
        a_blocks = list(seq_a.swa_block_table)
        seq_b = Sequence(list(range(100, 200)))
        assert bm.can_allocate(seq_b) == 0
        bm.allocate(seq_b, 0)
        seq_c = Sequence(list(range(200, 300)))
        assert bm.can_allocate(seq_c) == -1       # SWA 池 8 块耗尽（4+4）
        bm.deallocate(seq_a)                      # A 退出 → 块归还
        assert all(bm.blocks_swa[i].ref_count == 0 for i in a_blocks)
        assert bm.can_allocate(seq_c) == 0
        bm.allocate(seq_c, 0)
        assert set(seq_c.swa_block_table) == set(a_blocks)   # 精确复用 A 释放的块
    finally:
        Sequence.block_size = 256


def test_ed3_swa_greedy_reference(swa_model_dir, c1_weights):
    """E-D3：pools=2 混合模型 batch greedy ≡ serial ≡ fp32 带窗参考（decisive 步）。"""
    global ED3_BATCH
    fp32, _ = c1_weights
    engine = LLMEngine(swa_model_dir, enforce_eager=True, gpu_memory_utilization=0.5)
    try:
        # 账面（E-D2 引擎侧）：预算公式（决策④）+ 池张量形状 + kv_org 声明
        assert engine.config.num_kvcache_blocks_swa == \
            engine.config.max_num_seqs * (WINDOW // engine.config.kvcache_block_size + 2)
        assert engine.model_runner.kv_cache_swa.shape == \
            (2, len(SWA_LAYERS), engine.config.num_kvcache_blocks_swa, 256, 2, 32)
        assert engine.model_runner.model.kv_org.pools == 2
        assert engine.model_runner.model.kv_org.layout == "paged_kt_vt|paged_kt_vt_swa"

        sp = SamplingParams(temperature=0, max_tokens=16)
        batch = engine.generate(PROMPTS, sp, use_tqdm=False)
        serial = []
        for p in PROMPTS:
            serial += engine.generate([p], sp, use_tqdm=False)
        ED3_BATCH = [o["token_ids"] for o in batch]

        tok = engine.tokenizer
        # batch vs serial（E-B2 同款结构）：首分叉必为 near-tie（共享 ctx），随后双尾独立 teacher-force
        checked, total, diverged = [0], 0, 0
        for p, bout, sout in zip(PROMPTS, batch, serial):
            bt, st = bout["token_ids"], sout["token_ids"]
            ctx = tok.encode(p)
            for i, (x, y) in enumerate(zip(bt, st)):
                total += 1
                if x == y:
                    top1, gap = _ref_top1_gap(fp32, ctx)
                    if gap > GAP:                       # decisive 步：与 fp32 带窗参考强制一致
                        assert top1 == x
                        checked[0] += 1
                    ctx.append(x)
                else:                                   # 分叉只允许在 near-tie
                    diverged += 1
                    _, gap = _ref_top1_gap(fp32, ctx)   # 分叉点两路历史相同 → 共享 ctx
                    assert gap <= GAP, f"batch/serial diverged at decisive step {i} of {p!r}"
                    checked[0] += _teacher_force(fp32, ctx, bt[i:])          # 两路各自跟随自己的
                    checked[0] += _teacher_force(fp32, list(ctx), st[i:])    # token 流（分叉不级联）
                    break
        assert checked[0] / total > 0.8, f"decisive coverage too low: {checked[0]}/{total}, diverged={diverged}"
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()   # E-D5 需完整 VRAM 预算重建引擎


def test_ed5_graph_smoke(swa_model_dir):
    """E-D5：pools=2 + CUDA graph decode 输出与 eager 一致（三 SWA 变量进图）。"""
    if ED3_BATCH is None:
        pytest.skip("E-D3 未产出基准（前置失败）")
    engine = LLMEngine(swa_model_dir, enforce_eager=False, gpu_memory_utilization=0.5)
    try:
        out = engine.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
        assert [o["token_ids"] for o in out] == ED3_BATCH
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

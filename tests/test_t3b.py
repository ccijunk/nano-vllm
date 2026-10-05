"""T3b · LV1-b 引擎验证（DESIGN v1.1 §4）：E-B1..B4 + 接缝断言。

运行（C5 契约命令，容器内）：
docker run --rm --gpus all --network none -v "$PWD":/w -w /w/nano-vllm \
  -e PYTHONPATH=/w/nano-vllm -v "$PWD/.cache":/opt/cache -e HF_HOME=/opt/cache/hf \
  nano-stack/engine:vllm python3 -m pytest tests -q

E-B2 噪声对策（T2 两阶段判据同源教训）：bf16/批量形状噪声只可能翻转 near-tie 的
argmax，故 batch vs serial 的分叉只允许发生在参考实现 top1-top2 gap < 5e-2 的步；
decisive 步（gap ≥ 5e-2）强制与 fp32 参考一致（teacher-forced，无分叉级联）。
"""

import pytest
import torch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.layers.sampler import Sampler
from nanovllm.models.nano_model import NanoForCausalLM
from nanovllm.sampling_params import SamplingParams

from reference_model import CFG
from reference_model import forward as ref_forward

GAP = 5e-2  # decisive 步阈值：低于此值视为 bf16 噪声区，不判 argmax


@pytest.fixture(scope="module")
def engine(model_dir):
    # 单 engine 全模块共享：dist tcp://localhost:2333 只允许 init 一次；eager 路径（无 CUDA graph）
    engine = LLMEngine(model_dir, enforce_eager=True, gpu_memory_utilization=0.5)
    yield engine
    engine.exit()  # 销毁 dist 进程组：test_t3c 同进程再建 engine 需端口释放（exit 已幂等）


def _ref_top1_gap(fp32, ctx):
    logits = ref_forward(fp32, ctx, CFG)
    top2 = torch.topk(logits, 2)
    return top2.indices[0].item(), (top2.values[0] - top2.values[1]).item()


def _teacher_force(fp32, ctx: list[int], stream: list[int], checked: list[int]):
    """从公共前缀 ctx 出发，逐步校验 stream 中每个 decisive 步的 argmax。"""
    ctx = list(ctx)
    for tid in stream:
        top1, gap = _ref_top1_gap(fp32, ctx)
        if gap > GAP:
            assert top1 == tid
            checked[0] += 1
        ctx.append(tid)


def test_e_b1_weight_load_bitwise(engine, c1_weights):
    _, bf16 = c1_weights
    params = dict(engine.model_runner.model.named_parameters())
    H, KVH, D, I = CFG["n_heads"], CFG["n_kv_heads"], CFG["head_dim"], CFG["intermediate_size"]
    for key, w in bf16.items():
        if "q_proj" in key:
            pname, sl = key.replace("q_proj", "qkv_proj"), slice(0, H * D)
        elif "k_proj" in key:
            pname, sl = key.replace("k_proj", "qkv_proj"), slice(H * D, (H + KVH) * D)
        elif "v_proj" in key:
            pname, sl = key.replace("v_proj", "qkv_proj"), slice((H + KVH) * D, (H + 2 * KVH) * D)
        elif "gate_proj" in key:
            pname, sl = key.replace("gate_proj", "gate_up_proj"), slice(0, I)
        elif "up_proj" in key:
            pname, sl = key.replace("up_proj", "gate_up_proj"), slice(I, 2 * I)
        else:
            pname, sl = key, None
        p = params[pname].detach().cpu()
        assert torch.equal(p[sl] if sl is not None else p, w), key


def test_e_b2_greedy_batch_serial_reference(engine, c1_weights):
    fp32, _ = c1_weights
    tok = engine.tokenizer
    prompts = ["!#%&", "ABCDEFGH", "0123456789()", "*+-/:;<=>?@"]  # 全部落在 42 字符表内
    sp = SamplingParams(temperature=0, max_tokens=16)              # greedy：无采样随机性
    batch = engine.generate(prompts, sp, use_tqdm=False)
    serial = []
    for p in prompts:
        serial += engine.generate([p], sp, use_tqdm=False)

    checked, total, diverged = [0], 0, 0
    for p, bout, sout in zip(prompts, batch, serial):
        bt, st = bout["token_ids"], sout["token_ids"]
        ctx = tok.encode(p)
        for i, (x, y) in enumerate(zip(bt, st)):
            total += 1
            top1, gap = _ref_top1_gap(fp32, ctx)
            if x == y:
                if gap > GAP:                       # decisive 步：与 fp32 参考强制一致
                    assert top1 == x
                    checked[0] += 1
                ctx.append(x)
            else:                                   # 分叉只允许在 near-tie 步
                diverged += 1
                assert gap < GAP, f"batch/serial diverged at decisive step {i} of {p!r}"
                _teacher_force(fp32, ctx, bt[i:], checked)
                _teacher_force(fp32, list(ctx), st[i:], checked)
                break
    assert checked[0] / total > 0.8, f"decisive coverage too low: {checked[0]}/{total}, diverged={diverged}"


def test_e_b3_kv_accounting(engine, model_dir):
    mr = engine.model_runner
    cfg = mr.config.hf_config
    kv_org = NanoForCausalLM.kv_org
    assert kv_org.pools == 1 and kv_org.layout == "paged_kt_vt"
    # GQA 单池账面：2(K+V) × n_kv × head_dim × 2B = 256 B/token/层
    assert kv_org.per_token_bytes(cfg) == 2 * 2 * 32 * cfg.dtype.itemsize == 256
    itemsize = cfg.dtype.itemsize
    two, layers, blocks, blk, kvh, d = mr.kv_cache.shape
    assert (two, kvh, d) == (2, CFG["n_kv_heads"], CFG["head_dim"])
    half_bytes = mr.kv_cache[0].numel() * itemsize   # 单边（K 或 V）总字节
    assert half_bytes == kvh * d * itemsize * layers * blocks * blk


def test_e_b4_prefix_cache(engine):
    p1 = list(range(CFG["vocab_size"])) * 7 + list(range(6))          # 300 tokens
    p2 = p1 + list(range(8)) * 6 + list(range(2))                     # 350 tokens（+50，id 均 < 42）
    assert len(p1) == 300 and len(p2) == 350
    sp = SamplingParams(temperature=0, max_tokens=1)
    engine.add_request(p1, sp)
    while not engine.is_finished():
        engine.step()
    engine.add_request(p2, sp)
    seqs, is_prefill = engine.scheduler.schedule()                    # 只读探查 + 正常跑完
    assert seqs[0].num_cached_tokens == 256                           # block0 前缀命中
    token_ids = engine.model_runner.call("run", seqs, is_prefill)
    engine.scheduler.postprocess(seqs, token_ids, is_prefill)
    while not engine.is_finished():
        engine.step()


def test_seam_greedy_and_tier():
    # SamplingParams：temperature=0 合法（C2 greedy），负数拒绝
    assert SamplingParams(temperature=0.0).temperature == 0.0
    with pytest.raises(AssertionError):
        SamplingParams(temperature=-0.1)
    # Sampler：greedy 行 = 未缩放 argmax（div 0 短路），随机行落在词表内
    torch.manual_seed(0)
    logits = torch.randn(3, 7)
    out = Sampler()(logits, torch.tensor([0.0, 0.7, 0.0]), torch.ones(3))
    assert out[0].item() == logits[0].argmax().item()
    assert out[2].item() == logits[2].argmax().item()
    assert 0 <= out[1].item() < 7
    # BlockManager：tier 参数位（接口先行）；None = 零行为变化
    assert BlockManager(4, 256).tier is None

    class DummyTier:
        def put(self, hs, bs): ...
        def get(self, hs): return [None] * len(hs)
        def capacity_bytes(self): return 0

    dummy = DummyTier()
    assert BlockManager(4, 256, tier=dummy).tier is dummy

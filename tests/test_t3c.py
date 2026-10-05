"""T3c · C2 合规测试（E-C1..C3 + E-D1）。契约事实源 = doc/contracts/server_api.md。

E-C1 合规矩阵（4 端点 × 正常/异常 + 错误体四字段 + finish_reason + usage）
E-C2 同 seed 同输出（单请求形态，进程级 RNG，§6②）
E-C3 /metrics 可解析且 7 指标名/类型/标签符合 C2 schema
E-D1 并发吞吐 > 串行（continuous batching 生效）
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn
from prometheus_client.parser import text_string_to_metric_families

from nanovllm.server import EngineService, create_app

PORT = 8123
PROMPT = "!" * 4  # in-vocab：itos = chr(33+i), i ∈ [0, 42)


@pytest.fixture(scope="module")
def service(request):
    """LV1：本地 EngineService；LV2（NANO_BASE_URL 已设）：None——引擎在集群 Deployment pod
    （C5 R5 断言复用：同一份测试换个 base_url 打 Service，不另写一套）。"""
    if os.environ.get("NANO_BASE_URL"):
        yield None
        return
    model_dir = request.getfixturevalue("model_dir")
    service = EngineService(model_dir, enforce_eager=True, gpu_memory_utilization=0.5)
    yield service
    service.stop()
    service.engine.exit()  # 销毁 dist 进程组（exit 幂等，atexit 二次触发安全）


@pytest.fixture(scope="module")
def client(service):
    remote = os.environ.get("NANO_BASE_URL")
    if remote:
        with httpx.Client(base_url=remote.rstrip("/"), timeout=120.0) as c:
            yield c
        return
    app = create_app(service)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=120.0) as c:
        yield c
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def model_name(client):
    data = client.get("/v1/models").json()["data"]
    assert len(data) == 1
    return data[0]["id"]


def _completions(model_name, **overrides):
    body = {"model": model_name, "prompt": PROMPT, "max_tokens": 8, "temperature": 0}
    body.update(overrides)
    return body


# ---------- E-C1 · 合规矩阵 ----------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models(client, model_name):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    entry = body["data"][0]
    assert entry["object"] == "model"
    assert entry["id"] == model_name
    assert entry["owned_by"] == "nano-ai-infra-stack"
    assert isinstance(entry["created"], int)


def test_completions_ok(client, model_name):
    r = client.post("/v1/completions", json=_completions(model_name))
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert body["model"] == model_name
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "length"  # 无 EOS → 仅 length 触发（§6③）
    assert len(choice["text"]) == 8
    usage = body["usage"]
    assert usage["prompt_tokens"] == len(PROMPT)
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": None},  # 缺 prompt → 仅从 body 删字段实现
        {"max_tokens": 0},
        {"max_tokens": 257},
        {"temperature": 3.0},
        {"temperature": -0.1},
        {"top_p": 1.5},
        {"stream": True},  # 字段子集：未知字段拒绝
    ],
)
def test_validation_400(client, model_name, overrides):
    body = _completions(model_name)
    for k, v in overrides.items():
        if v is None:
            body.pop(k)
        else:
            body[k] = v
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 400
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"}
    assert err["code"] is None
    assert err["type"] == "invalid_request_error"


def test_model_not_found(client):
    r = client.post("/v1/completions", json=_completions("wrong-model"))
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found"
    assert err["type"] == "invalid_request_error"


def test_context_length_422(client, model_name):
    r = client.post(
        "/v1/completions",
        json={"model": model_name, "prompt": "!" * 1000, "max_tokens": 64},  # 1064 > 1024
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "context_length_exceeded"


def test_engine_overloaded_503(client, model_name, service):
    if service is None:
        # LV2 集群模式：无法内省远端 service 注入排队上限；503 语义已在 LV1 验证（记录 engine/001）
        pytest.skip("503 trigger requires local service introspection (LV1-only)")
    service.max_waiting_seqs = 0
    try:
        r = client.post("/v1/completions", json=_completions(model_name))
        assert r.status_code == 503
        err = r.json()["error"]
        assert err["code"] == "engine_overloaded"
    finally:
        service.max_waiting_seqs = 8
    assert client.post("/v1/completions", json=_completions(model_name)).status_code == 200


# ---------- E-C2 · 同 seed 同输出 ----------

def test_seed_replay(client, model_name):
    body = _completions(model_name, max_tokens=16, temperature=1.0, seed=42)
    t1 = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    t2 = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    assert t1 == t2  # char 词表 decode ⇔ token ids bitwise 相等


def test_top_p_zero_is_deterministic(client, model_name):
    # nucleus 路径行为锚：top_p=0 仅保留最高概率 token → 等效贪心，两次一致
    body = _completions(model_name, max_tokens=16, temperature=1.0, top_p=0.0)
    t1 = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    t2 = client.post("/v1/completions", json=body).json()["choices"][0]["text"]
    assert t1 == t2


# ---------- E-C3 · /metrics schema ----------

EXPECTED_METRICS = {
    "nano_request_total": "counter",
    "nano_request_prompt_tokens": "counter",
    "nano_request_completion_tokens": "counter",
    "nano_ttft_seconds": "histogram",
    "nano_tpot_seconds": "histogram",
    "nano_queue_depth": "gauge",
    "nano_running_batch": "gauge",
    "nano_kv_cache_utilization": "gauge",
}


def test_metrics_schema(client, model_name):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    families = {f.name: f for f in text_string_to_metric_families(r.text)}

    def family_of(name, type_):
        # prometheus_client 对 counter 家族名剥 _total 后缀（样本名保留）：
        # Counter("nano_request_total") → family "nano_request"，sample "nano_request_total"
        key = name[:-6] if type_ == "counter" and name.endswith("_total") else name
        assert key in families, f"missing metric {name} (family {key})"
        assert families[key].type == type_, f"{name}: want {type_}, got {families[key].type}"
        return families[key]

    for name, type_ in EXPECTED_METRICS.items():
        family = family_of(name, type_)
        samples = [s for s in family.samples if not s.name.endswith("_created")]
        assert samples, f"{name} has no samples"
        for sample in samples:
            assert sample.labels.get("engine") == "nano-vllm"
            assert sample.labels.get("model") == model_name
    # histogram 结构完整性
    ttft = family_of("nano_ttft_seconds", "histogram")
    assert any(s.name.endswith("_bucket") for s in ttft.samples)
    assert any(s.name.endswith("_sum") for s in ttft.samples)
    assert any(s.name.endswith("_count") for s in ttft.samples)


# ---------- E-D1 · 并发吞吐 > 串行 ----------

def test_concurrent_beats_serial(client, model_name):
    body = {"model": model_name, "prompt": "!" * 40, "max_tokens": 64, "temperature": 0}
    assert client.post("/v1/completions", json=body).status_code == 200  # warmup

    n = 8
    barrier = threading.Barrier(n)

    def one(_):
        barrier.wait()
        r = client.post("/v1/completions", json=body)
        assert r.status_code == 200
        return r

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(one, range(n)))
    t_concurrent = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        assert client.post("/v1/completions", json=body).status_code == 200
    t_serial = time.perf_counter() - t0

    assert t_concurrent < t_serial, f"concurrent {t_concurrent:.2f}s >= serial {t_serial:.2f}s"

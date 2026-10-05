"""T3c · C2 server：OpenAI 兼容 /v1/completions + /v1/models + /health + /metrics。

设计要点（doc/topics/t3_nano_vllm.md §3 T3c）：
- 执行模型 = FastAPI handler 入队 → 引擎线程 step 循环（continuous batching）→ 每请求 Event 回填；
  引擎核心（scheduler/block_manager/runner）零改动，全部包装在本层（C2 H4：引擎保持请求作用域）。
- 错误语义：400 字段校验 / 404 model_not_found / 422 context_length_exceeded / 503 engine_overloaded。
- metrics：counter/histogram 在本层记；gauge 每次抓取时从 scheduler/block_manager 现读。
- seed：请求带 seed → 准入时 torch.manual_seed（进程级；并发下不保证 per-request 重放，§6②）。
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from time import perf_counter

import torch
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

ENGINE_LABEL = "nano-vllm"

# C2 §metrics：名称/类型/标签为契约（公共标签 engine + model；基数变更 = 破坏性）
REQUEST_TOTAL = Counter("nano_request_total", "C2 request count", ["engine", "model", "status"])
PROMPT_TOKENS = Counter("nano_request_prompt_tokens", "C2 prompt token throughput", ["engine", "model"])
COMPLETION_TOKENS = Counter("nano_request_completion_tokens", "C2 completion token throughput", ["engine", "model"])
TTFT = Histogram("nano_ttft_seconds", "C2 time to first token", ["engine", "model"])
TPOT = Histogram("nano_tpot_seconds", "C2 time per output token", ["engine", "model"])
QUEUE_DEPTH = Gauge("nano_queue_depth", "C2 queued requests", ["engine", "model"])
RUNNING_BATCH = Gauge("nano_running_batch", "C2 requests in running batch", ["engine", "model"])
KV_UTIL = Gauge("nano_kv_cache_utilization", "C2 KV pool occupancy 0..1", ["engine", "model"])


def error_body(message: str, type_: str = "invalid_request_error", param: str | None = None, code: str | None = None):
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


class OverloadedError(Exception):
    pass


class CompletionRequest(BaseModel):
    # C2 §completions 请求字段 = 契约；未知字段拒绝（字段子集语义）
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = None


@dataclass
class _Entry:
    seq: Sequence
    enqueue_t: float
    first_t: float | None = None
    error: Exception | None = None
    event: threading.Event = field(default_factory=threading.Event)


class EngineService:
    """引擎包装：后台 step 循环 + 请求准入 + 指标记录。"""

    def __init__(self, model: str, **kwargs):
        self.max_waiting_seqs = kwargs.pop("max_waiting_seqs", 8)
        self.engine = LLMEngine(model, **kwargs)
        self.model_name = os.path.basename(model.rstrip("/")) or "nano-model"
        self.created = int(time.time())
        self._pending: dict[int, _Entry] = {}
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="engine-loop")
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)

    def submit(self, prompt_ids: list[int], req: CompletionRequest) -> _Entry:
        with self._lock:
            if len(self.engine.scheduler.waiting) >= self.max_waiting_seqs:
                raise OverloadedError("engine queue is full")
            if req.seed is not None:
                # 进程级 seed：单请求形态可重放；并发批量下不保证 per-request 重放（设计 §6②）
                torch.manual_seed(req.seed)
            sp = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens)
            seq = Sequence(prompt_ids, sp)
            self._pending[seq.seq_id] = _Entry(seq=seq, enqueue_t=perf_counter())
            self.engine.scheduler.add(seq)
        return self._pending[seq.seq_id]

    def _loop(self):
        while not self._stop:
            if self.engine.scheduler.is_finished():
                time.sleep(0.0005)
                continue
            try:
                self.engine.step()
            except Exception as e:  # 引擎步进失败：fail-fast 所有在途请求
                with self._lock:
                    for entry in self._pending.values():
                        entry.error = e
                        entry.event.set()
                    self._pending.clear()
                continue
            now = perf_counter()
            with self._lock:
                for entry in list(self._pending.values()):
                    seq = entry.seq
                    if entry.first_t is None and seq.num_completion_tokens >= 1:
                        entry.first_t = now
                        TTFT.labels(ENGINE_LABEL, self.model_name).observe(now - entry.enqueue_t)
                    if seq.is_finished:
                        n = seq.num_completion_tokens
                        if n > 1:
                            TPOT.labels(ENGINE_LABEL, self.model_name).observe((now - entry.first_t) / (n - 1))
                        PROMPT_TOKENS.labels(ENGINE_LABEL, self.model_name).inc(seq.num_prompt_tokens)
                        COMPLETION_TOKENS.labels(ENGINE_LABEL, self.model_name).inc(n)
                        REQUEST_TOTAL.labels(ENGINE_LABEL, self.model_name, "ok").inc()
                        del self._pending[seq.seq_id]
                        entry.event.set()

    def snapshot(self) -> dict:
        scheduler = self.engine.scheduler
        block_manager = scheduler.block_manager
        num_blocks = len(block_manager.used_block_ids) + len(block_manager.free_block_ids)
        return {
            "queue_depth": len(scheduler.waiting),
            "running_batch": len(scheduler.running),
            "kv_utilization": len(block_manager.used_block_ids) / num_blocks,
        }


def create_app(service: EngineService) -> FastAPI:
    app = FastAPI(title="nano-vllm", docs_url=None, redoc_url=None, openapi_url=None)
    max_position = service.engine.config.hf_config.max_position_embeddings

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc: RequestValidationError):
        REQUEST_TOTAL.labels(ENGINE_LABEL, service.model_name, "error").inc()
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        return JSONResponse(status_code=400, content=error_body(f"{loc}: {first['msg']}", param=loc or None))

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if req.model != service.model_name:
            REQUEST_TOTAL.labels(ENGINE_LABEL, service.model_name, "error").inc()
            return JSONResponse(
                status_code=404,
                content=error_body(f"The model `{req.model}` does not exist", code="model_not_found"),
            )
        try:
            prompt_ids = service.engine.tokenizer.encode(req.prompt)
        except KeyError as e:
            return JSONResponse(
                status_code=400,
                content=error_body(f"prompt contains out-of-vocab character: {e}", param="prompt"),
            )
        if len(prompt_ids) + req.max_tokens > max_position:
            REQUEST_TOTAL.labels(ENGINE_LABEL, service.model_name, "error").inc()
            return JSONResponse(
                status_code=422,
                content=error_body(
                    f"This model's maximum context length is {max_position} tokens, "
                    f"however you requested {len(prompt_ids) + req.max_tokens} tokens "
                    f"({len(prompt_ids)} in the prompt, {req.max_tokens} for the completion)",
                    code="context_length_exceeded",
                ),
            )
        try:
            entry = service.submit(prompt_ids, req)
        except OverloadedError:
            REQUEST_TOTAL.labels(ENGINE_LABEL, service.model_name, "error").inc()
            return JSONResponse(
                status_code=503,
                content=error_body("engine overloaded, retry later", type_="server_error", code="engine_overloaded"),
            )
        entry.event.wait()
        if entry.error is not None:
            REQUEST_TOTAL.labels(ENGINE_LABEL, service.model_name, "error").inc()
            raise RuntimeError(f"engine step failed: {entry.error}") from entry.error
        seq = entry.seq
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": service.model_name,
            # char 模型无 EOS（config.eos=-1 永不匹配）→ 仅 "length" 触发；"stop" 分支按契约保留
            "choices": [{
                "index": 0,
                "text": service.engine.tokenizer.decode(seq.completion_token_ids),
                "finish_reason": "length" if seq.num_completion_tokens >= seq.max_tokens else "stop",
            }],
            "usage": {
                "prompt_tokens": seq.num_prompt_tokens,
                "completion_tokens": seq.num_completion_tokens,
                "total_tokens": seq.num_tokens,
            },
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{
                "id": service.model_name,
                "object": "model",
                "created": service.created,
                "owned_by": "nano-ai-infra-stack",
            }],
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        # gauge 语义 = 抓取时刻现读（设计 §3）
        snap = service.snapshot()
        engine, model = ENGINE_LABEL, service.model_name
        QUEUE_DEPTH.labels(engine, model).set(snap["queue_depth"])
        RUNNING_BATCH.labels(engine, model).set(snap["running_batch"])
        KV_UTIL.labels(engine, model).set(snap["kv_utilization"])
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


if __name__ == "__main__":
    # LV2 Deployment 入口：python3 -m nanovllm.server /data/model --port 8000
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="nano-vllm C2 server")
    parser.add_argument("model", help="C1 侧车模型目录")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-waiting-seqs", type=int, default=8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    _service = EngineService(
        args.model,
        max_waiting_seqs=args.max_waiting_seqs,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    uvicorn.run(create_app(_service), host=args.host, port=args.port, log_level="info")

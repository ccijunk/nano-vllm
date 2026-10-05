# T3a · L2 engine:vllm（C4 §3 清单 + §4.1 flash-attn 零编译策略；构建上下文 = stack 根）
# 自检（C4 §6）：docker run --rm nano-stack/engine:vllm python3 -c "import flash_attn; from nanovllm import LLM"
# triton launcher 编译链（gcc）需 GPU launch 才触发，build 时测不了 → 真正守卫 = LV1-b pytest。
FROM nano-stack/base:torch-cu13-py312

# 工具链层贴 FROM（ARG 声明之下无缓存断点：改 pip 参数/代码都不会打掉本层，只装一次）。
# 依赖事实（E-B1 实测 + 依赖分析）：engine 运行期必编 triton C launcher——手写 @triton.jit
# store_kvcache（attention.py）每步都走，@torch.compile（RMSNorm/Rotary/Silu）的 kernel 同样要；
# launcher 是 CPython C 扩展 → gcc（编 C）+ python3.12-dev（Python.h）。
# 只装 C 编译器不装 g++：g++ 仅供 inductor CPU codegen（C++ 路径），engine 张量全 CUDA 不触发。
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG ASTRAL_CU130=https://wheels.astral.sh/simple/cu130/

# 通用依赖走 TUNA（einops = flash-attn 运行期依赖，随 --no-deps 拆开单独装）；
# fastapi/uvicorn/prometheus_client = C2 server 依赖（C4 §3 清单外新增层，各线 L2 加层权限内）。
# --retries/--timeout：跨网链路默认 15s 超时必失败（root/012 同款教训）。
RUN pip install --break-system-packages --index-url ${PIP_INDEX} \
      --retries 10 --timeout 180 \
      "transformers>=4.56.0,<=4.57.3" xxhash einops \
      fastapi "uvicorn[standard]" prometheus_client httpx

# flash-attn 预编译 wheel（Astral cu130 index，local version 全串 pin —— 只写 2.8.3.post1 时 pip
# 会自选 torch.2.14 变体，与 base 的 torch 2.9.1 c10 ABI 不匹配（E-A1 实测 undefined symbol）；C4 §2 pin 依据）
RUN pip install --break-system-packages --no-deps --index-url ${ASTRAL_CU130} \
      flash-attn==2.8.3.post1+cu.13.0.torch.2.9

# 组件代码后置 COPY（代码改动只重建这两层，依赖层走 cache，C4 R1）
COPY nano-vllm/ /opt/nano-vllm/
RUN pip install --break-system-packages --no-deps /opt/nano-vllm

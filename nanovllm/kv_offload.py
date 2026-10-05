"""Offline KV 层接口（nano-LMCache 同构）——接口先行，主线不实现。

DESIGN v1.1 §6 偏差 9（用户裁决 2026-10-04）：本轮只冻结 Protocol 与 BlockManager
的 tier 参数位；任何具体 tier（CPU/host tier 等）留待后续 topic，tier=None 时行为
与现状完全一致（零行为变化）。

接缝语义（实现方必须遵守，当前仅约定）：
- preempt 时若 tier 存在则 put 落盘，否则按现状直接释放；
- 重调度时 get 命中则恢复块内容，未命中返回 None 并回退 recompute。

生产锚点（DESIGN §5）：
- vLLM ``KVConnectorBase_V1``：scheduler/worker 两侧接口 + 逐层异步 put/get +
  handle_preemptions；lmcache / offloading / simple_cpu_offload 均为其实现。
- SGLang ``HiCache``：hiradix_cache 驱逐 + memory_pool_host 页搬运 +
  hicache_storage 后端。
nano 取三者最小公分母：整块粒度、同步语义；有意丢弃逐层流式、两侧分离、
外部存储后端与事件流。
"""

from typing import Any, Protocol


class KVOffloadTier(Protocol):
    """offline 层协议：以块（block）为粒度的 put/get/capacity。"""

    def put(self, block_hashes: list[int], blocks: list[Any]) -> None:
        """GPU → offline：按块 hash 把整块 KV 拷贝到 offline 存储。

        blocks 的具体形态（布局/设备）由 BlockManager 与 tier 的约定决定，
        nano 阶段固定为 KVOrg.layout（paged_kt_vt）下的整块张量视图。
        """
        ...

    def get(self, block_hashes: list[int]) -> list[Any | None]:
        """offline → GPU：按块 hash 取回块数据；未命中位为 None（调用方回退 recompute）。"""
        ...

    def capacity_bytes(self) -> int:
        """offline 层总容量（字节），供驱逐/准入决策。"""
        ...

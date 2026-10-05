from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.kv_offload import KVOffloadTier


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """T3d 两池显式形态（doc/topics/t3d_swa_multipool.md §2.1，决策③）：

    - pool 0（全池）：blocks/free/hash_to_block_id，操作 seq.block_table，prefix cache 语义
    - pool 1（SWA）：*_swa 平行字段，操作 seq.swa_block_table，无 hash（决策②），
      块滑出窗口即回收（设计 §2.1——本类的存在理由）
    - window == 0 ⟺ 单池形态，全部 swa 方法短路（22 测试回归锚）
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_blocks_swa: int = 0, window: int = 0,
                 tier: KVOffloadTier | None = None):
        self.block_size = block_size
        self.tier = tier  # offline KV 接缝（接口先行）：None = 现状，零行为变化
        self.window = window
        # ---- pool 0（全池）----
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # ---- pool 1（SWA 池）----
        self.blocks_swa: list[Block] = [Block(i) for i in range(num_blocks_swa)]
        self.free_block_ids_swa: deque[int] = deque(range(num_blocks_swa))
        self.used_block_ids_swa: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _allocate_block_swa(self) -> int:
        block_id = self.free_block_ids_swa.popleft()
        block = self.blocks_swa[block_id]
        assert block.ref_count == 0
        block.reset()
        self.used_block_ids_swa.add(block_id)
        return block_id

    def _deallocate_block_swa(self, block_id: int):
        assert self.blocks_swa[block_id].ref_count == 0
        self.used_block_ids_swa.remove(block_id)
        self.free_block_ids_swa.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        if self.window:    # T3d 决策②实现期扩展：SWA 模型整体禁用 prefix cache——
            # 调度层单一 can_allocate 无法混组（全池 cacheable × SWA 池窗口键不可回放）；
            # vLLM 同构 = SlidingWindowSpec.prefix_cacheable=False
            if len(self.free_block_ids) < seq.num_blocks \
                    or len(self.free_block_ids_swa) < seq.swa_num_blocks(self.window):
                return -1
            return 0
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        if self.window and len(self.free_block_ids_swa) < seq.swa_num_blocks(self.window):
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size
        # SWA 池：按 prompt 终态窗口分配可见块（无前缀缓存，全部新块）
        if self.window:
            for _ in range(seq.swa_num_blocks(self.window)):
                seq.swa_block_table.append(self._allocate_block_swa())
            seq.swa_table_start = seq.swa_window_start(self.window)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        for block_id in reversed(seq.swa_block_table):
            block = self.blocks_swa[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block_swa(block_id)
        seq.swa_block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        if len(self.free_block_ids) < (len(seq) % self.block_size == 1):
            return False
        if not self.window:
            return True
        # SWA：可见块目标数与现表差额（may_append 先裁剪后补足，裁剪只释放，故净需求即此值）
        return len(self.free_block_ids_swa) >= max(0, seq.swa_num_blocks(self.window) - len(seq.swa_block_table))

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
        if not self.window:
            return
        bs = self.block_size
        start = seq.swa_window_start(self.window)   # 本步 L（含写入 token）的窗口起点，块对齐
        # 裁剪：弹出完全低于 start 的表首块（swa_table_start 为块对齐的表首绝对 token 位）
        while seq.swa_block_table and seq.swa_table_start + bs <= start:
            block_id = seq.swa_block_table.pop(0)
            block = self.blocks_swa[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block_swa(block_id)
            seq.swa_table_start += bs
        # 补足：表尾覆盖到当前长度 L（含本步写入 token）
        while seq.swa_table_start + len(seq.swa_block_table) * bs < seq.num_tokens:
            seq.swa_block_table.append(self._allocate_block_swa())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

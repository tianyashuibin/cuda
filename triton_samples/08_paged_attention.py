"""
Step 8: Paged Attention (vLLM 风格简化版)
==========================================
LLM Serving 的核心算子 — 理解 vLLM 如何高效管理 KV Cache

背景：为什么需要 Paged Attention？
  在 LLM 推理（尤其是 serving 场景）中，每个请求都有自己的 KV Cache：
  - Prefill 阶段：一次性计算所有 prompt token 的 KV，写入 cache
  - Decode 阶段：每步生成一个 token，读取整个 KV Cache 做 attention

  传统做法的问题：
  - 每个请求预分配 max_seq_len 大小的连续内存 → 严重的内存浪费
  - 不同请求的序列长度不同 → 内存碎片化
  - 无法动态调度 → 吞吐量受限

  vLLM 的解决方案 — Paged Attention：
  - 借鉴操作系统的虚拟内存/分页思想
  - KV Cache 被切分成固定大小的 "page"（通常 16 个 token 一页）
  - 每个请求维护一个 page table（逻辑页 → 物理页的映射）
  - 物理页可以不连续，按需分配/回收 → 内存利用率接近 100%

  本文件实现：
  - Paged KV Cache 的数据结构和管理
  - Paged Attention decode kernel（单 query token attend 到 paged KV）
  - 与标准连续内存 attention 的正确性对比和性能对比

  简化点（vs vLLM 完整实现）：
  - 不含 prefill kernel（prefill 直接用 flash attention 更优）
  - 不含 copy-on-write、prefix caching 等高级特性
  - 单 batch（不含 ragged batch / padding-free batching）
  - 重点在于理解 page table 间接寻址 + attention 计算的融合
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# Paged Attention Decode Kernel
# ============================================================

@triton.jit
def paged_attention_kernel(
    Out,          # 输出 [num_heads, head_dim]
    Q,            # Query [num_heads, head_dim]  (decode: 只有 1 个 token)
    K_cache,      # KV Cache (Key)  [num_blocks, page_size, num_kv_heads, head_dim]
    V_cache,      # KV Cache (Value) [num_blocks, page_size, num_kv_heads, head_dim]
    Page_table,   # 页表 [max_num_pages] — 逻辑页号 → 物理 block 号
    seq_len,      # 当前序列的实际长度（决定要读多少个 KV token）
    sm_scale,     # 1/sqrt(head_dim)
    stride_kb, stride_kp, stride_kh, stride_kd,  # K_cache strides
    stride_vb, stride_vp, stride_vh, stride_vd,  # V_cache strides
    num_kv_heads,
    PAGE_SIZE: tl.constexpr,     # 每页的 token 数（通常 16）
    HEAD_DIM: tl.constexpr,      # head dimension
    BLOCK_KV: tl.constexpr,      # 每次处理的 KV token 数（分块大小）
):
    """
    Paged Attention (Decode Phase)

    每个 program 处理一个 attention head:
      output[h] = softmax(Q[h] @ K_cache[all_pages, h]^T / sqrt(d)) @ V_cache[all_pages, h]

    核心挑战：K/V 不是连续存储的，需要通过 page table 间接寻址
    """
    head_idx = tl.program_id(0)

    # GQA 支持：多个 Q head 共享一个 KV head
    kv_head_idx = head_idx % num_kv_heads

    # 加载 Query [HEAD_DIM]
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + head_idx * HEAD_DIM + offs_d).to(tl.float32)

    # ---- Online Softmax + Paged KV 遍历 ----
    # 与 Flash Attention 相同的 online softmax 技巧
    m_i = float("-inf")           # running max
    l_i = 0.0                     # running sum of exp
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # 累加器

    num_pages = tl.cdiv(seq_len, PAGE_SIZE)

    # 逐页遍历 KV Cache
    for page_idx in range(0, num_pages):
        # ---- Page Table 间接寻址 ----
        # 这是 Paged Attention 与普通 Attention 的核心区别：
        # 普通 attention: K[token_idx] → 直接内存访问
        # Paged attention: token_idx → 逻辑页 → page_table → 物理 block → K[block][slot]
        physical_block = tl.load(Page_table + page_idx)

        # 当前页中有效的 token 数
        page_start = page_idx * PAGE_SIZE
        page_tokens = tl.minimum(PAGE_SIZE, seq_len - page_start)

        # 遍历当前页内的 token（按 BLOCK_KV 分块，通常 BLOCK_KV == PAGE_SIZE）
        offs_p = tl.arange(0, BLOCK_KV)
        mask_p = offs_p < page_tokens

        # 计算 K 在物理内存中的地址
        # K_cache layout: [num_blocks, page_size, num_kv_heads, head_dim]
        k_base = (physical_block * stride_kb +
                  offs_p[:, None] * stride_kp +
                  kv_head_idx * stride_kh +
                  offs_d[None, :] * stride_kd)
        k = tl.load(K_cache + k_base, mask=mask_p[:, None], other=0.0).to(tl.float32)

        # QK^T: [BLOCK_KV] = Q[d] @ K[BLOCK_KV, d]^T
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_KV]
        qk = qk * sm_scale

        # 用 -inf mask 无效位置
        qk = tl.where(mask_p, qk, float("-inf"))

        # ---- Online Softmax Update ----
        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = alpha * acc

        # 加载 V 并累加
        v_base = (physical_block * stride_vb +
                  offs_p[:, None] * stride_vp +
                  kv_head_idx * stride_vh +
                  offs_d[None, :] * stride_vd)
        v = tl.load(V_cache + v_base, mask=mask_p[:, None], other=0.0).to(tl.float32)

        # acc += p @ V: [d] += [BLOCK_KV] @ [BLOCK_KV, d]
        acc += tl.sum(p[:, None] * v, axis=0)

        m_i = m_new

    # 最终归一化
    acc = acc / l_i

    # 写回
    tl.store(Out + head_idx * HEAD_DIM + offs_d, acc.to(Out.dtype.element_ty))


# ============================================================
# Paged KV Cache Manager
# ============================================================

class PagedKVCache:
    """
    简化版 Paged KV Cache 管理器

    内存布局:
      K_cache: [num_blocks, page_size, num_kv_heads, head_dim]
      V_cache: [num_blocks, page_size, num_kv_heads, head_dim]

    每个请求维护一个 page_table: [逻辑页号] → 物理 block 号
    """
    def __init__(self, num_blocks, page_size, num_kv_heads, head_dim, dtype=torch.float16, device='cuda'):
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 物理 KV Cache 存储池
        self.k_cache = torch.zeros(
            num_blocks, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_blocks, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )

        # 空闲 block 列表（模拟 OS 的 free page list）
        self.free_blocks = list(range(num_blocks))

    def allocate_page(self):
        """分配一个物理 block"""
        if not self.free_blocks:
            raise RuntimeError("KV Cache 已满！（类比 OS: Out of Memory）")
        return self.free_blocks.pop(0)

    def free_page(self, block_idx):
        """回收一个物理 block"""
        self.free_blocks.append(block_idx)

    def write_kv(self, page_table, seq_pos, k, v):
        """
        写入一个 token 的 KV 到 cache

        Args:
            page_table: 当前请求的页表 list
            seq_pos: token 在序列中的位置
            k: [num_kv_heads, head_dim]
            v: [num_kv_heads, head_dim]
        """
        logical_page = seq_pos // self.page_size
        slot_in_page = seq_pos % self.page_size

        # 如果需要新页，分配之
        while logical_page >= len(page_table):
            page_table.append(self.allocate_page())

        physical_block = page_table[logical_page]
        self.k_cache[physical_block, slot_in_page] = k
        self.v_cache[physical_block, slot_in_page] = v

    def write_kv_batch(self, page_table, keys, values):
        """
        批量写入 KV（用于 prefill）

        Args:
            page_table: 当前请求的页表 list（会被原地修改）
            keys:  [seq_len, num_kv_heads, head_dim]
            values: [seq_len, num_kv_heads, head_dim]
        """
        seq_len = keys.shape[0]
        for i in range(seq_len):
            self.write_kv(page_table, i, keys[i], values[i])


# ============================================================
# Python wrapper
# ============================================================

def paged_attention(
    query,          # [num_heads, head_dim]
    kv_cache,       # PagedKVCache 对象
    page_table,     # list[int] 逻辑页 → 物理 block
    seq_len,        # 实际序列长度
):
    """
    Paged Attention (Decode)

    Args:
        query: [num_heads, head_dim] — 当前 decode step 的 query
        kv_cache: PagedKVCache 对象
        page_table: 页表映射
        seq_len: 当前序列长度
    Returns:
        output: [num_heads, head_dim]
    """
    num_heads, head_dim = query.shape
    num_kv_heads = kv_cache.num_kv_heads
    page_size = kv_cache.page_size

    output = torch.empty_like(query)

    # 页表转 tensor
    page_table_tensor = torch.tensor(page_table, dtype=torch.int32, device=query.device)

    sm_scale = 1.0 / math.sqrt(head_dim)

    grid = (num_heads,)

    paged_attention_kernel[grid](
        output, query,
        kv_cache.k_cache, kv_cache.v_cache,
        page_table_tensor,
        seq_len, sm_scale,
        kv_cache.k_cache.stride(0), kv_cache.k_cache.stride(1),
        kv_cache.k_cache.stride(2), kv_cache.k_cache.stride(3),
        kv_cache.v_cache.stride(0), kv_cache.v_cache.stride(1),
        kv_cache.v_cache.stride(2), kv_cache.v_cache.stride(3),
        num_kv_heads,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_KV=page_size,
    )

    return output


# ============================================================
# Reference Implementation (连续内存标准 attention)
# ============================================================

def reference_attention(query, keys, values):
    """
    标准 attention（用于正确性验证）

    Args:
        query:  [num_heads, head_dim]
        keys:   [seq_len, num_kv_heads, head_dim]
        values: [seq_len, num_kv_heads, head_dim]
    Returns:
        output: [num_heads, head_dim]
    """
    num_heads, head_dim = query.shape
    num_kv_heads = keys.shape[1]
    heads_per_kv = num_heads // num_kv_heads

    scale = 1.0 / math.sqrt(head_dim)
    outputs = []

    for h in range(num_heads):
        kv_h = h % num_kv_heads
        q = query[h].float()                  # [d]
        k = keys[:, kv_h, :].float()          # [seq_len, d]
        v = values[:, kv_h, :].float()        # [seq_len, d]

        attn = torch.matmul(k, q) * scale     # [seq_len]
        attn = torch.softmax(attn, dim=0)     # [seq_len]
        out = torch.matmul(attn.unsqueeze(0), v).squeeze(0)  # [d]
        outputs.append(out)

    return torch.stack(outputs).to(query.dtype)


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    torch.manual_seed(0)

    configs = [
        # (seq_len, num_heads, num_kv_heads, head_dim, page_size)
        (256,  32, 32, 128, 16),    # MHA, 短序列
        (1024, 32, 32, 128, 16),    # MHA, 中序列
        (4096, 32, 32, 128, 16),    # MHA, 长序列
        (1024, 32, 8,  128, 16),    # GQA (4:1), LLaMA-2 风格
        (4096, 32, 8,  128, 16),    # GQA, 长序列
    ]

    gqa_str = lambda nh, nkv: "MHA" if nh == nkv else f"GQA({nh//nkv}:1)"

    print(f"{'Config':>30} | {'Standard (ms)':>13} | {'Paged (ms)':>11} | {'Overhead':>9}")
    print("-" * 75)

    for seq_len, num_heads, num_kv_heads, head_dim, page_size in configs:
        num_blocks = (seq_len // page_size) + 32  # 多分配一些 block

        # 准备数据
        query = torch.randn(num_heads, head_dim, device='cuda', dtype=torch.float16)
        keys = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        values = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)

        # 初始化 paged KV cache
        kv_cache = PagedKVCache(num_blocks, page_size, num_kv_heads, head_dim, device='cuda')
        page_table = []
        kv_cache.write_kv_batch(page_table, keys, values)

        # 标准 attention 的 benchmark
        # 把 keys/values reshape 成连续 layout 做 matmul
        def standard_attn():
            scale = 1.0 / math.sqrt(head_dim)
            # [num_heads, 1, d] @ [num_heads, d, seq_len] → [num_heads, 1, seq_len]
            k_expand = keys.permute(1, 0, 2)  # [num_kv_heads, seq_len, d]
            v_expand = values.permute(1, 0, 2)
            outs = []
            for h in range(num_heads):
                kv_h = h % num_kv_heads
                s = torch.matmul(query[h:h+1], k_expand[kv_h].T) * scale
                p = torch.softmax(s, dim=-1)
                outs.append(torch.matmul(p, v_expand[kv_h]))
            return torch.cat(outs, dim=0)

        # Paged attention benchmark
        def paged_attn():
            return paged_attention(query, kv_cache, page_table, seq_len)

        ms_standard = triton.testing.do_bench(standard_attn)
        ms_paged = triton.testing.do_bench(paged_attn)

        overhead = ms_paged / ms_standard
        config_str = f"seq={seq_len} {gqa_str(num_heads, num_kv_heads)} d={head_dim}"
        print(f"{config_str:>30} | {ms_standard:>13.4f} | {ms_paged:>11.4f} | {overhead:>8.2f}x")

    print()
    print("注意：Paged Attention 的优势不在于单次 kernel 速度（间接寻址有开销），")
    print("而在于 serving 场景下的内存利用率（减少碎片 → 更大 batch → 更高吞吐）。")


if __name__ == "__main__":
    print("=" * 75)
    print("Step 8: Paged Attention (vLLM 风格)")
    print("=" * 75)

    # ---- 核心概念演示 ----
    print("\n--- 核心概念：分页 KV Cache ---")
    PAGE_SIZE = 16
    NUM_BLOCKS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    SEQ_LEN = 50

    kv_cache = PagedKVCache(NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device='cuda')
    print(f"KV Cache 池: {NUM_BLOCKS} blocks × {PAGE_SIZE} tokens/block = {NUM_BLOCKS * PAGE_SIZE} token 容量")
    print(f"每个 block: [{PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}] = {PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM * 2 / 1024:.1f} KB (FP16)")
    print(f"总显存: {kv_cache.k_cache.nelement() * 2 * 2 / 1024 / 1024:.1f} MB (K+V)")

    # 模拟一个请求的 KV Cache 写入
    keys = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device='cuda', dtype=torch.float16)
    values = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device='cuda', dtype=torch.float16)

    page_table = []  # 空页表
    kv_cache.write_kv_batch(page_table, keys, values)

    num_pages = len(page_table)
    print(f"\n请求: seq_len={SEQ_LEN}")
    print(f"分配了 {num_pages} 个物理 block (逻辑页)")
    print(f"页表: {page_table}")
    print(f"  → 逻辑页 0 → 物理 block {page_table[0]} (token 0-{PAGE_SIZE-1})")
    print(f"  → 逻辑页 1 → 物理 block {page_table[1]} (token {PAGE_SIZE}-{2*PAGE_SIZE-1})")
    if num_pages > 2:
        last_page_tokens = SEQ_LEN - (num_pages - 1) * PAGE_SIZE
        print(f"  → 逻辑页 {num_pages-1} → 物理 block {page_table[-1]} (token {(num_pages-1)*PAGE_SIZE}-{SEQ_LEN-1}, 部分填充 {last_page_tokens}/{PAGE_SIZE})")
    print(f"内存利用率: {SEQ_LEN / (num_pages * PAGE_SIZE) * 100:.1f}% (最后一页可能有空洞)")
    print(f"对比连续分配: 如果预分配 max_seq_len=2048, 利用率仅 {SEQ_LEN / 2048 * 100:.1f}%")

    # ---- 正确性验证 ----
    print("\n--- 正确性验证 ---")
    NUM_HEADS = 32

    test_configs = [
        (64, 32, 32, 64, 16, "MHA, short"),
        (128, 32, 32, 128, 16, "MHA, d=128"),
        (200, 32, 8, 128, 16, "GQA 4:1"),
        (50, 16, 4, 64, 16, "GQA 4:1, small"),
    ]

    for seq_len, num_heads, num_kv_heads, head_dim, page_size, desc in test_configs:
        num_blocks = (seq_len // page_size) + 16

        query = torch.randn(num_heads, head_dim, device='cuda', dtype=torch.float16)
        keys = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        values = torch.randn(seq_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)

        # Paged
        kv_cache = PagedKVCache(num_blocks, page_size, num_kv_heads, head_dim, device='cuda')
        page_table = []
        kv_cache.write_kv_batch(page_table, keys, values)
        paged_out = paged_attention(query, kv_cache, page_table, seq_len)

        # Reference
        ref_out = reference_attention(query, keys, values)

        max_diff = torch.max(torch.abs(ref_out.float() - paged_out.float())).item()
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"  [{status}] {desc:>20} | seq={seq_len:>4} heads={num_heads:>2}/{num_kv_heads:>2} d={head_dim:>3} | max_diff={max_diff:.2e}")

    # ---- 内存碎片演示 ----
    print("\n--- 内存碎片场景演示 ---")
    print("模拟 3 个请求的分配/回收过程：")
    kv_cache2 = PagedKVCache(16, 16, 4, 64, device='cuda')

    # 请求 A: 48 tokens → 3 pages
    pt_a = []
    kv_a = torch.randn(48, 4, 64, device='cuda', dtype=torch.float16)
    kv_cache2.write_kv_batch(pt_a, kv_a, kv_a)
    print(f"  请求 A (48 tokens): 页表={pt_a}, 已用 block: {16 - len(kv_cache2.free_blocks)}/{16}")

    # 请求 B: 32 tokens → 2 pages
    pt_b = []
    kv_b = torch.randn(32, 4, 64, device='cuda', dtype=torch.float16)
    kv_cache2.write_kv_batch(pt_b, kv_b, kv_b)
    print(f"  请求 B (32 tokens): 页表={pt_b}, 已用 block: {16 - len(kv_cache2.free_blocks)}/{16}")

    # 请求 A 完成，释放
    for block in pt_a:
        kv_cache2.free_page(block)
    print(f"  请求 A 完成, 释放 {len(pt_a)} 个 block, 已用: {16 - len(kv_cache2.free_blocks)}/{16}")

    # 请求 C: 48 tokens → 3 pages (会复用 A 释放的 block)
    pt_c = []
    kv_c = torch.randn(48, 4, 64, device='cuda', dtype=torch.float16)
    kv_cache2.write_kv_batch(pt_c, kv_c, kv_c)
    print(f"  请求 C (48 tokens): 页表={pt_c}  ← 复用了 A 释放的物理 block!")
    print(f"  已用 block: {16 - len(kv_cache2.free_blocks)}/{16}")
    print("  → 连续分配做不到这种灵活复用，会导致内存碎片")

    # ---- Benchmark ----
    print("\n--- 性能对比 (Decode: 1 query token, attend to cached KV) ---")
    benchmark()

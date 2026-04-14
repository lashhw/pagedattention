import torch
import triton
import triton.language as tl

from common import _validate_decode_inputs


@triton.jit
def _paged_attention_decode_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    chunk_kv_head_ptr,
    chunk_start_block_ptr,
    chunk_num_blocks_ptr,
    softmax_scale,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    num_kv_groups,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_sh,
    stride_bh,
    stride_bb,
    stride_pmc,
    stride_pmg,
    stride_plc,
    stride_plg,
    stride_pac,
    stride_pag,
    stride_pad,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    tl.static_assert(BLOCK_N % BLOCK_T == 0, "BLOCK_N must be divisible by BLOCK_T.")
    BLOCKS_PER_STEP: tl.constexpr = BLOCK_N // BLOCK_T

    block_offs = tl.arange(0, BLOCKS_PER_STEP)
    t_offs = tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)

    kv_head_idx = tl.load(chunk_kv_head_ptr + chunk_idx)
    q_head_idx = kv_head_idx * num_kv_groups + group_idx

    seqlen = tl.load(cache_seqlens_ptr + kv_head_idx * stride_sh)
    start_block = tl.load(chunk_start_block_ptr + chunk_idx)
    num_blocks_in_chunk = tl.load(chunk_num_blocks_ptr + chunk_idx)

    q_ptrs = q_ptr + q_head_idx * stride_qh + d_offs * stride_qd
    q_bf16 = tl.load(q_ptrs)

    block_table_head_ptr = block_table_ptr + kv_head_idx * stride_bh
    k_block_offsets = t_offs[None, :, None] * stride_kt + d_offs[None, None, :] * stride_kd
    v_block_offsets = t_offs[None, :, None] * stride_vt + d_offs[None, None, :] * stride_vd

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for chunk_block_offset in tl.range(0, num_blocks_in_chunk, BLOCKS_PER_STEP):
        live_block_offs = chunk_block_offset + block_offs
        logical_block_idxs = start_block + live_block_offs
        block_mask = live_block_offs < num_blocks_in_chunk

        physical_block_ptrs = block_table_head_ptr + logical_block_idxs * stride_bb
        physical_block_idxs = tl.load(physical_block_ptrs, mask=block_mask, other=0).to(tl.int64)

        token_offsets = logical_block_idxs[:, None] * BLOCK_T + t_offs[None, :]
        t_mask = block_mask[:, None] & (token_offsets < seqlen)
        kv_mask = t_mask[:, :, None]

        k_ptrs = k_cache_ptr + physical_block_idxs[:, None, None] * stride_kb + k_block_offsets
        k_bf16 = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        k_bf16 = tl.reshape(k_bf16, BLOCK_N, BLOCK_D)

        t_mask = tl.reshape(t_mask, BLOCK_N)
        logits = tl.sum(q_bf16[None, :] * k_bf16, axis=1, dtype=tl.float32) * softmax_scale
        logits = tl.where(t_mask, logits, -float("inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(logits - m_new)
        p = tl.where(t_mask, p, 0.0)

        v_ptrs = v_cache_ptr + physical_block_idxs[:, None, None] * stride_vb + v_block_offsets
        v_bf16 = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        v_bf16 = tl.reshape(v_bf16, BLOCK_N, BLOCK_D)

        p_bf16 = p.to(tl.bfloat16)
        pv = tl.sum(p_bf16[:, None] * v_bf16, axis=0, dtype=tl.float32)

        m_i = m_new
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + pv

    partial_m_ptrs = partial_m_ptr + chunk_idx * stride_pmc + group_idx * stride_pmg
    partial_l_ptrs = partial_l_ptr + chunk_idx * stride_plc + group_idx * stride_plg
    partial_acc_ptrs = partial_acc_ptr + chunk_idx * stride_pac + group_idx * stride_pag + d_offs * stride_pad

    tl.store(partial_m_ptrs, m_i)
    tl.store(partial_l_ptrs, l_i)
    tl.store(partial_acc_ptrs, acc)


@triton.jit
def _paged_attention_decode_reduce_kernel(
    chunk_offsets_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    num_kv_groups,
    stride_pmc,
    stride_pmg,
    stride_plc,
    stride_plg,
    stride_pac,
    stride_pag,
    stride_pad,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    d_offs = tl.arange(0, BLOCK_D)

    start_chunk = tl.load(chunk_offsets_ptr + kv_head_idx)
    end_chunk = tl.load(chunk_offsets_ptr + kv_head_idx + 1)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for chunk_idx in tl.range(start_chunk, end_chunk):
        partial_m = tl.load(partial_m_ptr + chunk_idx * stride_pmc + group_idx * stride_pmg)
        partial_l = tl.load(partial_l_ptr + chunk_idx * stride_plc + group_idx * stride_plg)
        partial_acc_ptrs = partial_acc_ptr + chunk_idx * stride_pac + group_idx * stride_pag + d_offs * stride_pad
        partial_acc = tl.load(partial_acc_ptrs)

        has_acc = l_i > 0
        has_partial = partial_l > 0
        has_both = has_acc & has_partial

        m_new = tl.maximum(m_i, partial_m)
        alpha = tl.where(has_both, tl.exp(m_i - m_new), tl.where(has_acc, 1.0, 0.0))
        beta = tl.where(has_both, tl.exp(partial_m - m_new), tl.where(has_partial, 1.0, 0.0))

        m_i = m_new
        l_i = l_i * alpha + partial_l * beta
        acc = acc * alpha + partial_acc * beta

    q_head_idx = kv_head_idx * num_kv_groups + group_idx
    out_ptrs = out_ptr + q_head_idx * stride_oh + d_offs * stride_od

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom
    tl.store(out_ptrs, out)


def _ceil_div(numer, denom):
    return (numer + denom - 1) // denom


def flash_attn_with_kvcache_wrapper_triton(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    num_splits=1,
    block_n=128,
    num_warps=4,
    num_stages=1,
):
    num_query_heads, num_kv_heads, num_kv_groups, head_size = _validate_decode_inputs(q, cache_seqlens, block_table)

    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("This kernel only supports BF16 inputs for q, k_cache, and v_cache.")

    q_heads = q[0, 0].contiguous()
    cache_seqlens_heads = cache_seqlens[0].contiguous()
    block_table_heads = block_table[0].contiguous()
    out = torch.empty_like(q_heads)

    block_t = k_cache.shape[1]
    block_d = head_size

    num_blocks_per_head = _ceil_div(cache_seqlens_heads, block_t).cpu().tolist()
    total_blocks = sum(num_blocks_per_head)
    target_chunk_count = num_kv_heads * num_splits
    blocks_per_chunk = _ceil_div(total_blocks, target_chunk_count)

    chunk_offsets = [0]
    chunk_kv_heads = []
    chunk_start_blocks = []
    chunk_num_blocks = []

    for kv_head_idx, num_blocks in enumerate(num_blocks_per_head):
        num_chunks = _ceil_div(num_blocks, blocks_per_chunk)
        for chunk_idx in range(num_chunks):
            start_block = chunk_idx * blocks_per_chunk
            live_blocks = min(blocks_per_chunk, num_blocks - start_block)
            chunk_kv_heads.append(kv_head_idx)
            chunk_start_blocks.append(start_block)
            chunk_num_blocks.append(live_blocks)
        chunk_offsets.append(len(chunk_kv_heads))

    chunk_offsets = torch.tensor(chunk_offsets, dtype=torch.int32, device=q.device)
    chunk_kv_heads = torch.tensor(chunk_kv_heads, dtype=torch.int32, device=q.device)
    chunk_start_blocks = torch.tensor(chunk_start_blocks, dtype=torch.int32, device=q.device)
    chunk_num_blocks = torch.tensor(chunk_num_blocks, dtype=torch.int32, device=q.device)

    total_live_chunks = chunk_kv_heads.numel()
    partial_m = torch.empty(
        (total_live_chunks, num_kv_groups),
        dtype=torch.float32, device=q.device
    )
    partial_l = torch.empty(
        (total_live_chunks, num_kv_groups),
        dtype=torch.float32, device=q.device
    )
    partial_acc = torch.empty(
        (total_live_chunks, num_kv_groups, head_size),
        dtype=torch.float32, device=q.device
    )

    split_grid = (total_live_chunks, num_kv_groups)
    _paged_attention_decode_split_kernel[split_grid](
        q_heads,
        k_cache,
        v_cache,
        cache_seqlens_heads,
        block_table_heads,
        chunk_kv_heads,
        chunk_start_blocks,
        chunk_num_blocks,
        softmax_scale,
        partial_m,
        partial_l,
        partial_acc,
        num_kv_groups,
        q_heads.stride(0),
        q_heads.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        cache_seqlens_heads.stride(0),
        block_table_heads.stride(0),
        block_table_heads.stride(1),
        partial_m.stride(0),
        partial_m.stride(1),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        BLOCK_N=block_n,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    reduce_grid = (num_kv_heads, num_kv_groups)
    _paged_attention_decode_reduce_kernel[reduce_grid](
        chunk_offsets,
        partial_m,
        partial_l,
        partial_acc,
        out,
        num_kv_groups,
        partial_m.stride(0),
        partial_m.stride(1),
        partial_l.stride(0),
        partial_l.stride(1),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out.view(1, 1, num_query_heads, head_size)

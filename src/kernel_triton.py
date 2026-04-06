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
    softmax_scale,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    num_kv_groups,
    head_size,
    blocks_per_split,
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
    stride_pmh,
    stride_pms,
    stride_plh,
    stride_pls,
    stride_pah,
    stride_pas,
    stride_pad,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // num_kv_groups

    seqlen = tl.load(cache_seqlens_ptr + kv_head_idx * stride_sh)
    num_blocks = tl.cdiv(seqlen, BLOCK_T)

    start_block = split_idx * blocks_per_split
    end_block = tl.minimum(start_block + blocks_per_split, num_blocks)
    num_blocks_in_split = tl.maximum(end_block - start_block, 0)

    t_offs = tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size

    q_ptrs = q_ptr + q_head_idx * stride_qh + d_offs * stride_qd
    q_bf16 = tl.load(q_ptrs, mask=d_mask, other=0.0)

    block_table_head_ptr = block_table_ptr + kv_head_idx * stride_bh
    k_block_offsets = t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    v_block_offsets = t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for split_block_offset in tl.range(0, num_blocks_in_split):
        logical_block_idx = start_block + split_block_offset
        physical_block_idx = tl.load(block_table_head_ptr + logical_block_idx * stride_bb)

        token_offsets = logical_block_idx * BLOCK_T + t_offs
        t_mask = token_offsets < seqlen
        kv_mask = t_mask[:, None] & d_mask[None, :]

        k_ptrs = k_cache_ptr + physical_block_idx * stride_kb + k_block_offsets
        k_bf16 = tl.load(k_ptrs, mask=kv_mask, other=0.0)

        logits = tl.sum(q_bf16[None, :] * k_bf16, axis=1, dtype=tl.float32) * softmax_scale
        logits = tl.where(t_mask, logits, -float("inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(logits - m_new)
        p = tl.where(t_mask, p, 0.0)

        v_ptrs = v_cache_ptr + physical_block_idx * stride_vb + v_block_offsets
        v_bf16 = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        p_bf16 = p.to(tl.bfloat16)
        pv = tl.sum(p_bf16[:, None] * v_bf16, axis=0, dtype=tl.float32)

        m_i = m_new
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + pv

    partial_m_ptrs = partial_m_ptr + q_head_idx * stride_pmh + split_idx * stride_pms
    partial_l_ptrs = partial_l_ptr + q_head_idx * stride_plh + split_idx * stride_pls
    partial_acc_ptrs = partial_acc_ptr + q_head_idx * stride_pah + split_idx * stride_pas + d_offs * stride_pad

    tl.store(partial_m_ptrs, m_i)
    tl.store(partial_l_ptrs, l_i)
    tl.store(partial_acc_ptrs, acc, mask=d_mask)


@triton.jit
def _paged_attention_decode_reduce_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    head_size,
    num_splits,
    stride_pmh,
    stride_pms,
    stride_plh,
    stride_pls,
    stride_pah,
    stride_pas,
    stride_pad,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    q_head_idx = tl.program_id(0)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for split_idx in tl.range(0, num_splits):
        partial_m = tl.load(partial_m_ptr + q_head_idx * stride_pmh + split_idx * stride_pms)
        partial_l = tl.load(partial_l_ptr + q_head_idx * stride_plh + split_idx * stride_pls)
        partial_acc_ptrs = partial_acc_ptr + q_head_idx * stride_pah + split_idx * stride_pas + d_offs * stride_pad
        partial_acc = tl.load(partial_acc_ptrs, mask=d_mask, other=0.0)

        has_acc = l_i > 0
        has_partial = partial_l > 0
        has_both = has_acc & has_partial

        m_new = tl.where(
            has_both,
            tl.maximum(m_i, partial_m),
            tl.where(has_acc, m_i, tl.where(has_partial, partial_m, 0.0)),
        )
        alpha = tl.where(has_both, tl.exp(m_i - m_new), tl.where(has_acc, 1.0, 0.0))
        beta = tl.where(has_both, tl.exp(partial_m - m_new), tl.where(has_partial, 1.0, 0.0))

        l_i = l_i * alpha + partial_l * beta
        acc = acc * alpha + partial_acc * beta
        m_i = tl.where(has_partial, m_new, m_i)

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom
    out_ptrs = out_ptr + q_head_idx * stride_oh + d_offs * stride_od
    tl.store(out_ptrs, out, mask=d_mask)


def flash_attn_with_kvcache_wrapper_triton(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    num_splits,
):
    num_query_heads, _, num_kv_groups, head_size = _validate_decode_inputs(q, cache_seqlens, block_table)

    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("This kernel only supports BF16 inputs for q, k_cache, and v_cache.")

    q_heads = q[0, 0].contiguous()
    cache_seqlens_heads = cache_seqlens[0].contiguous()
    block_table_heads = block_table[0].contiguous()
    out = torch.empty_like(q_heads)

    block_t = k_cache.shape[1]
    global_max_seqlen = cache_seqlens_heads.max().item()
    global_num_blocks = (global_max_seqlen + block_t - 1) // block_t
    blocks_per_split = (global_num_blocks + num_splits - 1) // num_splits

    partial_m = torch.empty(
        (num_query_heads, num_splits),
        device=q.device, dtype=torch.float32
    )
    partial_l = torch.empty(
        (num_query_heads, num_splits),
        device=q.device, dtype=torch.float32
    )
    partial_acc = torch.empty(
        (num_query_heads, num_splits, head_size),
        device=q.device, dtype=torch.float32
    )

    grid = (num_query_heads,)
    block_d = triton.next_power_of_2(head_size)
    num_warps = 4
    num_stages = 1

    split_grid = (num_query_heads, num_splits)
    _paged_attention_decode_split_kernel[split_grid](
        q_heads,
        k_cache,
        v_cache,
        cache_seqlens_heads,
        block_table_heads,
        softmax_scale,
        partial_m,
        partial_l,
        partial_acc,
        num_kv_groups,
        head_size,
        blocks_per_split,
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
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _paged_attention_decode_reduce_kernel[grid](
        partial_m,
        partial_l,
        partial_acc,
        out,
        head_size,
        num_splits,
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

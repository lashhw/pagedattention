import torch
import triton
import triton.language as tl

from pagedattention_common import _validate_decode_inputs


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    out_ptr,
    num_kv_groups,
    head_size,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_sl,
    stride_bth,
    stride_btb,
    stride_oh,
    stride_od,
    softmax_scale,
    BLOCK_D: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx // num_kv_groups

    seqlen = tl.load(cache_seqlens_ptr + kv_head_idx * stride_sl)
    num_blocks = tl.cdiv(seqlen, 16)

    t_offs = tl.arange(0, 16)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size

    q_ptrs = q_ptr + q_head_idx * stride_qh + d_offs * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    block_table_head_ptr = block_table_ptr + kv_head_idx * stride_bth
    k_block_offsets = t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
    v_block_offsets = t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for logical_block_idx in tl.range(0, num_blocks):
        physical_block_idx = tl.load(block_table_head_ptr + logical_block_idx * stride_btb)

        token_offsets = logical_block_idx * 16 + t_offs
        t_mask = token_offsets < seqlen
        kv_mask = t_mask[:, None] & d_mask[None, :]

        k_ptrs = k_cache_ptr + physical_block_idx * stride_kb + k_block_offsets
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        logits = tl.sum(k * q[None, :], axis=1) * softmax_scale
        logits = tl.where(t_mask, logits, -float("inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(logits - m_new)
        p = tl.where(t_mask, p, 0.0)

        v_ptrs = v_cache_ptr + physical_block_idx * stride_vb + v_block_offsets
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        m_i = m_new
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom
    out_ptrs = out_ptr + q_head_idx * stride_oh + d_offs * stride_od
    tl.store(out_ptrs, out, mask=d_mask)


def flash_attn_with_kvcache_wrapper_triton(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale):
    num_query_heads, _, num_kv_groups, head_size = _validate_decode_inputs(q, cache_seqlens, block_table)

    block_size = k_cache.shape[1]
    assert block_size == 16, "flash_attn_with_kvcache_wrapper_triton requires block_size == 16"

    q_heads = q[0, 0].contiguous()
    cache_seqlens_heads = cache_seqlens[0].contiguous()
    block_table_heads = block_table[0].contiguous()
    out = torch.empty_like(q_heads)

    grid = (num_query_heads,)
    block_d = triton.next_power_of_2(head_size)
    num_warps = 4

    _paged_attention_decode_kernel[grid](
        q_heads,
        k_cache,
        v_cache,
        cache_seqlens_heads,
        block_table_heads,
        out,
        num_kv_groups,
        head_size,
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
        out.stride(0),
        out.stride(1),
        softmax_scale,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )

    return out.view(1, 1, num_query_heads, head_size)

import torch
import triton
import triton.language as tl

from pagedattention_common import _validate_decode_inputs

_BLOCK_SIZE = 16
_BLOCKS_PER_PARTITION = 8
_MAX_GROUP_HEADS = 8


@triton.jit
def _paged_attention_decode_partition_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
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
    stride_sl,
    stride_bth,
    stride_btb,
    stride_pmh,
    stride_pmp,
    stride_plh,
    stride_plp,
    stride_pah,
    stride_pap,
    stride_pad,
    softmax_scale,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PARTITION: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    partition_idx = tl.program_id(1)
    group_tile_idx = tl.program_id(2)

    offs_t = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, BLOCK_D)
    offs_g = tl.arange(0, BLOCK_G)

    q_group_offset = group_tile_idx * BLOCK_G + offs_g
    q_head_offsets = kv_head_idx * num_kv_groups + q_group_offset
    q_head_mask = q_group_offset < num_kv_groups

    q_ptrs = q_ptr + q_head_offsets[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_mask = q_head_mask[:, None] & (offs_d[None, :] < HEAD_SIZE)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    seqlen = tl.load(cache_seqlens_ptr + kv_head_idx * stride_sl)
    num_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    partition_start = partition_idx * BLOCKS_PER_PARTITION

    m_i = tl.full([BLOCK_G], -1.0e20, tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    for block_offset in tl.static_range(0, BLOCKS_PER_PARTITION):
        logical_block_idx = partition_start + block_offset
        block_active = logical_block_idx < num_blocks
        physical_block_idx = tl.load(
            block_table_ptr + kv_head_idx * stride_bth + logical_block_idx * stride_btb,
            mask=block_active,
            other=0,
        )

        token_start = logical_block_idx * BLOCK_SIZE
        token_offsets = token_start + offs_t
        token_mask = block_active & (token_offsets < seqlen)
        kv_mask = token_mask[:, None] & (offs_d[None, :] < HEAD_SIZE)

        k_ptrs = (
            k_cache_ptr
            + physical_block_idx * stride_kb
            + offs_t[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        logits = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * softmax_scale
        logits = tl.where(token_mask[None, :], logits, -1.0e20)

        m_ij = tl.max(logits, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new[:, None])
        p = tl.where(token_mask[None, :], p, 0.0)

        v_ptrs = (
            v_cache_ptr
            + physical_block_idx * stride_vb
            + offs_t[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p, v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    partial_m_ptrs = partial_m_ptr + q_head_offsets * stride_pmh + partition_idx * stride_pmp
    partial_l_ptrs = partial_l_ptr + q_head_offsets * stride_plh + partition_idx * stride_plp
    partial_acc_ptrs = (
        partial_acc_ptr
        + q_head_offsets[:, None] * stride_pah
        + partition_idx * stride_pap
        + offs_d[None, :] * stride_pad
    )

    tl.store(partial_m_ptrs, m_i, mask=q_head_mask)
    tl.store(partial_l_ptrs, l_i, mask=q_head_mask)
    tl.store(partial_acc_ptrs, acc, mask=q_mask)


@triton.jit
def _paged_attention_decode_reduce_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    out_ptr,
    max_num_partitions,
    stride_pmh,
    stride_pmp,
    stride_plh,
    stride_plp,
    stride_pah,
    stride_pap,
    stride_pad,
    stride_oh,
    stride_od,
    BLOCK_D: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    m_i = -1.0e20
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for partition_idx in tl.range(0, max_num_partitions):
        m_p = tl.load(partial_m_ptr + q_head_idx * stride_pmh + partition_idx * stride_pmp)
        l_p = tl.load(partial_l_ptr + q_head_idx * stride_plh + partition_idx * stride_plp)
        acc_p_ptrs = partial_acc_ptr + q_head_idx * stride_pah + partition_idx * stride_pap + offs_d * stride_pad
        acc_p = tl.load(acc_p_ptrs, mask=offs_d < HEAD_SIZE, other=0.0)

        m_new = tl.maximum(m_i, m_p)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_p - m_new)

        acc = acc * alpha + acc_p * beta
        l_i = l_i * alpha + l_p * beta
        m_i = m_new

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom
    out_ptrs = out_ptr + q_head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=offs_d < HEAD_SIZE)


def flash_attn_with_kvcache_wrapper_triton(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale):
    num_query_heads, num_kv_heads, num_kv_groups, head_size = _validate_decode_inputs(q, cache_seqlens, block_table)

    block_size = k_cache.shape[1]
    assert block_size == _BLOCK_SIZE, f"Triton decode kernel requires block_size == {_BLOCK_SIZE}, got {block_size}"

    block_d = triton.next_power_of_2(head_size)
    block_g = min(_MAX_GROUP_HEADS, triton.next_power_of_2(num_kv_groups))

    q_heads = q[0, 0].contiguous()
    cache_seqlens_heads = cache_seqlens[0].contiguous()
    block_table_heads = block_table[0].contiguous()
    out = torch.empty_like(q_heads)

    max_num_blocks_per_head = block_table_heads.shape[1]
    max_num_partitions = max(1, triton.cdiv(max_num_blocks_per_head, _BLOCKS_PER_PARTITION))
    partial_m = torch.empty((num_query_heads, max_num_partitions), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((num_query_heads, max_num_partitions), device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((num_query_heads, max_num_partitions, head_size), device=q.device, dtype=torch.float32)

    num_warps = 4 if block_d <= 128 else 8
    partition_grid = (num_kv_heads, max_num_partitions, triton.cdiv(num_kv_groups, block_g))
    _paged_attention_decode_partition_kernel[partition_grid](
        q_heads,
        k_cache,
        v_cache,
        cache_seqlens_heads,
        block_table_heads,
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
        softmax_scale,
        BLOCK_D=block_d,
        BLOCK_G=block_g,
        BLOCK_SIZE=_BLOCK_SIZE,
        BLOCKS_PER_PARTITION=_BLOCKS_PER_PARTITION,
        HEAD_SIZE=head_size,
        num_warps=num_warps,
        num_stages=1,
    )

    _paged_attention_decode_reduce_kernel[(num_query_heads,)](
        partial_m,
        partial_l,
        partial_acc,
        out,
        max_num_partitions,
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
        HEAD_SIZE=head_size,
        num_warps=4,
        num_stages=1,
    )

    return out.view(1, 1, num_query_heads, head_size)

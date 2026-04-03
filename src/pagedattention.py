import torch
import triton
import triton.language as tl


"""
Interface for flash_attn_with_kvcache_wrapper

Args:
    q: Query tensor of shape (batch_size, 1, num_query_heads, head_size)
    k_cache: KV cache for keys of shape (num_blocks, block_size, head_size)
    v_cache: KV cache for values of shape (num_blocks, block_size, head_size)
    cache_seqlens: Sequence lengths of shape (batch_size, num_kv_heads)
    block_table: Block table mapping of shape (batch_size, num_kv_heads, max_num_blocks_per_head)
    softmax_scale: Scaling factor for attention scores

Returns:
    Output tensor of shape (batch_size, 1, num_query_heads, head_size)
"""


def _validate_decode_inputs(q, cache_seqlens, block_table):
    assert q.shape[0] == 1
    assert cache_seqlens.shape[0] == 1
    assert block_table.shape[0] == 1

    _, q_len, num_query_heads, head_size = q.shape
    _, num_kv_heads = cache_seqlens.shape

    assert q_len == 1
    assert num_query_heads % num_kv_heads == 0

    num_kv_groups = num_query_heads // num_kv_heads

    return num_query_heads, num_kv_heads, num_kv_groups, head_size


def _materialize_kvcache(k_cache, v_cache, cache_seqlens, block_table):
    assert cache_seqlens.shape[0] == 1
    assert block_table.shape[0] == 1

    _, num_kv_heads = cache_seqlens.shape
    _, block_size, _ = k_cache.shape

    k_heads = []
    v_heads = []
    for kv_head_idx in range(num_kv_heads):
        seqlen = cache_seqlens[0, kv_head_idx].item()
        num_blocks = (seqlen + block_size - 1) // block_size

        physical_block_ids = block_table[0, kv_head_idx, :num_blocks]
        k_head = k_cache[physical_block_ids].flatten(0, 1)[:seqlen]
        v_head = v_cache[physical_block_ids].flatten(0, 1)[:seqlen]

        k_heads.append(k_head)
        v_heads.append(v_head)

    return k_heads, v_heads


def flash_attn_with_kvcache_wrapper_eager(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale):
    _, num_kv_heads, num_kv_groups, _ = _validate_decode_inputs(q, cache_seqlens, block_table)
    q = q.transpose(1, 2).contiguous()
    k_heads, v_heads = _materialize_kvcache(k_cache, v_cache, cache_seqlens, block_table)

    out_heads = []
    for kv_head_idx in range(num_kv_heads):
        q_start = kv_head_idx * num_kv_groups
        q_end = q_start + num_kv_groups
        q_head_group = q[:, q_start:q_end]

        k_head = k_heads[kv_head_idx][None, None, :, :]
        v_head = v_heads[kv_head_idx][None, None, :, :]

        attn_weights = torch.matmul(q_head_group, k_head.transpose(-2, -1)) * softmax_scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out_head_group = torch.matmul(attn_weights, v_head)
        out_heads.append(out_head_group)

    return torch.cat(out_heads, dim=1).transpose(1, 2).contiguous()


@triton.jit
def _paged_attention_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_seqlens_ptr,
    block_table_ptr,
    out_ptr,
    num_kv_groups,
    block_size,
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
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx // num_kv_groups

    seqlen = tl.load(cache_seqlens_ptr + kv_head_idx * stride_sl)
    num_blocks = tl.cdiv(seqlen, block_size)

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + q_head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < head_size, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for logical_block_idx in tl.range(0, num_blocks):
        physical_block_idx = tl.load(block_table_ptr + kv_head_idx * stride_bth + logical_block_idx * stride_btb)

        token_start = logical_block_idx * block_size
        token_offsets = token_start + offs_t
        token_mask = token_offsets < seqlen

        k_ptrs = (
            k_cache_ptr
            + physical_block_idx * stride_kb
            + offs_t[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=token_mask[:, None] & (offs_d[None, :] < head_size), other=0.0).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * softmax_scale
        logits = tl.where(token_mask, logits, -float("inf"))

        m_ij = tl.max(logits, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)
        p = tl.where(token_mask, p, 0.0)

        v_ptrs = (
            v_cache_ptr
            + physical_block_idx * stride_vb
            + offs_t[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=token_mask[:, None] & (offs_d[None, :] < head_size), other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    denom = tl.where(l_i > 0, l_i, 1.0)
    out = acc / denom
    out_ptrs = out_ptr + q_head_idx * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, out, mask=offs_d < head_size)


def flash_attn_with_kvcache_wrapper_triton(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale):
    num_query_heads, _, num_kv_groups, head_size = _validate_decode_inputs(q, cache_seqlens, block_table)

    block_size = k_cache.shape[1]
    block_d = triton.next_power_of_2(head_size)

    q_heads = q[0, 0].contiguous()
    cache_seqlens_heads = cache_seqlens[0].contiguous()
    block_table_heads = block_table[0].contiguous()
    out = torch.empty_like(q_heads)

    block_t = triton.next_power_of_2(block_size)
    num_warps = 4 if block_d <= 128 else 8

    grid = (num_query_heads,)
    _paged_attention_decode_kernel[grid](
        q_heads,
        k_cache,
        v_cache,
        cache_seqlens_heads,
        block_table_heads,
        out,
        num_kv_groups,
        block_size,
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
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )

    return out.view(1, 1, num_query_heads, head_size)

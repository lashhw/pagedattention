import torch

from common import _validate_decode_inputs


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


def flash_attn_with_kvcache_wrapper_eager(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    num_splits,
):
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

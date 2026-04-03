"""
Interface for flash_attn_with_kvcache_wrapper_*

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

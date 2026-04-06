from einops import rearrange
from vllm.vllm_flash_attn import flash_attn_with_kvcache

def flash_attn_with_kvcache_wrapper_vllm(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    softmax_scale,
    num_splits,
):
    batch_size, _, num_query_heads, _ = q.shape
    _, num_kv_heads = cache_seqlens.shape
    num_kv_groups = num_query_heads // num_kv_heads

    q = rearrange(
        q,
        'batch_size 1 (num_kv_heads num_kv_groups) head_size -> (batch_size num_kv_heads) 1 num_kv_groups head_size',
        num_kv_heads=num_kv_heads,
        num_kv_groups=num_kv_groups
    )
    k_cache = rearrange(k_cache, 'num_blocks block_size head_size -> num_blocks block_size 1 head_size')
    v_cache = rearrange(v_cache, 'num_blocks block_size head_size -> num_blocks block_size 1 head_size')
    cache_seqlens = rearrange(cache_seqlens, 'batch_size num_kv_heads -> (batch_size num_kv_heads)')
    block_table = rearrange(block_table, 'batch_size num_kv_heads max_num_blocks_per_head -> (batch_size num_kv_heads) max_num_blocks_per_head')

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=num_splits,
    )

    return rearrange(
        out,
        '(batch_size num_kv_heads) 1 num_kv_groups head_size -> batch_size 1 (num_kv_heads num_kv_groups) head_size',
        batch_size=batch_size,
        num_kv_heads=num_kv_heads
    )
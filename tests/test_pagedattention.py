import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedattention_eager import flash_attn_with_kvcache_wrapper_eager
from pagedattention_triton import flash_attn_with_kvcache_wrapper_triton


def _build_decode_inputs(
    *,
    dtype,
    block_size,
    head_size,
    num_kv_heads,
    num_kv_groups,
    seqlens,
):
    assert len(seqlens) == num_kv_heads

    torch.manual_seed(42)

    device = "cuda"
    num_query_heads = num_kv_heads * num_kv_groups
    blocks_per_head = [(seqlen + block_size - 1) // block_size for seqlen in seqlens]
    max_num_blocks_per_head = max(blocks_per_head)
    num_blocks = sum(blocks_per_head)

    q = torch.randn((1, 1, num_query_heads, head_size), device=device, dtype=dtype)
    k_cache = torch.randn((num_blocks, block_size, head_size), device=device, dtype=dtype)
    v_cache = torch.randn((num_blocks, block_size, head_size), device=device, dtype=dtype)
    cache_seqlens = torch.tensor([seqlens], device=device, dtype=torch.int32)
    block_table = torch.zeros((1, num_kv_heads, max_num_blocks_per_head), device=device, dtype=torch.int32)

    next_block_id = 0
    for kv_head_idx, num_head_blocks in enumerate(blocks_per_head):
        block_table[0, kv_head_idx, :num_head_blocks] = torch.arange(
            next_block_id,
            next_block_id + num_head_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block_id += num_head_blocks

    return q, k_cache, v_cache, cache_seqlens, block_table


@pytest.mark.parametrize(
    ("dtype", "block_size", "head_size", "num_kv_heads", "num_kv_groups", "seqlens", "atol", "rtol"),
    [
        (torch.bfloat16, 16, 128, 8, 4, [1000, 7000, 2000, 4000, 8000, 5000, 6000, 3000], 2e-2, 2e-2),
        (torch.bfloat16, 32, 128, 8, 4, [1000, 7000, 2000, 4000, 8000, 5000, 6000, 3000], 2e-2, 2e-2),
    ],
)
def test_flash_attn_with_kvcache_wrapper_triton_matches_eager(
    dtype,
    block_size,
    head_size,
    num_kv_heads,
    num_kv_groups,
    seqlens,
    atol,
    rtol,
):
    q, k_cache, v_cache, cache_seqlens, block_table = _build_decode_inputs(
        dtype=dtype,
        block_size=block_size,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_kv_groups=num_kv_groups,
        seqlens=seqlens,
    )
    softmax_scale = 1.0 / math.sqrt(head_size)

    eager_out = flash_attn_with_kvcache_wrapper_eager(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
    )
    triton_out = flash_attn_with_kvcache_wrapper_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
    )

    torch.testing.assert_close(triton_out, eager_out, atol=atol, rtol=rtol)

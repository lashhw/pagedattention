import math
import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedattention_eager import flash_attn_with_kvcache_wrapper_eager
from pagedattention_triton import flash_attn_with_kvcache_wrapper_triton


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to run the Triton paged-attention kernel.",
)


def _build_decode_inputs(
    *,
    dtype,
    block_size,
    head_size,
    num_kv_heads,
    num_kv_groups,
    seqlens,
    seed,
):
    assert len(seqlens) == num_kv_heads

    torch.manual_seed(seed)

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
    ("dtype", "block_size", "head_size", "num_kv_heads", "num_kv_groups", "seqlens", "seed", "atol", "rtol"),
    [
        (torch.float16, 16, 64, 2, 2, [9, 27], 0, 2e-2, 2e-2),
        (torch.float32, 16, 96, 3, 1, [7, 35, 50], 1, 5e-4, 5e-4),
        (torch.float16, 16, 128, 2, 4, [16, 193], 2, 3e-2, 3e-2),
    ],
)
def test_flash_attn_with_kvcache_wrapper_triton_matches_eager(
    dtype,
    block_size,
    head_size,
    num_kv_heads,
    num_kv_groups,
    seqlens,
    seed,
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
        seed=seed,
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

import math
import unittest

import torch

from src.pagedattention import (
    flash_attn_with_kvcache_wrapper,
    flash_attn_with_kvcache_wrapper_eager,
    flash_attn_with_kvcache_wrapper_triton,
)


def _make_inputs(device, dtype):
    torch.manual_seed(0)

    num_kv_heads = 2
    num_query_heads = 4
    block_size = 4
    head_size = 32
    seqlens = [7, 5]
    blocks_per_head = [(seqlen + block_size - 1) // block_size for seqlen in seqlens]
    total_blocks = sum(blocks_per_head)
    max_blocks = max(blocks_per_head)

    q = torch.randn(1, 1, num_query_heads, head_size, device=device, dtype=dtype)
    k_cache = torch.randn(total_blocks, block_size, head_size, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)

    cache_seqlens = torch.tensor([seqlens], device=device, dtype=torch.int32)
    block_table = torch.zeros(1, num_kv_heads, max_blocks, device=device, dtype=torch.int64)

    physical_blocks = torch.randperm(total_blocks, device=device, dtype=torch.int64)
    cursor = 0
    for kv_head_idx, num_blocks in enumerate(blocks_per_head):
        block_table[0, kv_head_idx, :num_blocks] = physical_blocks[cursor:cursor + num_blocks]
        cursor += num_blocks

    softmax_scale = 1.0 / math.sqrt(head_size)
    return q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale


class PagedAttentionTest(unittest.TestCase):
    def test_wrapper_falls_back_to_eager_on_cpu(self):
        inputs = _make_inputs(device="cpu", dtype=torch.float32)
        out = flash_attn_with_kvcache_wrapper(*inputs)
        ref = flash_attn_with_kvcache_wrapper_eager(*inputs)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the Triton kernel test.")
    def test_triton_matches_eager(self):
        inputs = _make_inputs(device="cuda", dtype=torch.float16)
        out = flash_attn_with_kvcache_wrapper_triton(*inputs)
        ref = flash_attn_with_kvcache_wrapper_eager(*inputs)
        self.assertTrue(torch.allclose(out, ref, atol=2e-2, rtol=2e-2))


if __name__ == "__main__":
    unittest.main()

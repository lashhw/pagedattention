import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pagedattention import (
    _validate_decode_inputs,
    flash_attn_with_kvcache_wrapper_eager,
    flash_attn_with_kvcache_wrapper_triton,
)


def _make_inputs(
    device,
    dtype,
    *,
    num_kv_heads=2,
    num_query_heads=4,
    block_size=4,
    head_size=32,
    seqlens=(7, 5),
):
    torch.manual_seed(0)
    assert len(seqlens) == num_kv_heads

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


def _manual_grouped_attention_reference(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale):
    _, _, num_query_heads, head_size = q.shape
    _, num_kv_heads = cache_seqlens.shape
    num_kv_groups = num_query_heads // num_kv_heads
    block_size = k_cache.shape[1]

    out = torch.empty_like(q)
    for kv_head_idx in range(num_kv_heads):
        seqlen = cache_seqlens[0, kv_head_idx].item()
        num_blocks = (seqlen + block_size - 1) // block_size
        physical_block_ids = block_table[0, kv_head_idx, :num_blocks]

        k_head = k_cache[physical_block_ids].reshape(-1, head_size)[:seqlen]
        v_head = v_cache[physical_block_ids].reshape(-1, head_size)[:seqlen]

        q_start = kv_head_idx * num_kv_groups
        q_end = q_start + num_kv_groups
        q_head_group = q[0, 0, q_start:q_end]

        attn_weights = torch.matmul(q_head_group, k_head.transpose(-2, -1)) * softmax_scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out[0, 0, q_start:q_end] = torch.matmul(attn_weights, v_head)

    return out


def test_validate_decode_inputs_returns_num_kv_groups():
    q, k_cache, v_cache, cache_seqlens, block_table, _ = _make_inputs(
        device="cpu",
        dtype=torch.float32,
        num_query_heads=6,
    )

    num_query_heads, num_kv_heads, num_kv_groups, head_size = _validate_decode_inputs(
        q, k_cache, v_cache, cache_seqlens, block_table
    )

    assert (num_query_heads, num_kv_heads, num_kv_groups, head_size) == (6, 2, 3, 32)


def test_validate_decode_inputs_rejects_non_divisible_query_heads():
    q, k_cache, v_cache, cache_seqlens, block_table, _ = _make_inputs(
        device="cpu",
        dtype=torch.float32,
        num_query_heads=4,
    )
    invalid_q = q[:, :, :3, :]

    with pytest.raises(AssertionError):
        _validate_decode_inputs(invalid_q, k_cache, v_cache, cache_seqlens, block_table)


def test_eager_matches_manual_grouped_attention_reference():
    inputs = _make_inputs(
        device="cpu",
        dtype=torch.float32,
        num_query_heads=6,
    )

    out = flash_attn_with_kvcache_wrapper_eager(*inputs)
    ref = _manual_grouped_attention_reference(*inputs)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel test.")
def test_triton_matches_eager():
    inputs = _make_inputs(device="cuda", dtype=torch.float16)

    out = flash_attn_with_kvcache_wrapper_triton(*inputs)
    ref = flash_attn_with_kvcache_wrapper_eager(*inputs)

    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)

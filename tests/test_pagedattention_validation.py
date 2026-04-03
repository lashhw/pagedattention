import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedattention_triton import flash_attn_with_kvcache_wrapper_triton


def test_flash_attn_with_kvcache_wrapper_triton_requires_block_size_16():
    q = torch.randn((1, 1, 4, 64), dtype=torch.float32)
    k_cache = torch.randn((4, 32, 64), dtype=torch.float32)
    v_cache = torch.randn((4, 32, 64), dtype=torch.float32)
    cache_seqlens = torch.tensor([[32, 64]], dtype=torch.int32)
    block_table = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int32)

    with pytest.raises(AssertionError, match=r"block_size == 16"):
        flash_attn_with_kvcache_wrapper_triton(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=1.0,
        )

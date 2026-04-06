import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_eager import flash_attn_with_kvcache_wrapper_eager
from kernel_triton import flash_attn_with_kvcache_wrapper_triton
from kernel_vllm import flash_attn_with_kvcache_wrapper_vllm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_kv_groups",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--head_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=[100000],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
    )
    return parser.parse_args()


def _build_decode_inputs(
    block_size,
    head_size,
    num_kv_heads,
    num_kv_groups,
    seqlens,
    seed,
):
    if len(seqlens) != num_kv_heads:
        raise ValueError(f"Expected {num_kv_heads} sequence lengths, got {len(seqlens)}")

    torch.manual_seed(seed)

    device = "cuda"
    dtype = torch.bfloat16

    num_query_heads = num_kv_heads * num_kv_groups
    blocks_per_head = [(seqlen + block_size - 1) // block_size for seqlen in seqlens]
    max_num_blocks_per_head = max(blocks_per_head)
    num_blocks = sum(blocks_per_head)

    q = torch.randn((1, 1, num_query_heads, head_size), device=device, dtype=dtype)
    k_cache = torch.randn((num_blocks, block_size, head_size), device=device, dtype=dtype)
    v_cache = torch.randn((num_blocks, block_size, head_size), device=device, dtype=dtype)
    cache_seqlens = torch.tensor([seqlens], device=device, dtype=torch.int32)
    block_table = torch.zeros((1, num_kv_heads, max_num_blocks_per_head), device=device, dtype=torch.int32)

    physical_block_ids = torch.randperm(num_blocks, device=device, dtype=torch.int32)
    next_block_offset = 0
    for kv_head_idx, num_head_blocks in enumerate(blocks_per_head):
        head_block_ids = physical_block_ids[next_block_offset : next_block_offset + num_head_blocks]
        block_table[0, kv_head_idx, :num_head_blocks] = head_block_ids
        next_block_offset += num_head_blocks

    return q, k_cache, v_cache, cache_seqlens, block_table


def _benchmark(fn, warmup, iters, **kwargs):
    for _ in range(warmup):
        fn(**kwargs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        fn(**kwargs)
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / iters
    return avg_ms


def main():
    args = _parse_args()

    if len(args.seqlens) == 1:
        args.seqlens = args.seqlens * args.num_kv_heads

    q, k_cache, v_cache, cache_seqlens, block_table = _build_decode_inputs(
        args.block_size,
        args.head_size,
        args.num_kv_heads,
        args.num_kv_groups,
        args.seqlens,
        args.seed,
    )

    common_kwargs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "softmax_scale": 1.0 / math.sqrt(args.head_size),
        "num_splits": args.num_splits,
    }

    eager_out = flash_attn_with_kvcache_wrapper_eager(**common_kwargs)
    triton_out = flash_attn_with_kvcache_wrapper_triton(**common_kwargs)
    vllm_out = flash_attn_with_kvcache_wrapper_vllm(**common_kwargs)

    triton_max_abs_diff = (triton_out - eager_out).abs().max().item()
    triton_check_status = "passed" if torch.isclose(triton_out, eager_out, atol=args.atol, rtol=args.rtol).all() else "failed"

    vllm_max_abs_diff = (vllm_out - eager_out).abs().max().item()
    vllm_check_status = "passed" if torch.isclose(vllm_out, eager_out, atol=args.atol, rtol=args.rtol).all() else "failed"

    eager_ms = _benchmark(
        flash_attn_with_kvcache_wrapper_eager,
        args.warmup,
        args.iters,
        **common_kwargs,
    )
    triton_ms = _benchmark(
        flash_attn_with_kvcache_wrapper_triton,
        args.warmup,
        args.iters,
        **common_kwargs,
    )
    vllm_ms = _benchmark(
        flash_attn_with_kvcache_wrapper_vllm,
        args.warmup,
        args.iters,
        **common_kwargs,
    )

    triton_speedup = eager_ms / triton_ms
    vllm_speedup = eager_ms / vllm_ms

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("config:")
    print(
        f"  "
        f"num_kv_heads={args.num_kv_heads}, "
        f"num_kv_groups={args.num_kv_groups}, "
        f"block_size={args.block_size}, "
        f"head_size={args.head_size}, "
        f"num_splits={args.num_splits}"
    )
    print(f"  seqlens={args.seqlens}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print(
        "correctness: "
        f"triton={triton_check_status} (max_abs_diff={triton_max_abs_diff:.6f}), "
        f"vllm={vllm_check_status} (max_abs_diff={vllm_max_abs_diff:.6f})"
    )
    print()
    print(f"{'implementation':<16} {'avg_ms':>10}")
    print(f"{'-' * 16} {'-' * 10}")
    print(f"{'eager':<16} {eager_ms:>10.3f}")
    print(f"{'triton':<16} {triton_ms:>10.3f}")
    print(f"{'vllm':<16} {vllm_ms:>10.3f}")
    print()
    print(f"speedup (eager / triton): {triton_speedup:.2f}x")
    print(f"speedup (eager / vllm): {vllm_speedup:.2f}x")


if __name__ == "__main__":
    main()

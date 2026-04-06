import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pagedattention_eager import flash_attn_with_kvcache_wrapper_eager
from pagedattention_triton import flash_attn_with_kvcache_wrapper_triton


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark paged attention Triton vs eager implementations.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Input dtype for q/k/v tensors.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for the paged KV cache.",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=128,
        help="Head dimension.",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads.",
    )
    parser.add_argument(
        "--num-kv-groups",
        type=int,
        default=4,
        help="Number of query head groups per KV head.",
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        default="1000,7000,2000,4000,8000,5000,6000,3000",
        help="Comma-separated per-head sequence lengths.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations per implementation.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed iterations per implementation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to build the benchmark inputs.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the output correctness comparison.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance used for correctness checking.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
        help="Relative tolerance used for correctness checking.",
    )
    return parser.parse_args()


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
    if len(seqlens) != num_kv_heads:
        raise ValueError(f"Expected {num_kv_heads} sequence lengths, got {len(seqlens)}")

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


def _benchmark(fn, *, warmup, iters, **kwargs):
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for idx in range(iters):
        start_events[idx].record()
        fn(**kwargs)
        end_events[idx].record()

    torch.cuda.synchronize()
    latencies_ms = [start.elapsed_time(end) for start, end in zip(start_events, end_events)]
    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)

    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def _format_dtype(dtype):
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    dtype = getattr(torch, args.dtype)
    seqlens = [int(value) for value in args.seqlens.split(",") if value.strip()]
    softmax_scale = 1.0 / math.sqrt(args.head_size)

    q, k_cache, v_cache, cache_seqlens, block_table = _build_decode_inputs(
        dtype=dtype,
        block_size=args.block_size,
        head_size=args.head_size,
        num_kv_heads=args.num_kv_heads,
        num_kv_groups=args.num_kv_groups,
        seqlens=seqlens,
        seed=args.seed,
    )

    common_kwargs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "cache_seqlens": cache_seqlens,
        "block_table": block_table,
        "softmax_scale": softmax_scale,
    }

    eager_out = flash_attn_with_kvcache_wrapper_eager(**common_kwargs)
    triton_out = flash_attn_with_kvcache_wrapper_triton(**common_kwargs)
    torch.cuda.synchronize()

    max_abs_diff = (triton_out - eager_out).abs().max().item()
    check_status = "skipped"
    if not args.skip_check:
        torch.testing.assert_close(triton_out, eager_out, atol=args.atol, rtol=args.rtol)
        check_status = "passed"

    eager_stats = _benchmark(
        flash_attn_with_kvcache_wrapper_eager,
        warmup=args.warmup,
        iters=args.iters,
        **common_kwargs,
    )
    triton_stats = _benchmark(
        flash_attn_with_kvcache_wrapper_triton,
        warmup=args.warmup,
        iters=args.iters,
        **common_kwargs,
    )

    speedup = eager_stats["avg_ms"] / triton_stats["avg_ms"]

    print("Paged attention benchmark")
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"dtype: {_format_dtype(dtype)}")
    print(
        "config: "
        f"block_size={args.block_size}, head_size={args.head_size}, "
        f"num_kv_heads={args.num_kv_heads}, num_kv_groups={args.num_kv_groups}"
    )
    print(f"seqlens: {seqlens}")
    print(f"warmup={args.warmup}, iters={args.iters}, correctness={check_status}, max_abs_diff={max_abs_diff:.6f}")
    print()
    print(f"{'implementation':<16} {'avg_ms':>10} {'min_ms':>10} {'max_ms':>10}")
    print(f"{'-' * 16} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(
        f"{'eager':<16} "
        f"{eager_stats['avg_ms']:>10.3f} "
        f"{eager_stats['min_ms']:>10.3f} "
        f"{eager_stats['max_ms']:>10.3f}"
    )
    print(
        f"{'triton':<16} "
        f"{triton_stats['avg_ms']:>10.3f} "
        f"{triton_stats['min_ms']:>10.3f} "
        f"{triton_stats['max_ms']:>10.3f}"
    )
    print()
    print(f"speedup (eager / triton): {speedup:.2f}x")


if __name__ == "__main__":
    main()

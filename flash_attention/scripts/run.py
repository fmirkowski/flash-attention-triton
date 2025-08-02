from benchmark import benchmark_op
import gc
import argparse
import torch

def main():

    parser = argparse.ArgumentParser(description="Benchmark Flash Attention")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16) 
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--causal", action="store_true")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    batc
    print(f"\n=== Testing {args.batch_size}x{args.num_heads}x{args.seq_len}x{args.head_dim} ===")

    
    benchmark_op(BATCH_SIZE=args.batch_size, NUM_HEADS=args.num_heads, SEQ_LEN=args.seq_len, HEAD_DIM=args.head_dim, causal=args.causal)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

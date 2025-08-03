import torch
import torch.nn.functional as F
import triton
from ..ops.attention import TritonAttention

def benchmark_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal_(mean=0.0, std=0.5).requires_grad_()
    K = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal_(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal_(mean=0.0, std=0.5).requires_grad_()
    softmax_scale = 1 / (HEAD_DIM**0.5)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== CRITICAL: Proper warmup for both PyTorch and Triton =====
    print("🔥 Warming up kernels (including Triton autotuning)...")
    
    # PyTorch warmup
    for _ in range(10):
        _ = torch.matmul(Q, K.transpose(2, 3))
    
    # Optimized PyTorch warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=softmax_scale)
    
    # TRITON WARMUP - This triggers autotuning and caches optimal configs
    for _ in range(10):
        # Warmup forward pass
        tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
        # Warmup backward pass  
        tri_out.backward(torch.randn_like(tri_out), retain_graph=True)
        # Clear gradients
        V.grad = K.grad = Q.grad = None
    
    print("✅ Warmup complete - autotuning cached, ready for fair timing!")
    
    # Pre-create mask outside timing loop for fair naive comparison
    
    torch_naive_times_fwd = []
    torch_naive_times_bwd = []
    torch_opt_times_fwd = []
    torch_opt_times_bwd = []
    # Clear CUDA cache before benchmarking
    # torch.cuda.empty_cache()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Benchmark naive PyTorch implementation
    torch.cuda.synchronize()
    for _ in range(500):
        torch.cuda.synchronize()
        V.grad = K.grad = Q.grad = None

        start_event.record()
        dO = torch.randn_like(Q)

        # Naive torch implementation
        P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
        if causal:
            MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).to(device)
            P[:, :, MASK == 0] = float('-inf')
        P = torch.softmax(P, dim=-1).half()
        naive_O = torch.matmul(P, V)

        end_event.record()
        torch.cuda.synchronize()
        torch_naive_times_fwd.append(start_event.elapsed_time(end_event))

        start_event.record()
        naive_O.backward(dO)
        end_event.record()
        torch.cuda.synchronize()
        torch_naive_times_bwd.append(start_event.elapsed_time(end_event))

    # Benchmark optimized PyTorch implementation
    torch.cuda.synchronize()
    for _ in range(500):
        torch.cuda.synchronize()
        V.grad = K.grad = Q.grad = None

        start_event.record()
        dO = torch.randn_like(Q)

        # Optimized PyTorch implementation using SDPA
        opt_O = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=softmax_scale)

        end_event.record()
        torch.cuda.synchronize()
        torch_opt_times_fwd.append(start_event.elapsed_time(end_event))

        start_event.record()
        opt_O.backward(dO)
        end_event.record()
        torch.cuda.synchronize()
        torch_opt_times_bwd.append(start_event.elapsed_time(end_event))

    # Benchmark Triton implementation
    triton_times_fwd = []
    triton_times_bwd = []

    for _ in range(500):
        torch.cuda.synchronize()
        V.grad = K.grad = Q.grad = None

        start_event.record()
        tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()

        end_event.record()
        torch.cuda.synchronize()
        triton_times_fwd.append(start_event.elapsed_time(end_event))

        start_event.record()
        tri_out.backward(dO)
        end_event.record()
        torch.cuda.synchronize()
        triton_times_bwd.append(start_event.elapsed_time(end_event))

    # Calculate means
    torch_naive_fwd_mean = sum(torch_naive_times_fwd) / len(torch_naive_times_fwd)
    torch_naive_bwd_mean = sum(torch_naive_times_bwd) / len(torch_naive_times_bwd)
    torch_opt_fwd_mean = sum(torch_opt_times_fwd) / len(torch_opt_times_fwd)
    torch_opt_bwd_mean = sum(torch_opt_times_bwd) / len(torch_opt_times_bwd)
    triton_fwd_mean = sum(triton_times_fwd) / len(triton_times_fwd)
    triton_bwd_mean = sum(triton_times_bwd) / len(triton_times_bwd)
    
    print(f"\n=== Timing Results (ms) ===")
    print(f"📊 Forward Pass:")
    print(f"  Naive PyTorch:    {torch_naive_fwd_mean:.3f} ± {(max(torch_naive_times_fwd) - min(torch_naive_times_fwd))/2:.3f}")
    print(f"  Optimized PyTorch: {torch_opt_fwd_mean:.3f} ± {(max(torch_opt_times_fwd) - min(torch_opt_times_fwd))/2:.3f}")
    print(f"  Triton Flash:     {triton_fwd_mean:.3f} ± {(max(triton_times_fwd) - min(triton_times_fwd))/2:.3f}")
    print()
    print(f"📊 Backward Pass:")
    print(f"  Naive PyTorch:    {torch_naive_bwd_mean:.3f} ± {(max(torch_naive_times_bwd) - min(torch_naive_times_bwd))/2:.3f}")
    print(f"  Optimized PyTorch: {torch_opt_bwd_mean:.3f} ± {(max(torch_opt_times_bwd) - min(torch_opt_times_bwd))/2:.3f}")
    print(f"  Triton Flash:     {triton_bwd_mean:.3f} ± {(max(triton_times_bwd) - min(triton_times_bwd))/2:.3f}")
    print()
    print(f"🚀 Speedup vs Naive PyTorch:")
    print(f"  Forward:  {torch_naive_fwd_mean/triton_fwd_mean:.1f}x")
    print(f"  Backward: {torch_naive_bwd_mean/triton_bwd_mean:.1f}x")
    print(f"  Total:    {(torch_naive_fwd_mean + torch_naive_bwd_mean)/(triton_fwd_mean + triton_bwd_mean):.1f}x")
    print()
    print(f"⚡ Speedup vs Optimized PyTorch:")
    print(f"  Forward:  {torch_opt_fwd_mean/triton_fwd_mean:.1f}x")
    print(f"  Backward: {torch_opt_bwd_mean/triton_bwd_mean:.1f}x")
    print(f"  Total:    {(torch_opt_fwd_mean + torch_opt_bwd_mean)/(triton_fwd_mean + triton_bwd_mean):.1f}x")
    print()
    print(f"📈 Total Times:")
    print(f"  Naive PyTorch:    {torch_naive_fwd_mean + torch_naive_bwd_mean:.3f} ms")
    print(f"  Optimized PyTorch: {torch_opt_fwd_mean + torch_opt_bwd_mean:.3f} ms")
    print(f"  Triton Flash:     {triton_fwd_mean + triton_bwd_mean:.3f} ms")


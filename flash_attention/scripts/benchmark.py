import torch
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
    # dO = torch.randn_like(Q)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ===== CRITICAL: Proper warmup for both PyTorch and Triton =====
    print("🔥 Warming up kernels (including Triton autotuning)...")
    
    # PyTorch warmup
    for _ in range(10):
        _ = torch.matmul(Q, K.transpose(2, 3))
    
    # TRITON WARMUP - This triggers autotuning and caches optimal configs
    for _ in range(10):
        # Warmup forward pass
        tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
        # Warmup backward pass  
        tri_out.backward(torch.randn_like(tri_out), retain_graph=True)
        # Clear gradients
        V.grad = K.grad = Q.grad = None
    
    print("✅ Warmup complete - autotuning cached, ready for fair timing!")
    torch_times_fwd = []
    torch_times_bwd = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(500):
        torch.cuda.synchronize()
        V.grad = K.grad = Q.grad = None

        start_event.record()
        softmax_scale = 1 / (HEAD_DIM**0.5)
        dO = torch.randn_like(Q)

        # torch implementation
        MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).to(device)
        P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
        if causal:
            P[:, :, MASK == 0] = float('-inf')
        P = torch.softmax(P, dim=-1).half()
        ref_O = torch.matmul(P, V)

        end_event.record()
        torch.cuda.synchronize()
        torch_times_fwd.append(start_event.elapsed_time(end_event))


        start_event.record()
        ref_O.backward(dO)
        end_event.record()
        torch.cuda.synchronize()
        torch_times_bwd.append(start_event.elapsed_time(end_event))

    triton_times_fwd = []
    triton_times_bwd = []

    for _ in range(500):
        torch.cuda.synchronize()
        V.grad = K.grad = Q.grad = None
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

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

    torch_fwd_mean = sum(torch_times_fwd) / len(torch_times_fwd)
    torch_bwd_mean = sum(torch_times_bwd) / len(torch_times_bwd)
    triton_fwd_mean = sum(triton_times_fwd) / len(triton_times_fwd)
    triton_bwd_mean = sum(triton_times_bwd) / len(triton_times_bwd)
    
    print(f"\n=== Timing Results (ms) ===")
    print(f"PyTorch Forward:  {torch_fwd_mean:.3f} ± {(max(torch_times_fwd) - min(torch_times_fwd))/2:.3f}")
    print(f"Triton Forward:   {triton_fwd_mean:.3f} ± {(max(triton_times_fwd) - min(triton_times_fwd))/2:.3f}")
    print(f"Forward Speedup:  {torch_fwd_mean/triton_fwd_mean:.2f}x")
    print()
    print(f"PyTorch Backward: {torch_bwd_mean:.3f} ± {(max(torch_times_bwd) - min(torch_times_bwd))/2:.3f}")
    print(f"Triton Backward:  {triton_bwd_mean:.3f} ± {(max(triton_times_bwd) - min(triton_times_bwd))/2:.3f}")
    print(f"Backward Speedup: {torch_bwd_mean/triton_bwd_mean:.2f}x")
    print()
    print(f"Total PyTorch:    {torch_fwd_mean + torch_bwd_mean:.3f}")
    print(f"Total Triton:     {triton_fwd_mean + triton_bwd_mean:.3f}")
    print(f"Total Speedup:    {(torch_fwd_mean + torch_bwd_mean)/(triton_fwd_mean + triton_bwd_mean):.2f}x")


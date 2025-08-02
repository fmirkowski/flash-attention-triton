import torch
import triton
import argparse
from ..ops.attention import TritonAttention

def run_test():
    parser = argparse.ArgumentParser(description="Test Flash Attention")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--causal", action="store_true")
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    NUM_HEADS = args.num_heads
    SEQ_LEN = args.seq_len
    HEAD_DIM = args.head_dim
    causal = args.causal
    dtype = torch.float16
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
    dO = torch.randn_like(Q)

    device = 'cuda' if torch.cuda.is_available else 'cpu'
        # warmup:



    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).to(device)
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P, dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)

    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None # Zeroing out the gradients, cloning them to refference
    ref_dQ, Q.grad = Q.grad.clone(), None



    # triton implementation

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None # Zeroing out the gradients, cloning them to triference
    tri_dQ, Q.grad = Q.grad.clone(), None


    # compare 

    rtol = 0.0
    atol = 1e-2
    # Check if outputs match within tolerance
    output_match = torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    dk_match = torch.allclose(ref_dK, tri_dK, rtol=rtol, atol=atol) 
    dq_match = torch.allclose(ref_dQ, tri_dQ, rtol=rtol, atol=atol)
    dv_match = torch.allclose(ref_dV, tri_dV, rtol=rtol, atol=atol)

    if output_match and dk_match and dq_match and dv_match:
        print("\n✅ All tests passed! 🎉")
    else:
        print("\n❌ Tests failed! 😢")
        if not output_match:
            print("❌ Output values do not match within tolerance")
        if not dk_match:
            print("❌ dK values do not match within tolerance")
        if not dq_match:
            print("❌ dQ values do not match within tolerance") 
        if not dv_match:
            print("❌ dV values do not match within tolerance")


    # Check shapes match
    assert ref_O.shape == tri_out.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    assert ref_dK.shape == tri_dK.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    assert ref_dQ.shape == tri_dQ.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) 
    assert ref_dV.shape == tri_dV.shape == (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    Check mask consistency
    if causal:
        mask_sum = MASK.sum()
        expected_sum = (SEQ_LEN * (SEQ_LEN + 1)) / 2  # Sum of lower triangular matrix
        assert mask_sum.item() == expected_sum, "Causal mask is not consistent"

    # Print max differences
    print("Reference output max diff:", (ref_O - tri_out).abs().max().item())
    print("Reference dK max diff:", (ref_dK - tri_dK).abs().max().item()) 
    print("Reference dQ max diff:", (ref_dQ - tri_dQ).abs().max().item())
    print("Reference dV max diff:", (ref_dV - tri_dV).abs().max().item())


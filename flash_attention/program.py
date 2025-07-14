import torch
import triton
import triton.language as tl


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM == K.shape[-1] and K.shape[-1] == V.shape[-1]

        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        # shapes, asssert shapes equations
        # define grid for triton, what we want to parallerlise, blocks, grid of programs , how many independent gpu threads should w elanuch? 

        # lambda returns a tuple with guidance on parallel exec of grid
        # this grid basically tells us how many prgrams can we launch that can work in parallel        
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']),
            BATCH_SIZE * NUM_HEADS,
            1 # Z dim in CUDA, we dont want to use it for now
        )
def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal(mean=0.0, std=0.5).requires_grad_()
    K = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal(mean=0.0, std=0.5).requires_grad_()
    V = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device='cuda'
                    ).normal(mean=0.0, std=0.5).requires_grad_()
    

    softmax_scale = 1 / (HEAD_DIM**0.5)
    d0 = torch.randn_like(Q)

    # torch implementation

    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN), device='cuda')
    P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P, dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(d0)
    
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None # Zeroing out the gradients, cloning them to refference
    ref_dQ, Q.grad = Q.grad.clone(), None



    # triton implementation

    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None # Zeroing out the gradients, cloning them to triference
    tri_dQ, Q.grad = Q.grad.clone(), None


    # compare 

    rtol = 0.0
    atol = 1e-2

    assert torch.allclose(ref_O, tri_out, atol, rtol)
    assert torch.allclose(ref_dK, tri_dK, atol, rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol, rtol)
    assert torch.allclose(ref_dV, tri_dV, atol, rtol)


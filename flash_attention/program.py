import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, M, softmax_scale, causal,
                stride_Q_batch, stride_Q_heads, stride_Q_seq, stride_Q_dim,
                stride_K_batch, stride_K_heads, stride_K_seq, stride_K_dim,
                stride_V_batch, stride_V_heads, stride_V_seq, stride_V_dim,
                stride_O_batch, stride_O_heads, stride_O_seq, stride_O_dim,
                BATCH_SIZE, NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, 
                HEAD_DIM: tl.constexpr, STAGE: tl.constexpr, 
                BLOCK_SIZE_KV: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr):
    
    #? tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)



    
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
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), #numbe of blocks in Q – which block are we going to work with
            BATCH_SIZE * NUM_HEADS,
            1 # Z dim in CUDA, we dont want to use it for now
        )
        
        # | so the number of parallel progframs is BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q
        
        # We also need M matrix which will be the log sumexp (max for each row)
        M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=torch.float32, device=Q.device)
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            M=M,
            softmax_scale=softmax_scale,
            causal=causal,
            stride_Q_batch=Q.stride[0], #Because we only need a pointer in triton so we have to figure out, thats why we're passing the stride
            stride_Q_heads=Q.stride[1],
            stride_Q_seq=Q.stride[2],
            stride_Q_dim=Q.stride[3],

            stride_K_batch=K.stride[0],
            stride_K_heads=K.stride[1],
            stride_K_seq=K.stride[2],
            stride_K_dim=K.stride[3],

            stride_V_batch=V.stride[0],
            stride_V_heads=V.stride[1],
            stride_V_seq=V.stride[2],
            stride_V_dim=V.stride[3],

            stride_O_batch=O.stride[0],
            stride_O_heads=O.stride[1],
            stride_O_seq=O.stride[2],
            stride_O_dim=O.stride[3],

            # Same dimeniosns actually

            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal





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


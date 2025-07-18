import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q: tl.constexpr,
            BLOCK_SIZE_KV: tl.constexpr,
            STAGE: tl.constexpr,
            offsets_q,
            offsets_kv,
            SEQ_LEN: tl.constexpr):
    # Now we will define the lower and higher index that this particular stage should be working with, remeber this is the inner loop so its enough for us to define just the lower and higer for it
    if STAGE == 1:
        lower, higher = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lower, higher = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lower = tl.multiple_of(lower, BLOCK_SIZE_Q) # tells tthose are multiples, so triton can do optimzations
    else:
        lower, higher = 0, SEQ_LEN

    # point kv blocks to the first block (advance), we're moving them to lower
    K_block_ptr = tl.advance(K_block_ptr, (0, lower))
    V_block_ptr = tl.advance(V_block_ptr, (lower, 0))
    
    # loop from lower to higher, load QK, load K (not V now), dot product, compute mask , apply mask + softmax ascale on qk block, compute running maximum
    for start_kv in range(lower, higher, BLOCK_SIZE_KV):
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        # differ?
        if STAGE == 2:
            mask = offsets_q[:, None] >= (start_kv + offsets_kv[None, :]) # because we're iterating on many blocks
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6) # we need a float and tl.where creates a this, 1, BLOCK_SIZE vector
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))
        else: 
            QK_block = QK_block * softmax_scale
            m_ij = tl.maximum(m_i, tl.max(QK_block, axis=1))

        P_block = tl.math.exp(QK_block - m_ij[:, None])
        alpha = tl.math.exp((m_i - m_ij))
        l_i = l_i * alpha + tl.sum(P_block, 1)

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None] # Why am i doing None? because otherwise it wouldnt be able to do element wise (ie shape would mismatch) element wise is equivalent to diagonal
        O_block = tl.dot(P_block, V_block, O_block) # equivalent too O*alpha + P @ V

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i



@triton.jit
def _attn_fwd(Q, K, V, O, M, softmax_scale, causal, #pointers
                stride_Q_batch, stride_Q_heads, stride_Q_seq, stride_Q_dim,
                stride_K_batch, stride_K_heads, stride_K_seq, stride_K_dim,
                stride_V_batch, stride_V_heads, stride_V_seq, stride_V_dim,
                stride_O_batch, stride_O_heads, stride_O_seq, stride_O_dim,
                BATCH_SIZE, NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, 
                HEAD_DIM: tl.constexpr, STAGE: tl.constexpr, 
                BLOCK_SIZE_KV: tl.constexpr, BLOCK_SIZE_Q: tl.constexpr):
    
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    # specify proigram id, which block in the sequence to process (tl.program_id(0), 1)
    # Remember, here: a specfic ONE program is launched, a combination of the grid (of shape of the tuple and bounnds)
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1) # flattened, combinations of some BATCH_SIZE[i] and NUM_HEADS[j]
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    # REMEMBER, NOTE When we pass the tensor to triton it just gets a pointer, we just get the starting pointer of it
    # but then how do we access the tensor? (offset)

    # this allows us to get (SEQ_LEN, HEAD_DIM) block in q, k, v by selecting index by batch and head
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch,
         + index_head.to(tl.int64) * stride_Q_heads
    )
    # pointer to the right index in the block of queries
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset, # take a pointer, pointing at particular batch and there particular head it should be working with Q[index_batch, index_head, :, :]
        shape=(SEQ_LEN, HEAD_DIM),  
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # whats the difference between offsets and base – basically the same but in offset we specify index which is more convenient, we could also move all offsets to base and just multiply by coresponding strides and vice-versa
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(0,1)
    )

    # needs to be transposed

    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset, # this is basically a memory address to the tensor we want to operate with (otherwise there is no way to identify!!  and we are adding offset to identify a specific batch and index;)
        shape=(HEAD_DIM, SEQ_LEN),  
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0), # 
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0,1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset, 
        shape=(SEQ_LEN, HEAD_DIM),  
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),  # why 0,0?
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(0,1)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset, 
        shape=(SEQ_LEN, HEAD_DIM),  
        strides=(stride_O_seq, stride_O_dim),
        offsets=(BLOCK_SIZE_Q * block_index_q, 0),  
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(0,1)
    )

    # but wait how does triton select which porgram to work with?

    # We parrallelise each program by the query block (right now we just select the right q block)

    offsets_q = BLOCK_SIZE_Q * block_index_q + tl.arrange(0, BLOCK_SIZE_Q) # loading particular queries
    offsets_kv = tl.arrange(0, BLOCK_SIZE_KV) # loading particular queries, this is not a specific place in memory, remember – its just offset, where do we have to move (ie index) thats why it can be the same for both key and value
    # running maximum for each query block Q_i
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)
    
    O_block = tl.load(O_block_ptr)
    #  initlaisatpon
    # we'll load K and V later, we'll load specific blocks to save on compute and not move the whole thing at once
    
    # and inenr loop now, with max computing, split into 2 steps - causal and non causal, we either have to compute it or not, we first do nomral attention non-causal, and basically skip all and mask it all out
    
    # hint for init:
    # we are calling twice for software pipelining
    # 3 for casual 1 for non casual

    '''
    NOTE:
    So the pipeline is:
        if attention is casual then:
            compute the left side and after that the diagonal (in the second iteration)
        if attention is non-casual then:
            compute all and skip the second function (no casual is 1)
    '''



    if STAGE == 3 or STAGE == 1:
        # in this step we're doing the causal attention or even non causal for the blocks that are on the left (we still have to do it in both cases, in one case it will be enough and the rest should be -inf (in the causal) and in the non causal we will still have to compute evrything)
        O_block, l_i, m_i = _attn_fwd_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                4 - STAGE,
                offsets_q,
                offsets_kv,
                SEQ_LEN
            )
    
    if STAGE == 3:
    # and in this step we're finsishing the diagonal of the diagonal for casual
        O_block, l_i, m_i = _attn_fwd_inner(
                O_block,
                l_i,
                m_i,
                Q_block,
                K_block_ptr,
                V_block_ptr,
                block_index_q,
                softmax_scale,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_KV,
                2,
                offsets_q,
                offsets_kv,
                SEQ_LEN
            )

    # The goal of logsumexp is for us to not haev to recompute the params of the softmax  during backward pass
    # remember we need to divide by l_i 
    O_block = O_block / l_i[:, None]
    # then we need to access the pointer to m to specific qoffset q and seq len
    m_ptrs = M + index_batch_head * SEQ_LEN + offsets_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

    # store everyhting 
  

    
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
            O=O,
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


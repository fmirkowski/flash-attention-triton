import torch
import os
os.environ(['TRITON_PRINT_AUTOTUNING']) = '1'
import triton
import triton.language as tl


@triton.autotune(
        [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_KV in [32, 64]
            for BLOCK_SIZE_KV in [64, 128]
            for num_stages in [3, 4, 7]
            for num_warps in [2, 4]
        ],
        key=["SEQ_LEN", "HEAD_DIM"]
)

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
            mask = offsets_q[:, None] >= (start_kv + offsets_kv)[None, :] # because we're iterating on many blocks
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
    m_i += tl.math.log(l_i) # why?
    O_block = O_block / l_i[:, None]
    # then we need to access the pointer to m to specific qoffset q and seq len
    m_ptrs = M + index_batch_head * SEQ_LEN + offsets_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

    # store everyhting 
  

@triton.jit
def _attn_bwd_preprocess(O, dO, D, BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr):
    # get the block of q indices, load O_block by composing it manually, read dO block and just add those into D_block, store D_block
    # get the offset_batch_head and evertything we neeed from program_id
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    O += batch_head_index * O.stride(2)
    # D += batch_head_index * O.stride(2)
    dO += batch_head_index * O.stride(2)
    dO += block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None] * O.stride(3) + tl.arange(0, HEAD_DIM)[None, :]
    # D += block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None] * O.stride(3) + tl.arange(0, HEAD_DIM)[None, :]
    O += block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None] * O.stride(3) + tl.arange(0, HEAD_DIM)[None, :]

    # D_block = tl.load(D).to(tl.float32) No need to load D we dont need its contents, its just a mem pinter for us
    O_block = tl.load(O).to(tl.float32)
    dO_block = tl.load(dO).to(tl.float32)
    
    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptr = D + block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) * O.stride(3) # we dont need head dim stuff because its a scalr
    tl.store(D_block_ptr, D_block)
# Pre process keernel , load programs, load manually the blocks O, dO, gt to th right points to whatg w want to operate with 
# Why do we need to do it tho?


@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K, # SHAPE: [batch_index, heads_index, block_index:block_index+BLOCK_KV (mem address), 0:HEAD_DIM (we want all, and we specify that in the arrange)]
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,): # commas at the end are a good practise
    
    # use comments for visualising shapes of what we actually process
    block_index_kv = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    
    batch_index = batch_head_index // NUM_HEADS
    head_index = batch_head_index % NUM_HEADS

    batch_head_seq_offset = (batch_head_index * SEQ_LEN).to(tl.int64)
    M += batch_head_seq_offset # those are currently pointers
    D += batch_head_seq_offset
    # create a kv block
    K += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    V += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    Q += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dO += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    M += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dQ += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dV += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dK += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    # [:, None] is the same thing as unsqueeze

    # we need to add the arange because block_index_kv * BLOCK_KV (specific row, where it starts) + tl.arange(0, BLOCK_KV)[:, None] – specyfing all rows that we will need to cover * stride_seq – specific memory addresses of those rows
    offsets_dim = tl.arange(0, HEAD_DIM)
    kv_start_block =  block_index_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)[:, None] * stride_seq # START ARANGE FROM 0 
    K_block = tl.load(K+kv_start_block + offsets_dim[None, :] * stride_head) # creates a 2D set of addresses of the block with the addtition!
    V_block = tl.load(V+kv_start_block + offsets_dim[None, :] * stride_head)
    dK_block = tl.empty_like(K_block)
    dV_block = tl.empty_like(V_block)
    offsets_q = tl.arange(0, BLOCK_Q)
    # Do the same for qT and dO ptrs (for backward thorugh matmul), load it transposed because its more efficient
    qT_ptrs = Q+offsets_q[None, :] * stride_seq + offsets_dim[:, None] * stride_dim 
    dO_ptrs = dO+offsets_q[:, None] * stride_seq + offsets_dim[None, :] * stride_dim 
    current_q = 0 # later in the loop just update it

    num_steps = SEQ_LEN // BLOCK_Q

    for step in range(num_steps):
        qT_block = tl.load(qT_ptrs) # we will be advancing it later! like in forward pass
        dO_block = tl.load(dO_ptrs)
        offsets_q = current_q + tl.arange(0, BLOCK_Q)
        M_block = tl.load(M + offsets_q)
        sT = softmax_scale * tl.dot(K_block, qT_block)
        pT = tl.maht.exp(sT - M_block[None, :]) # because element wise

        if STAGE == 3:
            # mask where with zeros
            mask = (offsets_q[:, None] >= (block_index_kv * BLOCK_KV + tl.arange(0, BLOCK_KV))[None, :]) # current?
            pT = tl.where(mask, pT, 0.0)
            
        dO_block = tl.load(dO_ptrs)
        # formula for d_Vblock form paper
        dV_block += tl.dot(pT.to(tl.float16), dO) 
        D_block = tl.load(D+offsets_q)
        dpT = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T = pT * (dpT - D_block).to(tl.float16)
        dK_block += softmax_scale * tl.dot(dS_T, tl.trans(qT_block))
        
        
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
        current_q += BLOCK_Q
        #why did we do tl.advance later?

    # store dV and dK, write those back to HBM
    dK_block_ptr = dK + kv_start_block[:, None] + offsets_dim[None, :] * stride_dim
    tl.store(dK_block_ptr, dK_block)
    dV_block_ptr = dV + kv_start_block[:, None] + offsets_dim[None, :] * stride_dim
    tl.store(dV_block_ptr, dV_block)
    




@triton.jit
def _attn_bwd_dq(Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,): 
    


    # use comments for visualising shapes of what we actually process
    block_index_q = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    
    batch_index = batch_head_index // NUM_HEADS
    head_index = batch_head_index % NUM_HEADS

    batch_head_seq_offset = (batch_head_index * SEQ_LEN).to(tl.int64)
    M += batch_head_seq_offset # those are currently pointers
    D += batch_head_seq_offset
    # create a kv block
    K += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    V += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    Q += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dO += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dQ += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dV += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)
    dK += (batch_index * stride_batch + head_index * stride_head).to(tl.int64)

    # load Q block with head idm too, aprticular q block
    offs_q = block_index_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    Q_block = tl.load((Q +
                      offs_q[:, None] * stride_seq + 
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim).to(tl.int64))
    dO_block = tl.load((dO +
                      offs_q[:, None] * stride_seq + 
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim).to(tl.int64)) 
    
    M_block = tl.load((M +
                      offs_q).to(tl.int64)) 
    
    # init D_block
    D_block = tl.load(D + offs_q[:, None] * stride_seq + 
                    tl.arange(0, HEAD_DIM)[None, :] * stride_dim).to(tl.int64)

    
    dQ_block = tl.zeros_like(Q_block, dtype=tl.float32)
    offset_kv = tl.arange(0, BLOCK_KV)
    # access k and v pointers as transposed blovk, load them transposed because its then free 
    K_T_block_ptr = K + offset_kv[None, :] * stride_seq + tl.arange(0, HEAD_DIM)[:, None] * stride_dim
    V_T_block_ptr = V + offset_kv[None, :] * stride_seq + tl.arange(0, HEAD_DIM)[:, None] * stride_dim
    
   
    # Why does this loop have to go trhough KV related number of blocks? - because as in earlier kernel we are now iterating through every k and v - lets find out why tho ;)

    num_steps = NUM_HEADS // BLOCK_KV
    curr_kv = 0
    for step in range(num_steps):
        K_T = tl.load(K_T_block_ptr)
        V_T = tl.load(V_T_block_ptr)
        S_block = softmax_scale * tl.dot(Q_block, K_T)
        P_block = tl.math.exp(S_block - M_block)

        if STAGE == 3:
            offset_kv += curr_kv
            mask = (
                offs_q[:, None] >= offset_kv[None, :]
            )
            P_block = tl.where(mask, P_block, 0.0)
        dP_block = tl.dot(dO_block, V_T).to(tl.float32)
        dS_block = P_block * (dP_block - D_block) # why are we not advancing D?
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T))
        
        # mask 
        # compute dQ as in paper dS 
        # movepointers and then store
        K_T_block_ptr += BLOCK_KV * stride_seq
        V_T_block_ptr += BLOCK_KV * stride_seq
        curr_kv += BLOCK_KV

    dQ_block_ptr = (dQ + offs_q[:, None] * stride_seq + 
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim).to(tl.int64)
    
    tl.store(dQ_block_ptr, dQ_block)
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
            1,  # Z dim in CUDA, we dont want to use it for now
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

        ctx.save_for_backward(Q, K, V, O, M) # we dont want to save product s of thosee (ex: QK^T), ecause tl.store in the hbm would be veery eexpensivee, more optimal to compute on thee fly
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal


    # chain rule is bascially product of gradients before 
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        # makee dO assert contigous, asserte strides
        assert dO.is_contigous()
        assert Q.stride == K.stride == V.stride == O.stride == dO.stride
        # init dQ, DK, dV
        dQ = torch.empty_like(Q)
        dV = torch.empty_like(V)
        dK = torch.empty_like(K)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128 # M/4d in the paper, because M is the total capacity for scalars in SRAM and 4d becausue we split to 4 blocks lol




        
        # Pre procss kernel, goal 
        
        preprocess_grid = lambda arg: (
            tl.cdiv(SEQ_LEN, BLOCK_SIZE_MACRO),
            BATCH_SIZE * NUM_HEADS,
            1
        )
        D = torch.empty(M)
        # finish preprocess later when you know what its needed for ;)
        _attn_bwd_preprocess[preprocess_grid](O=O, dO=dO, D=D, BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=ctx.HEAD_DIM)
        
        dk_dv_grid = (
            tl.cdiv(SEQ_LEN, BLOCK_SIZE_MICRO), # why micro tho?
            NUM_HEADS*BATCH_SIZE,
            1
        )

        stage = 3 if ctx.causal else 1
        
        _attn_bwd_dk_dv[dk_dv_grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=K.stride(0),
            stride_head=K.stride(1),
            stride_seq=K.stride(2),
            stride_dim=K.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,)
        

        _attn_bwd_dq[dk_dv_grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=K.stride(0),
            stride_head=K.stride(1),
            stride_seq=K.stride(2),
            stride_dim=K.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,)
        
     


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
    dO = torch.randn_like(Q)

    # torch implementation

    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN), device='cuda')
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

    assert torch.allclose(ref_O, tri_out, atol, rtol)
    assert torch.allclose(ref_dK, tri_dK, atol, rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol, rtol)
    assert torch.allclose(ref_dV, tri_dV, atol, rtol)


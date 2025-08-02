import torch
import os
import triton
import triton.language as tl



@triton.jit
def _attn_bwd_preprocess(O, dO, D, SEQ_LEN, BLOCK_SIZE_Q: tl.constexpr, HEAD_DIM: tl.constexpr,):
    # get the block of q indices, load O_block by composing it manually, read dO block and just add those into D_block, store D_block
    # get the offset_batch_head and evertything we neeed from program_id
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    O += batch_head_index * SEQ_LEN * HEAD_DIM
    D += batch_head_index * SEQ_LEN
    dO += batch_head_index * SEQ_LEN * HEAD_DIM
    dO += (block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]) * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    O += (block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]) * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]

    # D_block = tl.load(D).to(tl.float32) No need to load D we dont need its contents, its just a mem pinter for us
    O_block = tl.load(O).to(tl.float32)
    dO_block = tl.load(dO).to(tl.float32)
    
    D_block = tl.sum(dO_block * O_block, axis=1)
    D_block_ptr = D + block_index * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # we dont need head dim stuff because its a scalr
    tl.store(D_block_ptr, D_block)
    # Pre process keernel , load programs, load manually the blocks O, dO, gt to th right points to whatg w want to operate with 
    # Why do we need to do it tho?
@triton.jit
def _attn_bwd_dk_dv(
    Q, 
    K, # SHAPE: [batch_index, heads_index, block_index:block_index+BLOCK_KV (mem address), 0:HEAD_DIM (we want all, and we specify that in the arange)]
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
    batch_head_index = tl.program_id(2)
    
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
    # [:, None] is the same thing as unsqueeze(1)

    # we need to add the arange because block_index_kv * BLOCK_KV (specific row, where it starts) + tl.arange(0, BLOCK_KV)[:, None] – specyfing all rows that we will need to cover * stride_seq – specific memory addresses of those rows
    offsets_dim = tl.arange(0, HEAD_DIM)
    kv_start_block =  (block_index_kv * BLOCK_KV + tl.arange(0, BLOCK_KV)[:, None]) * stride_seq # START ARANGE FROM 0 
    K_block = tl.load(K+kv_start_block + offsets_dim[None, :] * stride_dim) # creates a 2D set of addresses of the block with the addtition!
    V_block = tl.load(V+kv_start_block + offsets_dim[None, :] * stride_dim)
    dK_block = tl.zeros_like(K_block).to(tl.float32)
    dV_block = tl.zeros_like(V_block).to(tl.float32)
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
        pT = tl.math.exp(sT - M_block[None, :]) # because element wise

        if STAGE == 3:
            # mask where with zeros
            mask = (offsets_q[None, :] >= (block_index_kv * BLOCK_KV + tl.arange(0, BLOCK_KV))[:, None]) # current?
            pT = tl.where(mask, pT, 0.0)
        dO_block = tl.load(dO_ptrs)
        # formula for d_Vblock form paper
        dV_block += tl.dot(pT, dO_block.to(tl.float32))
        D_block = tl.load(D+offsets_q)
        dpT = tl.dot(V_block, tl.trans(dO_block))
        dS_T = pT * (dpT - D_block)
        dK_block += softmax_scale * tl.dot(dS_T, tl.trans(qT_block).to(tl.float32)) # float32?
        
        
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
        # why not add stride 
        current_q += BLOCK_Q
        
        #why did we do tl.advance later?

    # store dV and dK, write those back to HBM
    dK_block_ptr = dK + kv_start_block + offsets_dim[None, :] * stride_dim
    tl.store(dK_block_ptr, dK_block)
    dV_block_ptr = dV + kv_start_block + offsets_dim[None, :] * stride_dim
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
    batch_head_index = tl.program_id(2)
    
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
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim))
    dO_block = tl.load((dO +
                      offs_q[:, None] * stride_seq + 
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim))
    
    M_block = tl.load((M +
                      offs_q)) 
    
    # init D_block
    D_block = tl.load(D + offs_q)

    
    dQ_block = tl.zeros_like(Q_block).to(tl.float32)
    offset_kv = tl.arange(0, BLOCK_KV)
    # access k and v pointers as transposed blovk, load them transposed because its then free 
    K_T_block_ptr = K + offset_kv[None, :] * stride_seq + tl.arange(0, HEAD_DIM)[:, None] * stride_dim
    V_T_block_ptr = V + offset_kv[None, :] * stride_seq + tl.arange(0, HEAD_DIM)[:, None] * stride_dim
    
   
    # Why does this loop have to go trhough KV related number of blocks? - because as in earlier kernel we are now iterating through every k and v - lets find out why tho ;)

    num_steps = SEQ_LEN // BLOCK_KV
    curr_kv = 0
    for block_idx in range(num_steps):
        K_T = tl.load(K_T_block_ptr)
        V_T = tl.load(V_T_block_ptr)
        S_block = softmax_scale * tl.dot(Q_block, K_T) # [BLOCK_Q, BLOCK_KV] => [128, 32]
        
        P_block = tl.math.exp(S_block - M_block[:, None]) 

        if STAGE == 3:
            offset_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask = (
                offs_q[:, None] >= offset_kv[None, :]
            )
            P_block = tl.where(mask, P_block, 0.0)
        dP_block = tl.dot(dO_block, V_T)
        dS_block = P_block * (dP_block - D_block[:, None])
        
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T).to(tl.float32))
        
        # mask 
        # compute dQ as in paper dS 
        # movepointers and then store
        K_T_block_ptr += BLOCK_KV * stride_seq
        V_T_block_ptr += BLOCK_KV * stride_seq # python saw this with comma as (,) because of += and interpreted asa a tuple look at 
        # AssertionError("cannot convert (<triton.language.core.tensor object at 0x7db3a032c6d0>,) remember that code hints are very informative so read them throughly
        curr_kv += BLOCK_KV

    dQ_block_ptr = (dQ + offs_q[:, None] * stride_seq + 
                      tl.arange(0, HEAD_DIM)[None, :] * stride_dim)
    
    tl.store(dQ_block_ptr, dQ_block)
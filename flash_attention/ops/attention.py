import torch
import os
import triton
import triton.language as tl
from ..kernels.forward import _attn_fwd
from ..kernels.backward import _attn_bwd_preprocess, _attn_bwd_dk_dv, _attn_bwd_dq


class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        assert HEAD_DIM == K.shape[-1] and K.shape[-1] == V.shape[-1]

        O = torch.empty_like(Q)
        stage = 3 if causal else 1
        # this grid basically tells us how many prgrams can we launch that can work in parallel        
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), #numbe of blocks in Q – which block are we going to work with
            BATCH_SIZE * NUM_HEADS,
            1,  # Z dim in CUDA, we dont want to use it for now
        )
        
        
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
            stride_Q_batch=Q.stride(0), # Because we only need a pointer in triton so we have to figure out, thats why were passing the stride
            stride_Q_heads=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),

            stride_K_batch=K.stride(0),
            stride_K_heads=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),

            stride_V_batch=V.stride(0),
            stride_V_heads=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),

            stride_O_batch=O.stride(0),
            stride_O_heads=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),

            # Same dimeniosns actually

            BATCH_SIZE=BATCH_SIZE,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            STAGE=stage
        )

        ctx.save_for_backward(Q, K, V, O, M) # we dont want to save products of thosee (ex: QK^T), because tl.store in the hbm would be veeery eexpensivee, more optimal to compute on the fly
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM
        ctx.causal = causal
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        # makee dO assert contigous, asserte strides
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        # assert Q.stride(0) == Q.shape[-1] * Q.shape[-2] * Q.shape[-3]

        # init dQ, DK, dV
        dQ = torch.empty_like(Q)
        dV = torch.empty_like(V)
        dK = torch.empty_like(K)

        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3
        # BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128 # M/4d in the paper, where M is the total capacity for scalars in SRAM


        preprocess_grid = lambda arg: (
            SEQ_LEN // BLOCK_SIZE_MACRO,
            BATCH_SIZE * NUM_HEADS,
            1
        )
        D = torch.empty_like(M)
        
        _attn_bwd_preprocess[preprocess_grid](O=O, dO=dO, D=D, SEQ_LEN=D.shape[-1], BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=ctx.HEAD_DIM)
        
        dk_dv_grid = (
            SEQ_LEN // BLOCK_SIZE_MACRO,
            1,
            NUM_HEADS*BATCH_SIZE,
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
            # BLOCK_Q=BLOCK_SIZE_MICRO,
            # BLOCK_KV=BLOCK_SIZE_MACRO,
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
            # BLOCK_Q=BLOCK_SIZE_MACRO,
            # BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,)

        # we are returning none none because torch autograd expects the same shape as input (save_for_backward)
        return dQ, dK, dV, None, None 
    
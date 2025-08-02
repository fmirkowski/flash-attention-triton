import triton

AUTOTUNE_CONFIGS = [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_KV in [16]
        for BLOCK_SIZE_Q in [32]
        for num_stages in [3]
        for num_warps in [2]
    ]
    

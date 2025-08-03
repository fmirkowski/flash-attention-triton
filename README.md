

# Easy and hackable implementation of FlashAttention in Triton kernel

---

##  Performance Results!

### Execution Time Comparison
<p align="center">
  <img src="data/flash_attention_execution_times.png" width="600"/>
</p>

### Speedup  
<p align="center">
  <img src="data/flash_attention_speedup.png" width="600"/>
</p>

---

## 📊 Benchmark Summary

| Operation | PyTorch (ms) | Triton (ms) | **Speedup** |
|-----------|--------------|-------------|-------------|
| Forward   | 4.586        | 0.105       | **43.7x**   |
| Backward  | 1.177        | 0.229       | **5.1x**    |
| **Total** | **5.763**    | **0.334**   | **17.3x**   |

> **Hardware:** 1x NVIDIA H100 SXM  
> **Configuration:** Batch=4, Heads=8, SeqLen=512, HeadDim=64  
> **Autotuning:** Enabled for forward pass

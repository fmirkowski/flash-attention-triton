import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for paper-ready figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Updated benchmark data
naive_pytorch_forward = 12.726
opt_pytorch_forward = 1.865
triton_forward = 3.449

naive_pytorch_backward = 18.322
opt_pytorch_backward = 5.471
triton_backward = 15.881

# Calculate speedups vs naive PyTorch
forward_speedup_vs_naive = naive_pytorch_forward / triton_forward
backward_speedup_vs_naive = naive_pytorch_backward / triton_backward
total_naive = naive_pytorch_forward + naive_pytorch_backward
total_triton = triton_forward + triton_backward
total_speedup_vs_naive = total_naive / total_triton

# Calculate speedups vs optimized PyTorch (note: these will be < 1 since Triton is slower)
forward_speedup_vs_opt = opt_pytorch_forward / triton_forward
backward_speedup_vs_opt = opt_pytorch_backward / triton_backward
total_opt = opt_pytorch_forward + opt_pytorch_backward
total_speedup_vs_opt = total_opt / total_triton

print(f"🚀 LATEST BENCHMARK RESULTS:")
print(f"Hardware: CUDA GPU")
print(f"Forward Speedup vs Naive: {forward_speedup_vs_naive:.1f}x")
print(f"Backward Speedup vs Naive: {backward_speedup_vs_naive:.1f}x") 
print(f"Total Speedup vs Naive: {total_speedup_vs_naive:.1f}x")
print(f"Forward vs Optimized PyTorch: {forward_speedup_vs_opt:.1f}x (slower)")
print(f"Backward vs Optimized PyTorch: {backward_speedup_vs_opt:.1f}x (slower)")
print(f"Total vs Optimized PyTorch: {total_speedup_vs_opt:.1f}x (slower)")

# Chart 1: Execution Time Comparison
plt.figure(figsize=(14, 8))

operations = ['Forward Pass', 'Backward Pass', 'Total']
naive_times = [naive_pytorch_forward, naive_pytorch_backward, total_naive]
opt_times = [opt_pytorch_forward, opt_pytorch_backward, total_opt]
triton_times = [triton_forward, triton_backward, total_triton]

x = np.arange(len(operations))
width = 0.25

bars1 = plt.bar(x - width, naive_times, width, label='Naive PyTorch', 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x, opt_times, width, label='Optimized PyTorch (SDPA)', 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = plt.bar(x + width, triton_times, width, label='Triton Flash Attention', 
                color='#FFE66D', alpha=0.8, edgecolor='black', linewidth=0.5)

plt.xlabel('Operation', fontsize=13, fontweight='bold')
plt.ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
plt.title('Flash Attention Performance Comparison\nCUDA GPU Benchmark', 
          fontsize=15, fontweight='bold', pad=20)
# Add smaller subtitle with hyperparameters
plt.text(0.5, 0.95, 'Batch=4, Heads=8, SeqLen=512, HeadDim=64 | 500 iterations per test',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
plt.xticks(x, operations, fontsize=12)
plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig('flash_attention_execution_times.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_execution_times.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Execution time chart saved as:")
print("  - flash_attention_execution_times.png")
print("  - flash_attention_execution_times.pdf")
plt.show()

# Chart 2: Speedup Chart vs Naive PyTorch
plt.figure(figsize=(12, 8))

speedup_ops = ['Forward Pass', 'Backward Pass', 'Total']
speedups_vs_naive = [forward_speedup_vs_naive, backward_speedup_vs_naive, total_speedup_vs_naive]
colors = ['#66BB6A', '#42A5F5', '#AB47BC']

bars = plt.bar(speedup_ops, speedups_vs_naive, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=0.5)

plt.xlabel('Operation', fontsize=13, fontweight='bold')
plt.ylabel('Speedup Factor', fontsize=13, fontweight='bold')
plt.title('Triton Flash Attention Speedup vs Naive PyTorch\nCUDA GPU Benchmark', 
          fontsize=15, fontweight='bold', pad=20)
# Add smaller subtitle with hyperparameters
plt.text(0.5, 0.95, 'Batch=4, Heads=8, SeqLen=512, HeadDim=64 | 500 iterations per test',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (1x)')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.xticks(fontsize=12)

# Add speedup labels on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups_vs_naive)):
    height = bar.get_height()
    label = f'{speedup:.1f}x\nfaster'
    color = 'darkgreen' if speedup > 2 else 'darkblue'
    fontweight = 'bold'
    
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             label, ha='center', va='bottom', fontweight=fontweight, 
             color=color, fontsize=10)

plt.tight_layout()
plt.savefig('flash_attention_speedup_vs_naive.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_speedup_vs_naive.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Speedup vs Naive chart saved as:")
print("  - flash_attention_speedup_vs_naive.png")
print("  - flash_attention_speedup_vs_naive.pdf")
plt.show()

# Chart 3: Performance vs Optimized PyTorch (showing slowdown)
plt.figure(figsize=(12, 8))

performance_ratios = [forward_speedup_vs_opt, backward_speedup_vs_opt, total_speedup_vs_opt]
colors = ['#FF7043', '#EC407A', '#AB47BC']  # More orange/red colors for slowdown

bars = plt.bar(speedup_ops, performance_ratios, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=0.5)

plt.xlabel('Operation', fontsize=13, fontweight='bold')
plt.ylabel('Performance Ratio (Triton / Optimized PyTorch)', fontsize=13, fontweight='bold')
plt.title('Triton Flash Attention vs Optimized PyTorch SDPA\nCUDA GPU Benchmark', 
          fontsize=15, fontweight='bold', pad=20)
# Add smaller subtitle with hyperparameters
plt.text(0.5, 0.95, 'Batch=4, Heads=8, SeqLen=512, HeadDim=64 | Values < 1.0 indicate Triton is slower',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Parity (1.0x)')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.xticks(fontsize=12)
plt.ylim(0, 1.1)  # Since all values are < 1

# Add performance ratio labels on bars
for i, (bar, ratio) in enumerate(zip(bars, performance_ratios)):
    height = bar.get_height()
    label = f'{ratio:.1f}x\n({(1/ratio):.1f}x slower)'
    color = 'darkred'
    fontweight = 'bold'
    
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             label, ha='center', va='bottom', fontweight=fontweight, 
             color=color, fontsize=10)

plt.tight_layout()
plt.savefig('flash_attention_vs_optimized_pytorch.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_vs_optimized_pytorch.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Performance vs Optimized PyTorch chart saved as:")
print("  - flash_attention_vs_optimized_pytorch.png")
print("  - flash_attention_vs_optimized_pytorch.pdf")
plt.show()

# Create comparison table
print("\n" + "="*100)
print("BENCHMARK RESULTS TABLE")
print("="*100)
print(f"Hardware: CUDA GPU")
print(f"Configuration: Batch=4, Heads=8, SeqLen=512, HeadDim=64")
print(f"Iterations: 500 per test")
print("-"*100)
print(f"{'Operation':<15} {'Naive (ms)':<12} {'Optimized (ms)':<15} {'Triton (ms)':<12} {'vs Naive':<12} {'vs Optimized':<15}")
print("-"*100)
print(f"{'Forward':<15} {naive_pytorch_forward:<12.3f} {opt_pytorch_forward:<15.3f} {triton_forward:<12.3f} {forward_speedup_vs_naive:<12.1f}x {forward_speedup_vs_opt:<15.1f}x")
print(f"{'Backward':<15} {naive_pytorch_backward:<12.3f} {opt_pytorch_backward:<15.3f} {triton_backward:<12.3f} {backward_speedup_vs_naive:<12.1f}x {backward_speedup_vs_opt:<15.1f}x")
print(f"{'Total':<15} {total_naive:<12.3f} {total_opt:<15.3f} {total_triton:<12.3f} {total_speedup_vs_naive:<12.1f}x {total_speedup_vs_opt:<15.1f}x")
print("-"*100)
print(f"🏆 KEY FINDINGS:")
print(f"   • Forward Pass: {forward_speedup_vs_naive:.1f}x faster than naive, but {1/forward_speedup_vs_opt:.1f}x slower than optimized PyTorch")
print(f"   • Backward Pass: {backward_speedup_vs_naive:.1f}x faster than naive, but {1/backward_speedup_vs_opt:.1f}x slower than optimized PyTorch")
print(f"   • Total Pipeline: {total_speedup_vs_naive:.1f}x faster than naive, but {1/total_speedup_vs_opt:.1f}x slower than optimized PyTorch")
print(f"   • Note: Optimized PyTorch uses highly optimized SDPA which may outperform Triton on some configurations")
print("="*100) 
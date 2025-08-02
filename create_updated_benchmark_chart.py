import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for paper-ready figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# CORRECTED benchmark data (with proper Triton warmup)
pytorch_forward = 5.899
triton_forward = 0.198
pytorch_backward = 2.271
triton_backward = 1.352

# Calculate speedups
forward_speedup = pytorch_forward / triton_forward
backward_speedup = pytorch_backward / triton_backward
total_pytorch = pytorch_forward + pytorch_backward
total_triton = triton_forward + triton_backward
total_speedup = total_pytorch / total_triton

print(f"🚀 CORRECTED RESULTS (with proper autotuning warmup):")
print(f"Forward Speedup: {forward_speedup:.2f}x")
print(f"Backward Speedup: {backward_speedup:.2f}x") 
print(f"Total Speedup: {total_speedup:.2f}x")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Execution Time Comparison
operations = ['Forward Pass', 'Backward Pass', 'Total']
pytorch_times = [pytorch_forward, pytorch_backward, total_pytorch]
triton_times = [triton_forward, triton_backward, total_triton]

x = np.arange(len(operations))
width = 0.35

bars1 = ax1.bar(x - width/2, pytorch_times, width, label='PyTorch (cuBLAS)', 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, triton_times, width, label='Flash Attention (Triton)', 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Operation', fontsize=13, fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Flash Attention Performance Comparison\n(Proper Autotuning Warmup Applied)', 
              fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(operations, fontsize=12)
ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)

add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)

# Subplot 2: Speedup Chart (with logarithmic scale for dramatic differences)
speedup_ops = ['Forward Pass', 'Backward Pass', 'Total']
speedups = [forward_speedup, backward_speedup, total_speedup]
colors = ['#66BB6A', '#42A5F5', '#AB47BC']

bars3 = ax2.bar(speedup_ops, speedups, color=colors, alpha=0.8, 
                edgecolor='black', linewidth=0.5)

ax2.set_xlabel('Operation', fontsize=13, fontweight='bold')
ax2.set_ylabel('Speedup Factor (log scale)', fontsize=13, fontweight='bold')
ax2.set_title('Flash Attention Speedup over PyTorch\n(Logarithmic Scale)', fontsize=15, fontweight='bold', pad=20)
ax2.set_yscale('log')
ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (1x)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax2.set_xticklabels(speedup_ops, fontsize=12)

# Add speedup labels on bars
for i, (bar, speedup) in enumerate(zip(bars3, speedups)):
    height = bar.get_height()
    if i == 0:  # Forward pass - highlight the massive speedup
        label = f'{speedup:.1f}x\nfaster!'
        color = 'darkgreen'
        fontsize = 11
        fontweight = 'bold'
    else:
        label = f'{speedup:.1f}x\nfaster'
        color = 'darkblue'
        fontsize = 10
        fontweight = 'bold'
    
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
             label, ha='center', va='bottom', fontweight=fontweight, 
             color=color, fontsize=fontsize)

# Highlight the incredible forward pass speedup
ax2.annotate('Nearly 30x\nSpeedup!', 
             xy=(0, forward_speedup), xytext=(0.3, forward_speedup * 2),
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3),
             fontsize=13, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('flash_attention_corrected_benchmark.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_corrected_benchmark.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("\n✅ CORRECTED Charts saved as:")
print("  - flash_attention_corrected_benchmark.png (high-res PNG)")
print("  - flash_attention_corrected_benchmark.pdf (vector PDF for papers)")

plt.show()

# Create comparison table
print("\n" + "="*80)
print("CORRECTED BENCHMARK RESULTS TABLE (Paper Ready)")
print("="*80)
print(f"Configuration: Batch=2, Heads=16, SeqLen=1024, HeadDim=32")
print(f"⚠️  CRITICAL: Results include proper Triton autotuning warmup")
print("-"*80)
print(f"{'Operation':<15} {'PyTorch (ms)':<12} {'Triton (ms)':<12} {'Speedup':<15}")
print("-"*80)
print(f"{'Forward':<15} {pytorch_forward:<12.3f} {triton_forward:<12.3f} {forward_speedup:<15.1f}x")
print(f"{'Backward':<15} {pytorch_backward:<12.3f} {triton_backward:<12.3f} {backward_speedup:<15.1f}x")
print(f"{'Total':<15} {total_pytorch:<12.3f} {total_triton:<12.3f} {total_speedup:<15.1f}x")
print("-"*80)
print(f"🏆 KEY RESULTS:")
print(f"   • Forward Pass: {forward_speedup:.1f}x speedup (nearly 30x!)")
print(f"   • Total Pipeline: {total_speedup:.1f}x speedup")
print(f"   • Proper autotuning warmup was CRITICAL for fair comparison")
print("="*80)

# Show the impact of autotuning
print(f"\n🔍 IMPACT OF AUTOTUNING WARMUP:")
print(f"   • Without warmup: Forward was 0.29x (slower)")
print(f"   • With warmup: Forward is {forward_speedup:.1f}x (faster)")
print(f"   • Improvement: {forward_speedup/0.29:.0f}x better measurement!") 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for paper-ready figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Updated benchmark data
pytorch_forward = 4.586
triton_forward = 0.105
pytorch_backward = 1.177
triton_backward = 0.229

# Calculate speedups
forward_speedup = pytorch_forward / triton_forward
backward_speedup = pytorch_backward / triton_backward
total_pytorch = pytorch_forward + pytorch_backward
total_triton = triton_forward + triton_backward
total_speedup = total_pytorch / total_triton

print(f"🚀 UPDATED BENCHMARK RESULTS:")
print(f"Hardware: 1x NVIDIA H100 SXM")
print(f"Settings: Default configuration")
print(f"Forward Speedup: {forward_speedup:.2f}x")
print(f"Backward Speedup: {backward_speedup:.2f}x") 
print(f"Total Speedup: {total_speedup:.2f}x")

# Chart 1: Execution Time Comparison
plt.figure(figsize=(12, 8))

operations = ['Forward Pass', 'Backward Pass', 'Total']
pytorch_times = [pytorch_forward, pytorch_backward, total_pytorch]
triton_times = [triton_forward, triton_backward, total_triton]

x = np.arange(len(operations))
width = 0.35

bars1 = plt.bar(x - width/2, pytorch_times, width, label='PyTorch (cuBLAS)', 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x + width/2, triton_times, width, label='Flash Attention (Triton)', 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)

plt.xlabel('Operation', fontsize=13, fontweight='bold')
plt.ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
plt.title('Flash Attention Performance Comparison\n1x NVIDIA H100 SXM', 
          fontsize=15, fontweight='bold', pad=20)
# Add smaller subtitle with hyperparameters
plt.text(0.5, 0.95, 'Batch=4, Heads=8, SeqLen=512, HeadDim=64 | Autotuning enabled for forward pass',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
plt.xticks(x, operations, fontsize=12)
plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.3f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)

add_value_labels(bars1)
add_value_labels(bars2)

plt.tight_layout()
plt.savefig('flash_attention_execution_times.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_execution_times.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Execution time chart saved as:")
print("  - flash_attention_execution_times.png")
print("  - flash_attention_execution_times.pdf")
plt.show()

# Chart 2: Speedup Chart
plt.figure(figsize=(12, 8))

speedup_ops = ['Forward Pass', 'Backward Pass', 'Total']
speedups = [forward_speedup, backward_speedup, total_speedup]
colors = ['#66BB6A', '#42A5F5', '#AB47BC']

bars = plt.bar(speedup_ops, speedups, color=colors, alpha=0.8, 
               edgecolor='black', linewidth=0.5)

plt.xlabel('Operation', fontsize=13, fontweight='bold')
plt.ylabel('Speedup Factor (log scale)', fontsize=13, fontweight='bold')
plt.title('Flash Attention Speedup over PyTorch\n1x NVIDIA H100 SXM', 
          fontsize=15, fontweight='bold', pad=20)
# Add smaller subtitle with hyperparameters
plt.text(0.5, 0.95, 'Batch=4, Heads=8, SeqLen=512, HeadDim=64 | Autotuning enabled for forward pass',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
plt.yscale('log')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline (1x)')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.xticks(fontsize=12)

# Add speedup labels on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
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
    
    plt.text(bar.get_x() + bar.get_width()/2., height * 1.1,
             label, ha='center', va='bottom', fontweight=fontweight, 
             color=color, fontsize=fontsize)

plt.tight_layout()
plt.savefig('flash_attention_speedup.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('flash_attention_speedup.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ Speedup chart saved as:")
print("  - flash_attention_speedup.png")
print("  - flash_attention_speedup.pdf")
plt.show()

# Create comparison table
print("\n" + "="*90)
print("BENCHMARK RESULTS TABLE")
print("="*90)
print(f"Hardware: 1x NVIDIA H100 SXM")
print(f"Configuration: Batch=4, Heads=8, SeqLen=512, HeadDim=64")
print(f"Autotuning: Enabled for forward pass")
print("-"*90)
print(f"{'Operation':<15} {'PyTorch (ms)':<12} {'Triton (ms)':<12} {'Speedup':<15}")
print("-"*90)
print(f"{'Forward':<15} {pytorch_forward:<12.3f} {triton_forward:<12.3f} {forward_speedup:<15.1f}x")
print(f"{'Backward':<15} {pytorch_backward:<12.3f} {triton_backward:<12.3f} {backward_speedup:<15.1f}x")
print(f"{'Total':<15} {total_pytorch:<12.3f} {total_triton:<12.3f} {total_speedup:<15.1f}x")
print("-"*90)
print(f"🏆 KEY RESULTS:")
print(f"   • Forward Pass: {forward_speedup:.1f}x speedup (with autotuning)")
print(f"   • Backward Pass: {backward_speedup:.1f}x speedup")
print(f"   • Total Pipeline: {total_speedup:.1f}x speedup")
print(f"   • Hardware: 1x NVIDIA H100 SXM")
print(f"   • Hyperparameters: Batch=4, Heads=8, SeqLen=512, HeadDim=64")
print("="*90) 
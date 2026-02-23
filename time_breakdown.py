#!/usr/bin/python3
"""
YTW Calculation Time Breakdown Analysis
Stacked bar charts showing composition of initialization, Newton solver, and other time
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Configure matplotlib for professional output
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*80)
print("YTW CALCULATION TIME BREAKDOWN ANALYSIS")
print("="*80)

# Read data
df = pd.read_csv('./temp/regression_data_with_residuals.csv')

print(f"\nLoaded {len(df)} bond records")

# ============================================================================
# Part 1: Overall Time Composition
# ============================================================================
print("\n" + "="*80)
print("[PART 1: OVERALL TIME COMPOSITION]")
print("="*80)

init_mean = df['init_ms'].mean()
generate_cf_mean = df['generate_cashflows_ms'].mean()
newton_mean = df['newton_solver_ms'].mean()
other_mean = df['calc_time_ms'].mean() - init_mean - generate_cf_mean - newton_mean

total_mean = init_mean + generate_cf_mean + newton_mean + other_mean

init_pct = (init_mean / total_mean) * 100
generate_cf_pct = (generate_cf_mean / total_mean) * 100
newton_pct = (newton_mean / total_mean) * 100
other_pct = (other_mean / total_mean) * 100

print(f"\nMean Calculation Time: {total_mean:.4f} ms")
print(f"  1. Initialization:      {init_mean:.4f} ms ({init_pct:.1f}%)")
print(f"  2. Generate Cashflows:  {generate_cf_mean:.4f} ms ({generate_cf_pct:.1f}%)")
print(f"  3. Newton Solver:       {newton_mean:.4f} ms ({newton_pct:.1f}%)")
print(f"  4. Other:              {other_mean:.4f} ms ({other_pct:.1f}%)")

# ============================================================================
# Figure 1: Overall Composition - Simple Bar Chart
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 1: OVERALL TIME COMPOSITION]")
print("="*80)

fig1, ax1 = plt.subplots(figsize=(12, 7))

categories = ['Initialization', 'Generate Cashflows', 'Newton Solver', 'Other']
times = [init_mean, generate_cf_mean, newton_mean, other_mean]
colors = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#95E1D3']

bars = ax1.bar(categories, times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels on bars
for i, (bar, time, pct) in enumerate(zip(bars, times, [init_pct, generate_cf_pct, newton_pct, other_pct])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.4f} ms\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Mean Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('YTW Calculation Time Composition\n(All 9,955 Bonds)', 
             fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(times) * 1.15])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/time_breakdown_composition.png', dpi=300, bbox_inches='tight')
print("✓ Composition chart saved to: ./temp/time_breakdown_composition.png")
plt.show()

# ============================================================================
# Figure 2: Stacked Bar Chart by Coupon Count Bins
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 2: TIME BREAKDOWN BY COUPON COUNT RANGES]")
print("="*80)

# Create coupon count bins
df['coupon_bin'] = pd.cut(df['estimated_coupons'], 
                          bins=[0, 20, 40, 60, 80, 100, 200, 400],
                          labels=['1-20', '21-40', '41-60', '61-80', '81-100', '101-200', '201+'])

# Calculate mean times by bin
bin_stats = df.groupby('coupon_bin', observed=True).agg({
    'init_ms': 'mean',
    'generate_cashflows_ms': 'mean',
    'newton_solver_ms': 'mean',
    'calc_time_ms': 'mean'
}).reset_index()

# Calculate "other" time for each bin
bin_stats['other_ms'] = bin_stats['calc_time_ms'] - bin_stats['init_ms'] - bin_stats['generate_cashflows_ms'] - bin_stats['newton_solver_ms']

# Remove any negative other_ms (data quality check)
bin_stats['other_ms'] = bin_stats['other_ms'].clip(lower=0)

print("\nTime composition by coupon count range:")
print(bin_stats.to_string(index=False))

# Create stacked bar chart
fig2, ax2 = plt.subplots(figsize=(14, 8))

x = np.arange(len(bin_stats))
width = 0.6

# Stacked bars
p1 = ax2.bar(x, bin_stats['init_ms'], width, label='Initialization', 
            color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.85)
p2 = ax2.bar(x, bin_stats['generate_cashflows_ms'], width, 
            bottom=bin_stats['init_ms'],
            label='Generate Cashflows', color='#45B7D1', edgecolor='black', linewidth=1.5, alpha=0.85)
p3 = ax2.bar(x, bin_stats['newton_solver_ms'], width,
            bottom=bin_stats['init_ms'] + bin_stats['generate_cashflows_ms'],
            label='Newton Solver', color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.85)
p4 = ax2.bar(x, bin_stats['other_ms'], width,
            bottom=bin_stats['init_ms'] + bin_stats['generate_cashflows_ms'] + bin_stats['newton_solver_ms'],
            label='Other', color='#95E1D3', edgecolor='black', linewidth=1.5, alpha=0.85)

# Add total time labels on top of each bar
for i, (idx, row) in enumerate(bin_stats.iterrows()):
    total = row['calc_time_ms']
    ax2.text(i, total, f'{total:.4f} ms', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# Add percentages within each segment
for i, (idx, row) in enumerate(bin_stats.iterrows()):
    total = row['calc_time_ms']
    
    # Init percentage
    init_pct_bin = (row['init_ms'] / total) * 100 if total > 0 else 0
    ax2.text(i, row['init_ms']/2, f'{init_pct_bin:.0f}%', 
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Generate Cashflows percentage
    gen_cf_pct_bin = (row['generate_cashflows_ms'] / total) * 100 if total > 0 else 0
    ax2.text(i, row['init_ms'] + row['generate_cashflows_ms']/2, f'{gen_cf_pct_bin:.1f}%',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Newton percentage
    newton_pct_bin = (row['newton_solver_ms'] / total) * 100 if total > 0 else 0
    ax2.text(i, row['init_ms'] + row['generate_cashflows_ms'] + row['newton_solver_ms']/2, f'{newton_pct_bin:.1f}%',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Other percentage
    other_pct_bin = (row['other_ms'] / total) * 100 if total > 0 else 0
    ax2.text(i, row['init_ms'] + row['generate_cashflows_ms'] + row['newton_solver_ms'] + row['other_ms']/2, 
            f'{other_pct_bin:.0f}%',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax2.set_xlabel('Coupon Count Range', fontsize=13, fontweight='bold')
ax2.set_ylabel('Calculation Time (ms)', fontsize=13, fontweight='bold')
ax2.set_title('YTW Calculation Time Breakdown by Coupon Count Range\n(Stacked Composition)', 
             fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(bin_stats['coupon_bin'])
ax2.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/time_breakdown_stacked.png', dpi=300, bbox_inches='tight')
print("\n✓ Stacked bar chart saved to: ./temp/time_breakdown_stacked.png")
plt.show()

# ============================================================================
# Figure 3: Proportional Stacked Bar (100%)
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 3: PROPORTIONAL TIME COMPOSITION (100% STACKED)]")
print("="*80)

fig3, ax3 = plt.subplots(figsize=(14, 8))

# Calculate percentages
bin_stats['init_pct'] = (bin_stats['init_ms'] / bin_stats['calc_time_ms']) * 100
bin_stats['gen_cf_pct'] = (bin_stats['generate_cashflows_ms'] / bin_stats['calc_time_ms']) * 100
bin_stats['newton_pct'] = (bin_stats['newton_solver_ms'] / bin_stats['calc_time_ms']) * 100
bin_stats['other_pct'] = (bin_stats['other_ms'] / bin_stats['calc_time_ms']) * 100

x = np.arange(len(bin_stats))
width = 0.6

p1 = ax3.bar(x, bin_stats['init_pct'], width, label='Initialization',
            color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.85)
p2 = ax3.bar(x, bin_stats['gen_cf_pct'], width, bottom=bin_stats['init_pct'],
            label='Generate Cashflows', color='#45B7D1', edgecolor='black', linewidth=1.5, alpha=0.85)
p3 = ax3.bar(x, bin_stats['newton_pct'], width,
            bottom=bin_stats['init_pct'] + bin_stats['gen_cf_pct'],
            label='Newton Solver', color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.85)
p4 = ax3.bar(x, bin_stats['other_pct'], width,
            bottom=bin_stats['init_pct'] + bin_stats['gen_cf_pct'] + bin_stats['newton_pct'],
            label='Other', color='#95E1D3', edgecolor='black', linewidth=1.5, alpha=0.85)

# Add percentage labels
for i, (idx, row) in enumerate(bin_stats.iterrows()):
    # Init
    ax3.text(i, row['init_pct']/2, f'{row["init_pct"]:.0f}%',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Generate Cashflows
    if row['gen_cf_pct'] > 2:  # Only label if visible
        ax3.text(i, row['init_pct'] + row['gen_cf_pct']/2, f'{row["gen_cf_pct"]:.1f}%',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Newton
    if row['newton_pct'] > 2:  # Only label if visible
        ax3.text(i, row['init_pct'] + row['gen_cf_pct'] + row['newton_pct']/2, f'{row["newton_pct"]:.1f}%',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Other
    ax3.text(i, row['init_pct'] + row['gen_cf_pct'] + row['newton_pct'] + row['other_pct']/2,
            f'{row["other_pct"]:.0f}%',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax3.set_xlabel('Coupon Count Range', fontsize=13, fontweight='bold')
ax3.set_ylabel('Proportion (%)', fontsize=13, fontweight='bold')
ax3.set_title('YTW Calculation Time Composition by Coupon Count Range\n(100% Stacked - Relative Proportions)', 
             fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(bin_stats['coupon_bin'])
ax3.set_ylim([0, 100])
ax3.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/time_breakdown_100percent.png', dpi=300, bbox_inches='tight')
print("✓ 100% stacked bar chart saved to: ./temp/time_breakdown_100percent.png")
plt.show()

# ============================================================================
# Figure 4: Line Chart Showing Trend Across Coupon Ranges
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 4: TIME COMPONENT TRENDS BY COUPON RANGE]")
print("="*80)

fig4, ax4 = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(bin_stats))

# Plot lines for each component
ax4.plot(x_pos, bin_stats['init_ms'], 'o-', linewidth=2.5, markersize=8,
        color='#FF6B6B', label='Initialization', markeredgecolor='black', markeredgewidth=1)
ax4.plot(x_pos, bin_stats['generate_cashflows_ms'], 's-', linewidth=2.5, markersize=8,
        color='#45B7D1', label='Generate Cashflows', markeredgecolor='black', markeredgewidth=1)
ax4.plot(x_pos, bin_stats['newton_solver_ms'], '^-', linewidth=2.5, markersize=8,
        color='#4ECDC4', label='Newton Solver', markeredgecolor='black', markeredgewidth=1)
ax4.plot(x_pos, bin_stats['other_ms'], 'D-', linewidth=2.5, markersize=8,
        color='#95E1D3', label='Other', markeredgecolor='black', markeredgewidth=1)

# Add value labels
for i, (idx, row) in enumerate(bin_stats.iterrows()):
    ax4.text(i, row['init_ms'], f'{row["init_ms"]:.4f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i, row['generate_cashflows_ms'], f'{row["generate_cashflows_ms"]:.4f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i, row['newton_solver_ms'], f'{row["newton_solver_ms"]:.4f}', ha='center', va='bottom', fontsize=8)
    ax4.text(i, row['other_ms'], f'{row["other_ms"]:.4f}', ha='center', va='bottom', fontsize=8)

ax4.set_xlabel('Coupon Count Range', fontsize=13, fontweight='bold')
ax4.set_ylabel('Mean Time (ms)', fontsize=13, fontweight='bold')
ax4.set_title('YTW Calculation Time Components Trend\n(Absolute Values by Coupon Range)', 
             fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(bin_stats['coupon_bin'])
ax4.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax4.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/time_breakdown_trends.png', dpi=300, bbox_inches='tight')
print("✓ Trend line chart saved to: ./temp/time_breakdown_trends.png")
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("[SUMMARY STATISTICS]")
print("="*80)

print(f"\nOverall Time Composition (All Bonds):")
print(f"  Initialization:       {init_mean:.4f} ms ({init_pct:.1f}%)")
print(f"  Generate Cashflows:   {generate_cf_mean:.4f} ms ({generate_cf_pct:.1f}%)")
print(f"  Newton Solver:        {newton_mean:.4f} ms ({newton_pct:.1f}%)")
print(f"  Other:                {other_mean:.4f} ms ({other_pct:.1f}%)")
print(f"  Total:                {total_mean:.4f} ms")

print(f"\nKey Observations:")
print(f"  ✓ Initialization dominates: {init_pct:.1f}% of total time")
print(f"  ✓ Generate Cashflows: {generate_cf_pct:.1f}% of total time")
print(f"  ✓ Newton solver is minimal: {newton_pct:.2f}% of total time")
print(f"  ✓ As coupon count increases:")

# Compare first and last bin
first_bin = bin_stats.iloc[0]
last_bin = bin_stats.iloc[-1]

init_change = ((last_bin['init_ms'] - first_bin['init_ms']) / first_bin['init_ms']) * 100
gen_cf_change = ((last_bin['generate_cashflows_ms'] - first_bin['generate_cashflows_ms']) / first_bin['generate_cashflows_ms']) * 100 if first_bin['generate_cashflows_ms'] > 0 else 0
newton_change = ((last_bin['newton_solver_ms'] - first_bin['newton_solver_ms']) / first_bin['newton_solver_ms']) * 100 if first_bin['newton_solver_ms'] > 0 else 0
other_change = ((last_bin['other_ms'] - first_bin['other_ms']) / first_bin['other_ms']) * 100 if first_bin['other_ms'] > 0 else 0

print(f"    - Init time changes: {init_change:+.1f}%")
print(f"    - Gen CF time changes: {gen_cf_change:+.1f}%")
print(f"    - Newton time changes: {newton_change:+.1f}%")
print(f"    - Other time changes: {other_change:+.1f}%")

print(f"\n✓ Generated 4 visualizations:")
print(f"  1. time_breakdown_composition.png - Simple bar chart (overall)")
print(f"  2. time_breakdown_stacked.png - Stacked bars (absolute values)")
print(f"  3. time_breakdown_100percent.png - 100% stacked (proportions)")
print(f"  4. time_breakdown_trends.png - Line chart (trends)")

print("\n" + "="*80)
print("TIME BREAKDOWN ANALYSIS COMPLETE")
print("="*80)

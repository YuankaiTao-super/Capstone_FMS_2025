#!/usr/bin/python3
"""
Coupon Count Distribution by Ranges
Bar chart showing coupon count distribution across predefined ranges
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
print("COUPON COUNT DISTRIBUTION ANALYSIS")
print("="*80)

# Read data
df = pd.read_csv('./temp/regression_data_with_residuals.csv')

print(f"\nLoaded {len(df)} bond records")

# ============================================================================
# Coupon Distribution by Ranges
# ============================================================================
print("\n" + "="*80)
print("[COUPON COUNT DISTRIBUTION BY RANGES]")
print("="*80)

# Create coupon count ranges
df['coupon_range'] = pd.cut(df['estimated_coupons'], 
                            bins=[0, 10, 20, 40, 60, 80, 100, 200, 400],
                            labels=['1-10', '11-20', '21-40', '41-60', '61-80', '81-100', '101-200', '201+'])

# Count bonds in each range
range_counts = df['coupon_range'].value_counts().sort_index()

print(f"\nCoupon count distribution by ranges:")
for range_label, count in range_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {range_label:>7}: {count:>5} bonds ({pct:>5.1f}%)")

# ============================================================================
# Bar Chart
# ============================================================================
print("\n" + "="*80)
print("[GENERATING BAR CHART]")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 8))

# Create bar chart
x_pos = np.arange(len(range_counts))
bars = ax.bar(x_pos, range_counts.values, color="#2D7FC2", 
              edgecolor='black', linewidth=1.5, alpha=0.85, width=0.7)

# Add value labels on top of each bar
for i, (bar, count) in enumerate(zip(bars, range_counts.values)):
    height = bar.get_height()
    pct = (count / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Labels and formatting
ax.set_xlabel('Coupon Count Range', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Bonds', fontsize=13, fontweight='bold')
ax.set_title('Coupon Count Distribution by Ranges', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(range_counts.index, fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Set y-axis to start from 0
ax.set_ylim([0, max(range_counts.values) * 1.1])

plt.tight_layout()
plt.savefig('./temp/coupon_distribution_histogram.png', dpi=300, bbox_inches='tight')
print("\n✓ Distribution chart saved to: ./temp/coupon_distribution_histogram.png")
plt.show()

print("\n" + "="*80)
print("COUPON DISTRIBUTION ANALYSIS COMPLETE")
print("="*80)

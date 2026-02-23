#!/usr/bin/python3
"""
YTW Calculation Time Regression Visualization
Separate plots for Model 1 and Model 3 with cleaner display
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import seaborn as sns

# Configure matplotlib for professional output
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Read data from regression analysis
df = pd.read_csv('./temp/regression_data_with_residuals.csv')

# Extract data
x = df['estimated_coupons'].values
y1 = df['calc_time_ms'].values
y3 = df['residual_time'].values

# Model parameters
slope1 = 0.005445
intercept1 = 0.687662

slope3 = 0.005268
intercept3 = 0.259971

print("Generating Model 1 and Model 3 visualizations...\n")

# ============================================================================
# Figure 1: Model 1 (Linear scale, coupon <= 100)
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Filter data: coupon <= 100
mask1 = x <= 100
x1_filtered = x[mask1]
y1_filtered = y1[mask1]

# Calculate regression line for filtered data
x_line1 = np.array([x1_filtered.min(), x1_filtered.max()])
y_line1 = intercept1 + slope1 * x_line1

# Plot
ax1.scatter(x1_filtered, y1_filtered, alpha=0.7, s=30, color='steelblue', 
           edgecolors='none', label='Calculation time')

ax1.plot(x_line1, y_line1, 'r-', linewidth=3, 
        label=f'Regression: y = {intercept1:.4f} + {slope1:.6f}x')

ax1.set_xlabel('Coupon Count', fontsize=13, fontweight='bold')
ax1.set_ylabel('Calculation Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Calculation Time to Coupon Count', 
             fontsize=14, fontweight='bold')
ax1.set_ylim([0.0, 3.0])
ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/model1_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Model 1 plot saved to: ./temp/model1_visualization.png")
plt.show()

fig2, ax2 = plt.subplots(figsize=(12, 7))

# Filter data: residual_time >= 0 AND coupon <= 100
mask3 = (y3 >= 0) & (x <= 100)
x3_filtered = x[mask3]
y3_filtered = y3[mask3]

# Calculate regression line for filtered data
x_line3 = np.array([x3_filtered.min(), x3_filtered.max()])
y_line3 = intercept3 + slope3 * x_line3

# Plot
ax2.scatter(x3_filtered, y3_filtered, alpha=0.7, s=30, color='purple', 
           edgecolors='none', label='Residual time')
ax2.plot(x_line3, y_line3, 'r-', linewidth=3, 
        label=f'Regression: y = {intercept3:.4f} + {slope3:.6f}x')

ax2.set_xlabel('Coupon Count', fontsize=13, fontweight='bold')
ax2.set_ylabel('Calculation Time - Initialization Time (ms)', fontsize=13, fontweight='bold')
ax2.set_title('(Calculation Time - Initialization Time) to Coupon Count', 
             fontsize=14, fontweight='bold')
ax2.set_ylim([0.0, 3.0])
ax2.legend(loc='upper left', fontsize=12, framealpha=0.95)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/model3_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Model 3 plot saved to: ./temp/model3_visualization.png")
plt.show()


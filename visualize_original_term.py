#!/usr/bin/python3
"""
Original Bond Term vs YTW Calculation Time
Two visualizations: scatter plot and hexbin heatmap
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read data
df_calc = pd.read_csv('./temp/regression_data_with_residuals.csv')
df_master = pd.read_csv('./muniSecMaster.csv', sep='|', low_memory=False)

# Parse dates and calculate original term
df_master['issueDate_parsed'] = pd.to_datetime(df_master['issueDate'], errors='coerce')
df_master['maturityDate_parsed'] = pd.to_datetime(df_master['maturityDate'], errors='coerce')

valid_dates = (df_master['issueDate_parsed'].notna()) & (df_master['maturityDate_parsed'].notna())
df_master['original_term_years'] = np.where(
    valid_dates,
    (df_master['maturityDate_parsed'] - df_master['issueDate_parsed']).dt.days / 365.25,
    np.nan
)

# Merge data
df_merged = df_calc.merge(
    df_master[['cusip', 'original_term_years']],
    on='cusip',
    how='left'
)

df_analysis = df_merged[df_merged['original_term_years'].notna()].copy()

x = df_analysis['original_term_years'].values
y = df_analysis['calc_time_ms'].values

# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r2 = r_value ** 2

print(f"Data: {len(df_analysis)} bonds")
print(f"R2 = {r2:.4f}")
print(f"Equation: y = {intercept:.4f} + {slope:.6f}x")

# ============================================================================
# Figure 1: Scatter Plot
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 9))
ax1.scatter(x, y, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
ax1.set_xlabel('Original Bond Term (Years, Issue to Maturity)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Calculation Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Original Bond Term vs YTW Calculation Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('./temp/original_term_scatter.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved")

# ============================================================================
# Figure 2: Hexbin Heatmap
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 9))
hexbin = ax2.hexbin(x, y, gridsize=40, cmap='YlOrRd', mincnt=1, 
                     edgecolors='white', linewidths=0.2,
                     norm=plt.matplotlib.colors.LogNorm())

# Plot regression line
x_line = np.array([x.min(), x.max()])
y_line = intercept + slope * x_line
ax2.plot(x_line, y_line, 'b--', linewidth=3, label=f'Linear fit: R2={r2:.4f}')

ax2.set_xlabel('Original Bond Term (Years, Issue to Maturity)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Calculation Time (ms)', fontsize=13, fontweight='bold')
ax2.set_title('Original Bond Term vs YTW Calculation Time (Hexbin)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(hexbin, ax=ax2)
cbar.set_label('Observation Count (log scale)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('./temp/original_term_hexbin.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved")

plt.show()

#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Fine Grid Hexbin Visualization
Single focused visualization with fine grid
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

print("="*90)
print("FINE GRID HEXBIN VISUALIZATION")
print("="*90)

# Load data
df_res = pd.read_csv('./temp/regression_data_with_residuals.csv')
print(f"\nLoaded {len(df_res)} bond records")

# Calculate regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_res['estimated_coupons'].values, 
    df_res['generate_cashflows_ms'].values
)
r2 = r_value**2

x_line = np.array([df_res['estimated_coupons'].min(), df_res['estimated_coupons'].max()])
y_line = intercept + slope * x_line

fig, ax = plt.subplots(figsize=(14, 9))

# Hexbin with fine grid
hexbin = ax.hexbin(df_res['estimated_coupons'], df_res['generate_cashflows_ms'],
                   gridsize=50, cmap='YlOrRd', mincnt=1, edgecolors='none',
                   norm='log')

# Add colorbar
cb = plt.colorbar(hexbin, ax=ax)
cb.set_label('Count (log scale)', fontsize=12, fontweight='bold')

# Add regression line
label_str = 'Linear Fit: y = {:.6f} + {:.6f}*x'.format(intercept, slope)
ax.plot(x_line, y_line, 'b--', linewidth=2, label=label_str)

# Labels and title
ax.set_xlabel('Coupon Count', fontsize=13, fontweight='bold')
ax.set_ylabel('Generate Cashflows Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('Hexbin Visualization \nGenCF Time to Coupon Count',
            fontsize=14, fontweight='bold')
ax.set_xlim([0, 250])
ax.set_ylim([0, 1.15])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper left', framealpha=0.95)

plt.tight_layout()
plt.savefig('./temp/hexbin_fine_grid.png', dpi=300, bbox_inches='tight')
print("[OK] Fine grid hexbin saved: ./temp/hexbin_fine_grid.png")
plt.show()

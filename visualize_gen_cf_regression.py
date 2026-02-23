#!/usr/bin/python3
"""
Visualization of Generate Cashflows Time Regression Analysis
Scatter plots with regression lines showing coupon count vs time relationships
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn as sns

# Configure matplotlib for professional output
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*80)
print("GENERATE CASHFLOWS TIME - REGRESSION VISUALIZATION")
print("="*80)

# Load data
df_res = pd.read_csv('./temp/regression_data_with_residuals.csv')

print(f"\nLoaded {len(df_res)} bond records")

# ============================================================================
# Figure 1: Generate Cashflows vs Coupon Count (Scatter + Regression Line)
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 1: Generate Cashflows Time vs Coupon Count]")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 9))

# Scatter plot
scatter = ax.scatter(df_res['estimated_coupons'], df_res['generate_cashflows_ms'],
                    alpha=0.7, s=30, color='steelblue', edgecolors='none')

# Regression line (Model 5)
X = df_res[['estimated_coupons']].values
y = df_res['generate_cashflows_ms'].values
model = LinearRegression().fit(X, y)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_res['estimated_coupons'].values, y)
r2 = r_value**2

x_line = np.array([df_res['estimated_coupons'].min(), df_res['estimated_coupons'].max()])
y_line = model.predict(x_line.reshape(-1, 1))

ax.plot(x_line, y_line, 'r-', linewidth=3, label=f'Linear Fit: y = {intercept:.4f} + {slope:.6f}*x')

# Confidence interval
from scipy.stats import t
n = len(df_res)
dof = n - 2
t_val = t.ppf(0.975, dof)
residuals = y - model.predict(X)
s_err = np.sqrt(np.sum(residuals**2) / dof)
x_mean = X.mean()
x_std = np.sqrt(np.sum((X - x_mean)**2))

y_pred = model.predict(x_line.reshape(-1, 1))
margin = t_val * s_err * np.sqrt(1/n + (x_line - x_mean)**2 / x_std**2)

ax.fill_between(x_line, y_pred - margin, y_pred + margin, alpha=0.2, color='red', label='95% Confidence Interval')

# Labels and formatting
ax.set_xlabel('Coupon Count', fontsize=13, fontweight='bold')
ax.set_ylabel('Generate Cashflows Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('Generate Cashflows Time to Coupon Count',
            fontsize=14, fontweight='bold')

# # Statistics box
# stats_text = f'R² = {r2:.4f} ({r2*100:.2f}%)\nSlope = {slope:.6f} ms/coupon\nIntercept = {intercept:.6f} ms\nt-stat = {slope/std_err:.2f}***\nn = {n:,}'
# ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
#        fontsize=11, verticalalignment='bottom', horizontalalignment='right',
#        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(fontsize=12, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0, 100])
ax.set_ylim([0, 2.0])

plt.tight_layout()
plt.savefig('./temp/gen_cf_regression_model5.png', dpi=300, bbox_inches='tight')
print("✓ Model 5 scatter plot saved to: ./temp/gen_cf_regression_model5.png")
plt.show()

# ============================================================================
# Figure 3: Binned Analysis - Time Components by Coupon Ranges
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 3: Time Breakdown by Coupon Count Ranges]")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Create coupon bins
df_res['coupon_bin'] = pd.cut(df_res['estimated_coupons'],
                              bins=[0, 10, 20, 40, 60, 100, 400],
                              labels=['1-10', '11-20', '21-40', '41-60', '61-100', '101+'])

bin_stats = df_res.groupby('coupon_bin', observed=True).agg({
    'init_ms': 'mean',
    'generate_cashflows_ms': 'mean',
    'newton_solver_ms': 'mean',
    'calc_time_ms': 'count'  # for sample size
}).reset_index()
bin_stats.columns = ['coupon_bin', 'init_ms', 'gen_cf_ms', 'newton_ms', 'count']

# Left: Generate Cashflows Component
x_pos = np.arange(len(bin_stats))
bars1 = ax1.bar(x_pos, bin_stats['gen_cf_ms'], color='#45B7D1',
               edgecolor='black', linewidth=1.5, alpha=0.85, width=0.6)

# Add value labels and sample sizes
for i, (bar, val, count) in enumerate(zip(bars1, bin_stats['gen_cf_ms'], bin_stats['count'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f} ms\n(n={int(count):,})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Coupon Count Range', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Generate Cashflows Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Generate Cashflows Time by Coupon Range',
             fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(bin_stats['coupon_bin'])
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1.5])

# Right: All Components Comparison
x_pos2 = np.arange(len(bin_stats))
width = 0.25

bars_init = ax2.bar(x_pos2 - width, bin_stats['init_ms'], width, label='Initialization',
                    color='#FF6B6B', edgecolor='black', linewidth=1, alpha=0.85)
bars_gen = ax2.bar(x_pos2, bin_stats['gen_cf_ms'], width, label='Generate CF',
                   color='#45B7D1', edgecolor='black', linewidth=1, alpha=0.85)
bars_newton = ax2.bar(x_pos2 + width, bin_stats['newton_ms'], width, label='Newton',
                      color='#4ECDC4', edgecolor='black', linewidth=1, alpha=0.85)

ax2.set_xlabel('Coupon Count Range', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('All Time Components by Coupon Range',
             fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(bin_stats['coupon_bin'])
ax2.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 0.8])

plt.tight_layout()
plt.savefig('./temp/gen_cf_binned_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Binned analysis saved to: ./temp/gen_cf_binned_analysis.png")
plt.show()

# ============================================================================
# Figure 4: Distribution + KDE Plot
# ============================================================================
print("\n" + "="*80)
print("[FIGURE 4: Generate Cashflows Time Distribution]")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Histogram with KDE
ax1.hist(df_res['generate_cashflows_ms'], bins=50, color='#45B7D1',
        edgecolor='black', linewidth=0.5, alpha=0.7, density=True)
df_res['generate_cashflows_ms'].plot(kind='kde', ax=ax1, color='red', linewidth=2.5,
                                     label='KDE')

ax1.set_xlabel('Generate Cashflows Time (ms)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Generate Cashflows Time',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, linestyle='--')

# Right: Box plot by coupon ranges
data_for_box = [df_res[df_res['coupon_bin'] == bin_label]['generate_cashflows_ms'].values
               for bin_label in bin_stats['coupon_bin']]

bp = ax2.boxplot(data_for_box, labels=bin_stats['coupon_bin'], patch_artist=True,
                 notch=False, widths=0.6)

# Color the boxes
for patch, color in zip(bp['boxes'], ['#45B7D1']*len(bp['boxes'])):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for whisker in bp['whiskers']:
    whisker.set(linewidth=1.5)
for median in bp['medians']:
    median.set(color='red', linewidth=2.5)

ax2.set_xlabel('Coupon Count Range', fontsize=12, fontweight='bold')
ax2.set_ylabel('Generate Cashflows Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Generate Cashflows Time Distribution by Coupon Range',
             fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('./temp/gen_cf_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Distribution plot saved to: ./temp/gen_cf_distribution.png")
plt.show()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated 4 visualizations:")
print("  1. gen_cf_regression_model5.png - Scatter plot with regression line (Model 5)")
print("  2. gen_cf_model_comparison.png - Comparison of all 3 models (R-squared)")
print("  3. gen_cf_binned_analysis.png - Time breakdown by coupon ranges")
print("  4. gen_cf_distribution.png - Distribution and box plots")
print("\n" + "="*80)

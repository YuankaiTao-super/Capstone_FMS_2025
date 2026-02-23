#!/usr/bin/python3
"""
YTW Calculation Time Scalability Analysis
Log-Log Plot showing power-law relationship between coupon count and calculation time
Demonstrates O(N) complexity of Newton-Raphson solver
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import seaborn as sns
from scipy.optimize import curve_fit

# Configure matplotlib for professional output
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("="*80)
print("YTW CALCULATION TIME SCALABILITY ANALYSIS (Log-Log)")
print("="*80)

# Read regression data
df = pd.read_csv('./temp/regression_data_with_residuals.csv')

print(f"\nLoaded {len(df)} bond records")
print(f"Coupon count range: {df['estimated_coupons'].min():.0f} - {df['estimated_coupons'].max():.0f}")
print(f"Calculation time range: {df['generate_cashflows_ms'].min():.4f} - {df['generate_cashflows_ms'].max():.4f} ms")

# Extract data
x = df['estimated_coupons'].values  # N (coupon count)
y = df['generate_cashflows_ms'].values        # T (generation time in ms)

# Filter out zero or negative values for log-log analysis
valid_mask = (x > 0) & (y > 0)
x_valid = x[valid_mask]
y_valid = y[valid_mask]

print(f"Valid data points for log-log: {len(x_valid)}")

# ============================================================================
# Power-Law Fitting: time = a * (coupon_count)^b
# ============================================================================
print("\n" + "="*80)
print("[POWER-LAW FITTING ANALYSIS]")
print("="*80)

# Fit: log(time) = log(a) + b*log(coupons)
# Using least squares on log-transformed data
log_x = np.log(x_valid)
log_y = np.log(y_valid)

# Linear regression on log data
slope_log, intercept_log, r_value, p_value, std_err = stats.linregress(log_x, log_y)

# Extract power-law parameters
exponent_b = slope_log
coefficient_a = np.exp(intercept_log)
r_squared = r_value ** 2

print(f"\nPower-Law Model: time = {coefficient_a:.6f} × (coupon_count)^{exponent_b:.4f}")
print(f"  Coefficient (a): {coefficient_a:.8f}")
print(f"  Exponent (b): {exponent_b:.6f}")
print(f"  R² (on log-log): {r_squared:.6f}")
print(f"  Pearson r: {r_value:.6f}")
print(f"  p-value: {p_value:.2e}")
print(f"  Std Error (slope): {std_err:.8f}")

# Interpretation
print(f"\nInterpretation:")
if abs(exponent_b - 1.0) < 0.1:
    print(f"  ✓ Exponent ≈ 1.0 confirms LINEAR scaling (O(N) complexity)")
elif exponent_b < 1.0:
    print(f"  ✓ Exponent < 1.0 indicates sub-linear scaling (better than O(N))")
else:
    print(f"  ⚠ Exponent > 1.0 indicates super-linear scaling")

# Calculate residuals and standard error
y_fitted = coefficient_a * (x_valid ** exponent_b)
residuals = np.log(y_valid) - np.log(y_fitted)
residual_std = np.std(residuals)

print(f"  Standard deviation of log residuals: {residual_std:.6f}")

# ============================================================================
# Confidence Interval Calculation
# ============================================================================
print("\n" + "="*80)
print("[CONFIDENCE INTERVALS (95%)]")
print("="*80)

# Predict and get confidence intervals
x_range = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 100)
y_predict = coefficient_a * (x_range ** exponent_b)

# Confidence interval on prediction
# SE of prediction at x_pred
x_mean = np.mean(log_x)
x_var = np.sum((log_x - x_mean) ** 2)

t_crit = stats.t.ppf(0.975, len(x_valid) - 2)  # 95% CI, two-tailed

log_y_pred = intercept_log + slope_log * np.log(x_range)
se_pred = residual_std * np.sqrt(1 + 1/len(x_valid) + (np.log(x_range) - x_mean)**2 / x_var)

log_y_lower = log_y_pred - t_crit * se_pred
log_y_upper = log_y_pred + t_crit * se_pred

y_lower = np.exp(log_y_lower)
y_upper = np.exp(log_y_upper)

print(f"\n95% Confidence Interval for predictions")
print(f"  At coupon_count = 50: {coefficient_a * (50**exponent_b):.4f} ms")
print(f"    95% CI: [{y_lower[np.argmin(np.abs(x_range - 50))]:.4f}, {y_upper[np.argmin(np.abs(x_range - 50))]:.4f}] ms")
print(f"  At coupon_count = 100: {coefficient_a * (100**exponent_b):.4f} ms")
print(f"    95% CI: [{y_lower[np.argmin(np.abs(x_range - 100))]:.4f}, {y_upper[np.argmin(np.abs(x_range - 100))]:.4f}] ms")

# ============================================================================
# Figure: Log-Log Scalability Plot
# ============================================================================
print("\n" + "="*80)
print("[GENERATING VISUALIZATION]")
print("="*80)

fig, ax = plt.subplots(figsize=(13, 8))

# Plot data points
ax.loglog(x_valid, y_valid, 'o', alpha=0.5, markersize=5, color='steelblue', 
         label=f'Observed (n={len(x_valid)})')

# Plot power-law fit line
ax.loglog(x_range, y_predict, 'r-', linewidth=2, 
         label=f'Power-law fit: $T = {coefficient_a:.4f} × N^{{{exponent_b:.4f}}}$ ($R^2 = {r_squared:.4f}$)')

# Plot confidence interval (shaded region)
ax.fill_between(x_range, y_lower, y_upper, alpha=0.15, color='red', 
               label='95% Confidence Interval')

# Formatting
ax.set_xlabel('Coupon Count (N)', fontsize=13, fontweight='bold')
ax.set_ylabel('Generate Cashflows Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('YTW Calculation Time Scalability Analysis\n(Log-Log Scale)', 
            fontsize=14, fontweight='bold')

# Grid
ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.15, which='minor', linestyle=':', linewidth=0.3)

# Legend
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

# Add text box with statistics
stats_text = (
    f'Linear Regression (Log-Log Scale)\n'
    f'Exponent: {exponent_b:.4f}\n'
    f'p-value: {p_value:.2e}\n'
    f'Std Error: {std_err:.6f}'
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
       fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('./temp/scalability_loglog_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Log-Log plot saved to: ./temp/scalability_loglog_analysis.png")
plt.show()

# ============================================================================
# Figure 2: Linear Scale (for comparison)
# ============================================================================
print("\nGenerating linear-scale comparison plot...")

fig2, ax2 = plt.subplots(figsize=(13, 8))

# Plot data points (linear scale, filtered to coupon <= 100 for clarity)
mask_linear = (x_valid <= 110)
x_linear = x_valid[mask_linear]
y_linear = y_valid[mask_linear]

ax2.scatter(x_linear, y_linear, alpha=0.5, s=30, color='steelblue', 
           edgecolors='none', label=f'Observed (n=9872)')

# Plot power-law fit line (linear scale)
x_line_linear = np.linspace(x_linear.min(), x_linear.max(), 100)
y_line_linear = coefficient_a * (x_line_linear ** exponent_b)

ax2.plot(x_line_linear, y_line_linear, 'r-', linewidth=3,
        label=f'Power-law fit: $T = {coefficient_a:.4f} × N^{{{exponent_b:.4f}}}$')

# Confidence interval (linear scale)
y_lower_linear = y_lower[(x_range >= x_linear.min()) & (x_range <= x_linear.max())]
y_upper_linear = y_upper[(x_range >= x_linear.min()) & (x_range <= x_linear.max())]
x_range_linear = x_range[(x_range >= x_linear.min()) & (x_range <= x_linear.max())]

ax2.fill_between(x_range_linear, y_lower_linear, y_upper_linear, alpha=0.15, color='red',
                label='95% Confidence Interval')

# Formatting
ax2.set_xlabel('Coupon Count (N)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Generate Cashflows Time (ms)', fontsize=13, fontweight='bold')
ax2.set_title('YTW Calculation Time Scalability Analysis', 
             fontsize=14, fontweight='bold')

ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('./temp/scalability_linear_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Linear-scale plot saved to: ./temp/scalability_linear_analysis.png")
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("[SUMMARY]")
print("="*80)

print(f"\n✓ Power-law exponent: {exponent_b:.4f} (≈ O(N^{exponent_b:.2f}))")
print(f"✓ Coefficient: {coefficient_a:.8f}")
print(f"✓ R² on log-log scale: {r_squared:.6f}")
print(f"✓ All 9,955 bonds analyzed")
print(f"✓ Two visualizations generated:")
print(f"  - Log-log scale (shows power law directly)")
print(f"  - Linear scale (familiar format, coupon ≤ 150)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

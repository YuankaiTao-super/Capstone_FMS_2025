#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-12
completed: 2024-10-12
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./temp/ytw_calc_times.csv')

df['calc_time_us'] = df['calc_time_ms'] * 1000  # convert ms to us
df['newton_solver_us'] = df['newton_solver_ms'] * 1000  # convert ms to us

# Create 2 subplots: both showing calc_time and newton_solver overlaid
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('YTW Calculation Times: Total vs Newton Solver', fontsize=16, fontweight='bold')

# 1. Line chart - both curves overlaid
ax1.plot(range(len(df)), df['calc_time_us'], 'b-', linewidth=0.8, alpha=0.6, label='Total Calc Time')
ax1.plot(range(len(df)), df['newton_solver_us'], 'r-', linewidth=0.8, alpha=0.6, label='Newton Solver Time')
ax1.set_title('Time Series Comparison', fontweight='bold')
ax1.set_xlabel('Bond ID')
ax1.set_ylabel('Time (us)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Histogram - both distributions overlaid
ax2.hist(df['calc_time_us'], bins=50, color='blue', alpha=0.5, edgecolor='black', label='Total Calc Time')
ax2.hist(df['newton_solver_us'], bins=50, color='red', alpha=0.5, edgecolor='black', label='Newton Solver Time')
ax2.set_title('Distribution Comparison', fontweight='bold')
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Frequency')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./temp/ytw_calc_times_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n=== Statistics Summary ===")
print(f"\nTotal Calculation Time:")
print(f"  Mean: {df['calc_time_us'].mean():.2f} us")
print(f"  Median: {df['calc_time_us'].median():.2f} us")
print(f"  Max: {df['calc_time_us'].max():.2f} us")
print(f"  Min: {df['calc_time_us'].min():.2f} us")

print(f"\nNewton Solver Time:")
print(f"  Mean: {df['newton_solver_us'].mean():.2f} us")
print(f"  Median: {df['newton_solver_us'].median():.2f} us")
print(f"  Max: {df['newton_solver_us'].max():.2f} us")
print(f"  Min: {df['newton_solver_us'].min():.2f} us")

# top 10 bonds with the longest calculation times
print(f"\n=== Top 10 Bonds by Total Calculation Time ===")
top_slow = df.nlargest(10, 'calc_time_us')[['cusip', 'calc_time_us', 'newton_solver_us', 'ytw']]
for idx, row in top_slow.iterrows():
    print(f"CUSIP: {row['cusip']}, Total: {row['calc_time_us']:.2f}us, Newton: {row['newton_solver_us']:.2f}us, YTW: {row['ytw']:.4f}%")

# top 10 bonds with the longest newton solver times
print(f"\n=== Top 10 Bonds by Newton Solver Time ===")
top_newton = df.nlargest(10, 'newton_solver_us')[['cusip', 'calc_time_us', 'newton_solver_us', 'ytw']]
for idx, row in top_newton.iterrows():
    print(f"CUSIP: {row['cusip']}, Total: {row['calc_time_us']:.2f}us, Newton: {row['newton_solver_us']:.2f}us, YTW: {row['ytw']:.4f}%")


# YTW vs Calculation Time
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['ytw'], df['calc_time_us'], alpha=0.6, c=df['calc_time_us'], 
                     cmap='viridis', s=20)
plt.title('YTW vs Calculation Time')
plt.xlabel('YTW (%)')
plt.ylabel('Calculation Time (us)')
plt.colorbar(scatter, label='Calculation Time (us)')
plt.grid(True, alpha=0.3)
plt.show()

plt.savefig('./temp/plot_ytw_calc.png', dpi=300, bbox_inches='tight')

print(f"\n=== Top 5 Bonds by YTW ===")
print(df['ytw'].sort_values(ascending=False).head(5))
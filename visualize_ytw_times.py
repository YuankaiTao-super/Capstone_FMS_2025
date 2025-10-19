#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-12
completed: 2024-10-12
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./temp/ytw_calc_times_guess.csv')

df['calc_time_us'] = df['calc_time_ms'] * 1000  # convert ms to us

# create subplots
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle('Distribution of YTW Calculation Times with Initial Guess', fontsize=16, fontweight='bold')

# 1. Line chart - All calculation times in order
ax1.plot(range(len(df)), df['calc_time_us'], 'b-', linewidth=0.8, alpha=0.7)
ax1.set_title('Time Series of Calculation Times')
ax1.set_xlabel('Bond ID')
ax1.set_ylabel('Calculation Time (us)')
ax1.grid(True, alpha=0.3)

# 2. Histogram - Distribution of Calculation Times
ax2.hist(df['calc_time_us'], bins=50, color='blue', alpha=0.7, edgecolor='black')
ax2.set_title('Histogram of Calculation Time Distribution W Initial Guess')
ax2.set_xlabel('Calculation Time (us)')
ax2.set_ylabel('Frequency')
ax2.legend()

plt.tight_layout()
plt.savefig('./temp/ytw_calc_times_w_guess.png', dpi=300, bbox_inches='tight')
plt.show()

# Find the top 10 bonds with the longest calculation times
print(f"\n=== Top 10 Bonds by Calculation Time with Initial Guess ===")
top_slow = df.nlargest(10, 'calc_time_us')[['cusip', 'calc_time_us', 'ytw']]
for idx, row in top_slow.iterrows():
    print(f"CUSIP: {row['cusip']}, Time: {row['calc_time_us']:.2f}us, YTW: {row['ytw']:.2f}%")


# YTW vs Calculation Time
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['ytw'], df['calc_time_us'], alpha=0.6, c=df['calc_time_us'], 
                     cmap='viridis', s=20)
plt.title('YTW vs Calculation Time W Initial Guess')
plt.xlabel('YTW (%)')
plt.ylabel('Calculation Time (us)')
plt.colorbar(scatter, label='Calculation Time (us)')
plt.grid(True, alpha=0.3)
plt.show()

plt.savefig('./temp/plot_ytw_calc_w_guess.png', dpi=300, bbox_inches='tight')

print(f"\n=== Top 5 Bonds by YTW with Initial Guess ===")
print(df['ytw'].sort_values(ascending=False).head(5))
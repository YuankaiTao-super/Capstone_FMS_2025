#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-15
completed: 2024-10-15
issue #3
Clean PSV data - remove securities not in muniSecMaster
"""
import pandas as pd

# Read both files
print("Reading file...")
psv_df = pd.read_csv('ice_cep_prices_20251002.psv', sep='|')
muni_df = pd.read_csv('muniSecMaster.csv', sep='|')

# Get valid CUSIPs from muniSecMaster
valid_cusips = set(muni_df['cusip'])
print(f"Valid CUSIPs: {len(valid_cusips)}")

# Filter PSV data to keep only valid securities
print("Filtering data...")
original_count = len(psv_df)
cleaned_df = psv_df[psv_df['securityId'].isin(valid_cusips)]
filtered_count = len(cleaned_df)

print(f"Filtered results:")
print(f"  Before: {original_count:,} rows")
print(f"  After:  {filtered_count:,} rows")
print(f"  Removed: {original_count - filtered_count:,} rows ({(original_count - filtered_count)/original_count*100:.1f}%)")

# Save cleaned data
output_file = './ice_cep_prices_20251002_cleaned.psv'
cleaned_df.to_csv(output_file, sep='|', index=False)
print(f"Cleaned data saved to: {output_file}")

# Optional: Overwrite original file (uncomment if needed)
# cleaned_df.to_csv('ice_cep_prices_20251002.psv', sep='|', index=False)
# print("Original file overwritten with cleaned data")
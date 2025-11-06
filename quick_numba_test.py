#!/usr/bin/python3
"""Quick test of Numba optimization"""

import time
import pandas as pd
import muniBond

print("Testing Numba-optimized YTW calculation...")
print("=" * 60)

# Load small sample
df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned.psv', sep='|')
df_sample = df.sample(n=10, random_state=42)

muniBond.clear_timing_data()

results = []
for idx, row in df_sample.iterrows():
    cusip = row.get('securityId')
    bid_px = row.get('bidPx')
    
    try:
        price = int(bid_px)
        bond = muniBond.muniBond(cusip)
        
        start = time.time()
        ytw = bond.ytw(price)
        elapsed = (time.time() - start) * 1000
        
        timing = muniBond.get_cusip_timing(cusip)
        results.append({
            'cusip': cusip,
            'ytw': ytw,
            'total_ms': elapsed,
            'init_ms': timing['init_ms'],
            'cf_ms': timing['generate_cashflows_ms'],
            'newton_ms': timing['newton_solver_ms']
        })
        print(f"✓ {cusip}: {elapsed:.2f}ms (YTW={ytw:.3f}%)")
    except Exception as e:
        print(f"✗ {cusip}: {e}")

print("\n" + "=" * 60)
results_df = pd.DataFrame(results)
print(f"Average total time: {results_df['total_ms'].mean():.2f}ms")
print(f"Average Newton time: {results_df['newton_ms'].mean():.2f}ms")
print(f"Average CF time: {results_df['cf_ms'].mean():.2f}ms")
print("\nNumba optimization active!")

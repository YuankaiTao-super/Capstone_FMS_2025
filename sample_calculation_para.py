#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-10
completed: In progress
issue #3,4,9
this version does NOT use an initial guess from PSV data for YTW calculation
UPDATED: Parallel processing at invocation layer for 8x speedup
"""
import time
import pandas as pd
import numpy as np
import gc
from multiprocessing import Pool, cpu_count
import muniBond

def warm_up_numba():
    """Pre-compile Numba with cusip warm-up"""
    print("Warming up Numba...")
    
    warm_cusip = '797299KR4'
    warm_bond = muniBond.muniBond(warm_cusip)
    _ = warm_bond.ytw(100.0)

    print("Warm-up complete!") 

def calculate_ytw_for_bond(args):
    """Worker function for parallel processing"""
    cusip, bid_px = args
    
    try:
        price = int(bid_px)
        bond = muniBond.muniBond(cusip)
        
        start_time = time.time()
        ytw = bond.ytw(price)
        calc_time = (time.time() - start_time) * 1000
        
        timing_details = muniBond.get_cusip_timing(cusip)
        
        return {
            'cusip': cusip,
            'original_price': bid_px,
            'truncated_price': price,
            'calc_time_ms': calc_time,
            'generate_cashflows_ms': timing_details['generate_cashflows_ms'],
            'newton_solver_ms': timing_details['newton_solver_ms'],
            'ytw': ytw,
            'success': True
        }
    except Exception as e:
        return {
            'cusip': cusip,
            'original_price': bid_px,
            'error': str(e),
            'success': False
        }

def process_data_parallel():
    """
    10_000 samples, multiprocessing
    """
    df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned_01.psv', sep='|')
    print(f"Total records: {len(df)}")

    df_sample = df.sample(n=10_000, random_state=42)
    
    muniBond.clear_timing_data()
    warm_up_numba()
    muniBond.clear_timing_data()

    # Prepare arguments for parallel processing
    args_list = [(row['securityId'], row['bidPx']) for _, row in df_sample.iterrows()]
    
    # Determine number of processes (leave 1 core free)
    n_processes = 4
    print(f"Using {n_processes} processes for parallel calculation")
    
    print("Beginning parallel processing...")
    total_start_time = time.time()
    
    # Main calculation
    with Pool(processes=n_processes) as pool:
        results = pool.map(calculate_ytw_for_bond, args_list, chunksize=100)
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    successes = [r for r in results if r.get('success', False)]
    errors = [r for r in results if not r.get('success', False)]
    
    results_df = pd.DataFrame(successes)
    
    print(f"\n===== PARA SUMMARY =====")
    print(f"Bonds processed: {len(successes)}")
    print(f"Errors: {len(errors)}")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Throughput: {len(successes)/total_processing_time:.1f} bonds/sec")

    print(f"\nTotal Calculation Time:")
    print(f"  Mean: {results_df['calc_time_ms'].mean():.4f}ms")
    print(f"  Median: {results_df['calc_time_ms'].median():.4f}ms")
    print(f"  Max: {results_df['calc_time_ms'].max():.4f}ms")
    print(f"  Min: {results_df['calc_time_ms'].min():.4f}ms")
    
    print(f"\nNewton Solver Time:")
    print(f"  Mean: {results_df['newton_solver_ms'].mean():.4f}ms")
    print(f"  Median: {results_df['newton_solver_ms'].median():.4f}ms")
    print(f"  Max: {results_df['newton_solver_ms'].max():.4f}ms")
    print(f"  Min: {results_df['newton_solver_ms'].min():.4f}ms")

    results_df.to_csv('./temp/ytw_calc_times_para.csv', index=False)
    print("\nResult file saved to ./temp/ytw_calc_times_para.csv")

    return results_df

if __name__ == "__main__":
    results = process_data_parallel()
    print("DONE")
#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-10
completed: In progress
issue #3,4,9
this version does NOT use an initial guess from PSV data for YTW calculation
real cusips from cleaned PSV data(cleaned with sample_cusip.csv) with warm-up function
"""
import time
import pandas as pd
import numpy as np
import gc
from contextlib import contextmanager
import muniBond
from multiprocessing import Pool, cpu_count

def warm_up_numba():
    """Pre-compile Numba with cusip warm-up"""
    print("Warming up...")
    
    warm_cusip = '797299KR4'
    warm_bond = muniBond.muniBond(warm_cusip)
    _ = warm_bond.ytw(100.0)

    print("Warm-up complete!") 

def process_psv_data():
    """
    num of samples -> 10,000
    """
    df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned_01.psv', sep='|')
    print(len(df))

    df_sample = df.sample(n=10_000, random_state=42)
    
    muniBond.clear_timing_data()

    warm_up_numba()
    muniBond.clear_timing_data()

    results = []
    blacklist_cusips = []
    problem_cusips = [
        '130795DH7'
    ]
    
    print("begin to calc...")
    
    total_start_time = time.time()
    
    for idx, row in df_sample.iterrows():

        cusip = row.get('securityId')
        bid_px = row.get('bidPx')

        if cusip in problem_cusips:
            print(f"skip: {cusip}")
            continue

        # print(f"cusip:{cusip}, bid_px:{bid_px}")

        try:
            price = int(bid_px)  # truncate decimal
            
            # create bond object
            bond = muniBond.muniBond(cusip)

            # calc time
            start_time = time.time()
            ytw = bond.ytw(price)
            end_time = time.time()
            calc_time = (end_time - start_time) * 1000  # -> ms

            timing_details = muniBond.get_cusip_timing(cusip)

            # result_cols
            result = {
                'cusip': cusip,
                'original_price': bid_px,
                'truncated_price': price,
                'calc_time_ms': calc_time,
                'init_ms': timing_details['init_ms'],
                'generate_cashflows_ms': timing_details['generate_cashflows_ms'],
                'newton_solver_ms': timing_details['newton_solver_ms'],
                'ytw': ytw
            }
            
            # save the original data if needed
            # for col in df_sample.columns:
            #     if col not in ['securityId', 'bidPx']:
            #         result[f'original_{col}'] = row[col]
            
            results.append(result)
            
            if len(results) % 1000 == 0:
                print(f"processing:{len(results)}")
                print(f"errors:{len(blacklist_cusips)}")
                # gc.collect()

        except Exception as e:
            error_info = {
                'cusip': cusip,
                'original_price': bid_px,
                'truncated_price': price,
                'error': str(e)
            }

            blacklist_cusips.append(error_info)
            continue

    results_df = pd.DataFrame(results)
    
    # Calculate total processing time
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    print(f"\n===== SUMMARY =====")
    print(f"Bonds processed: {len(results_df)}")
    print(f"Total processing time: {total_processing_time:.2f}s")

    print(f"\nTotal Calculation Time:")
    print(f"  Mean: {results_df['calc_time_ms'].mean():.4f}ms")
    print(f"  Median: {results_df['calc_time_ms'].median():.4f}ms")
    print(f"  Max: {results_df['calc_time_ms'].max():.4f}ms")
    print(f"  Min: {results_df['calc_time_ms'].min():.4f}ms")
    
    print(f"\nInit Time:")
    print(f"  Mean: {results_df['init_ms'].mean():.4f}ms")
    print(f"  Median: {results_df['init_ms'].median():.4f}ms")
    print(f"  Max: {results_df['init_ms'].max():.4f}ms")
    print(f"  Min: {results_df['init_ms'].min():.4f}ms")
    
    print(f"\nNewton Solver Time:")
    print(f"  Mean: {results_df['newton_solver_ms'].mean():.4f}ms")
    print(f"  Median: {results_df['newton_solver_ms'].median():.4f}ms")
    print(f"  Max: {results_df['newton_solver_ms'].max():.4f}ms")
    print(f"  Min: {results_df['newton_solver_ms'].min():.4f}ms")

    # save timing files
    results_df.to_csv('./temp/ytw_calc_times.csv', index=False)
    print("\n result file saved")

    # save error files
    # error_df = pd.DataFrame(blacklist_cusips)
    # print(f"Errors: {len(error_df)}")
    # error_df.to_csv('./ytw_calc_blackList.csv', index=False)
    # print("\n blacklist file saved")
    
    return results_df

if __name__ == "__main__":
    results = process_psv_data()
    print("DONE")
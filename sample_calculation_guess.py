#!/usr/bin/python3
"""
***Deprecated: no guess in the current muniBond module
author: @yuankai
created: 2024-10-10
completed: In progress
issue #3, #4
this version uses an initial guess from PSV data for YTW calculation
"""
# performance test for muniBond module using modified price from PSV data
import time
import pandas as pd
import numpy as np
import gc
from contextlib import contextmanager
import muniBond


def process_psv_data_guess():
    """
    num of samples -> 10_000
    """
    df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned.psv', sep='|')
    print(len(df))

    df_sample = df.sample(n=10_000, random_state=42)
    
    results = []
    blacklist_cusips = []
    problem_cusips = [
        '130795DH7'
    ]
    
    print("begin to calc...")
    
    for idx, row in df_sample.iterrows():

        cusip = row.get('securityId')
        bid_px = row.get('bidPx')
        yield_guess = row.get('bidYield')

        if cusip in problem_cusips:
            print(f"skip: {cusip}")
            continue

        print(f"cusip:{cusip}, bid_px:{bid_px}")

        try:
            price = int(bid_px)  # truncate decimal
            
            # create bond object
            bond = muniBond.muniBond(cusip)

            overrideGuess = yield_guess/100 if not np.isnan(yield_guess) else None
            
            # calc time
            start = time.perf_counter_ns()
            ytw = bond.ytw(price, None, overrideGuess)
            end = time.perf_counter_ns()
            calc_time = (end - start)

            # result_cols
            result = {
                'cusip': cusip,
                'bidPx': bid_px,
                'truncated_price': price,
                'bidYield': yield_guess,
                'calc_time_ns': calc_time,
                'ytw': ytw
            }
            
            # save the original data
            for col in df_sample.columns:
                if col not in ['securityId', 'bidPx', 'bidYield']:
                    result[f'original_{col}'] = row[col]
            
            results.append(result)
            
            if len(results) % 100 == 0:
                print(f"processing:{len(results)}")
                print(f"errors:{len(blacklist_cusips)}")
                gc.collect()

        except Exception as e:
            error_info = {
                'cusip': cusip,
                'bidPx': bid_px,
                'bidYield': yield_guess,
                'error': str(e)
            }

            blacklist_cusips.append(error_info)
            continue

    results_df = pd.DataFrame(results)
    
    # stats summary
    avg_calc_time = results_df['calc_time_ns'].mean()
    max_calc_time = results_df['calc_time_ns'].max()
    min_calc_time = results_df['calc_time_ns'].min()

    print(f"\n===== stats summary =====")
    print(f"N: {len(results_df)}")
    print(f"avg time with guess: {avg_calc_time:.2f}ns")
    print(f"max time with guess: {max_calc_time:.2f}ns")
    print(f"min time with guess: {min_calc_time:.2f}ns")

    results_df.to_csv('./temp/ytw_calc_times_guess.csv', index=False)
    print("\n result file saved")

    error_df = pd.DataFrame(blacklist_cusips)
    print(f"Errors: {len(error_df)}")
    error_df.to_csv('./temp/ytw_calc_blackList_guess.csv', index=False)
    print("\n blacklist file saved")
    
    return results_df

if __name__ == "__main__":
    results = process_psv_data_guess()
    print("DONE")

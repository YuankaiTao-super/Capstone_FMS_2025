#!/usr/bin/python3
"""
author: @yuankai
created: 2024-10-10
completed: In progress
issue #3, #4
"""
# performance test for muniBond module using modified price from PSV data
import time
import pandas as pd
import numpy as np
import gc
from contextlib import contextmanager
import muniBond


def process_psv_data():
    """
    num of samples -> 10,000
    """
    df = pd.read_csv('./ice_cep_prices_20251002_cleaned.psv', sep='|')
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

        if cusip in problem_cusips:
            print(f"skip: {cusip}")
            continue

        print(f"cusip:{cusip}, bid_px:{bid_px}")

        try:
            price = int(bid_px)  # truncate decimal
            
            # create bond object
            bond = muniBond.muniBond(cusip)

            # calc time
            start_time = time.time()
            ytw = bond.ytw(price)
            calc_time = (time.time() - start_time) * 1000  # -> ms

            # result_cols
            result = {
                'cusip': cusip,
                'original_price': bid_px,
                'truncated_price': price,
                'calc_time_ms': calc_time,
                'ytw': ytw
            }
            
            # save the original data
            for col in df_sample.columns:
                if col not in ['securityId', 'bidPx']:
                    result[f'original_{col}'] = row[col]
            
            results.append(result)
            
            if len(results) % 100 == 0:
                print(f"processing:{len(results)}")
                print(f"errors:{len(blacklist_cusips)}")
                gc.collect()

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
    
    # stats summary
    avg_calc_time = results_df['calc_time_ms'].mean()
    max_calc_time = results_df['calc_time_ms'].max()
    min_calc_time = results_df['calc_time_ms'].min()

    print(f"\n===== stats summary =====")
    print(f"N: {len(results_df)}")
    print(f"avg time: {avg_calc_time:.2f}ms")
    print(f"max time: {max_calc_time:.2f}ms")
    print(f"min time: {min_calc_time:.2f}ms")
    
    results_df.to_csv('./ytw_calc_times.csv', index=False)
    print("\n result file saved")

    error_df = pd.DataFrame(blacklist_cusips)
    print(f"Errors: {len(error_df)}")
    error_df.to_csv('./ytw_calc_blackList.csv', index=False)
    print("\n blacklist file saved")
    
    return results_df

if __name__ == "__main__":
    results = process_psv_data()
    print("DONE")

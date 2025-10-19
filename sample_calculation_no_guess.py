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
    df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned.psv', sep='|')
    print(len(df))

    df_sample = df.sample(n=10_000, random_state=42)
    
    muniBond.clear_timing_data()

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

        # print(f"cusip:{cusip}, bid_px:{bid_px}")

        try:
            price = int(bid_px)  # truncate decimal
            
            # create bond object
            bond = muniBond.muniBond(cusip)

            # calc time
            start_time = time.time()
            ytw = bond.ytw(price)
            end_time = time.time()
            elapsed = end_time - start_time
            calc_time = elapsed * 1000  # -> ms

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

    avg_init_time = results_df['init_ms'].mean()
    max_init_time = results_df['init_ms'].max()
    min_init_time = results_df['init_ms'].min()

    avg_cf_time = results_df['generate_cashflows_ms'].mean()
    max_cf_time = results_df['generate_cashflows_ms'].max()
    min_cf_time = results_df['generate_cashflows_ms'].min()

    avg_newton_time = results_df['newton_solver_ms'].mean()
    max_newton_time = results_df['newton_solver_ms'].max()
    min_newton_time = results_df['newton_solver_ms'].min()

    print(f"\n===== stats summary =====")
    print(f"N: {len(results_df)}")
    print(f"avg calc time: {avg_calc_time:.2f}ms")
    print(f"max time: {max_calc_time:.2f}ms")
    print(f"min time: {min_calc_time:.2f}ms")

    print(f"\n===== init stats summary =====")
    print(f"avg init time: {avg_init_time:.2f}ms")
    print(f"max init time: {max_init_time:.2f}ms")
    print(f"min init time: {min_init_time:.2f}ms")

    print(f"\n===== cf stats summary =====")
    print(f"avg generate_cashflows time: {avg_cf_time:.2f}ms")
    print(f"max generate_cashflows time: {max_cf_time:.2f}ms")
    print(f"min generate_cashflows time: {min_cf_time:.2f}ms")

    print(f"\n===== ns stats summary =====")
    print(f"avg newton_solver time: {avg_newton_time:.2f}ms")
    print(f"max newton_solver time: {max_newton_time:.2f}ms")
    print(f"min newton_solver time: {min_newton_time:.2f}ms")

    # save timing files
    # results_df.to_csv('./ytw_calc_times.csv', index=False)
    # print("\n result file saved")

    # save error files
    # error_df = pd.DataFrame(blacklist_cusips)
    # print(f"Errors: {len(error_df)}")
    # error_df.to_csv('./ytw_calc_blackList.csv', index=False)
    # print("\n blacklist file saved")
    
    return results_df

if __name__ == "__main__":
    results = process_psv_data()
    print("DONE")
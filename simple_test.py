#!/usr/bin/python3
"""
simple performance test for muniBond module
"""
import time
import muniBond
import random

def simple_test():
    # get a sample CUSIP
    random.seed(42)
    random_index = random.randint(0, len(muniBond.secMaster.index) - 1)
    cusip = list(muniBond.secMaster.index)[random_index]
    
    # test bond creation
    bond = muniBond.muniBond(cusip)
    
    # test price calculation (yield-to-price)
    start = time.time()
    try:
        price = bond.price(0.05)  # 5% yield
        price_time = (time.time() - start) * 1000
        print(f"price calculation: {price_time:.1f}ms")
    except Exception as e:
        print("fail to calculate price")

    # test calculation for price-to-yield
    start = time.time()
    try:
        ytw = bond.ytw(100.0)  # par value
        yield_time = (time.time() - start) * 1000
        print(f"yld calculation: {yield_time:.1f}ms")
    except:
        print("fail to calculate yld")

if __name__ == "__main__":
    simple_test()
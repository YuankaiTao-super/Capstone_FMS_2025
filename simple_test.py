#!/usr/bin/python3
"""
simple performance test for muniBond module
"""
import time
import muniBond
import random

# use scalene simple_test.py
# pick a random bond from the database
random.seed(42)
random_index = random.randint(0, len(muniBond.secMaster.index) - 1)
cusip = list(muniBond.secMaster.index)[random_index]
print(f"test CUSIP: {cusip}")
price = 105.0

bond = muniBond.muniBond(cusip)

def ytw_test():
    bond.ytw(price)

    start = time.time()
    ytw = bond.ytw(price)
    yield_time = (time.time() - start) * 1_000_000

    print(f"calculation time: {yield_time:.1f}us")
    print(f"YTW result: {ytw:.3f}%")

if __name__ == "__main__":
    ytw_test()
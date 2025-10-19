#!/usr/bin/python3
"""
simple performance test for muniBond module
"""
import time
import muniBond
import random
import cProfile
import pstats

# use scalene simple_test.py
# pick a random bond from the database
random.seed(42)
random_index = random.randint(0, len(muniBond.secMaster.index) - 1)
# cusip = list(muniBond.secMaster.index)[random_index]

cusip = '79130MUE6'

print(f"test CUSIP: {cusip}")
price = 105.0

bond = muniBond.muniBond(cusip)

profiler = cProfile.Profile()
profiler.enable()
# start = time.perf_counter_ns()
ytw = bond.ytw(price)
# end = time.perf_counter_ns()
# elapsed = (end - start)
profiler.disable()

# print(f"calculation time: {elapsed:.1f}ns")
print(f"YTW result: {ytw:.3f}%")

stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(5)
#!/usr/bin/python3
"""
Simple performance test: scipy.optimize.newton vs numba-optimized newton solver
"""
import time
import numpy as np
from scipy.optimize import newton
from numba_pricing import newton_solver_optimized

# Test function: f(x) = x^2 - 2, root at x = sqrt(2) ≈ 1.414
def test_func(x):
    return x**2 - 2

# Run benchmark
def benchmark_solvers(num_iterations=10000):
    x0 = 1.0
    
    # Scipy Newton
    start = time.perf_counter_ns()
    for _ in range(num_iterations):
        result_scipy = newton(test_func, x0, tol=1e-8, maxiter=100)
    scipy_time = time.perf_counter_ns() - start
    
    # Numba Newton
    start = time.perf_counter_ns()
    for _ in range(num_iterations):
        result_numba = newton_solver_optimized(test_func, x0, tol=1e-8, maxiter=100)
    numba_time = time.perf_counter_ns() - start
    
    # Results
    print(f"Iterations: {num_iterations}")
    print(f"Scipy result: {result_scipy:.10f}")
    print(f"Numba result: {result_numba:.10f}")
    print(f"Scipy time: {scipy_time/1_000_000:.2f}ms ({scipy_time/num_iterations:.0f}ns per iteration)")
    print(f"Numba time: {numba_time/1_000_000:.2f}ms ({numba_time/num_iterations:.0f}ns per iteration)")
    print(f"Speedup: {scipy_time/numba_time:.2f}x")

if __name__ == "__main__":
    benchmark_solvers()

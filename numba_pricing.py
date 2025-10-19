import functools
import numpy as np
from numba import njit
from scipy.optimize import newton

ngjit = functools.partial(njit, cache=True, nogil=True)

@ngjit("f8(f8, f8, f8, f8)")
def within_tol(x: float, y: float, atol: float, rtol: float) -> bool:
    """Check if two float numbers are within a tolerance"""
    return np.abs(x - y) <= atol + rtol * np.abs(y)

@ngjit("f8(f8, f8)")
def newton_step_numba(f_val: float, f_prime: float) -> float:
    """Numba optimized Newton step calculation"""
    if np.abs(f_prime) < 1e-15:
        return np.inf
    return f_val / f_prime

@ngjit("f8(f8, f8, f8)")
def numerical_derivative(func_at_x: float, func_at_x_plus: float, epsilon: float) -> float:
    """Numerical derivative calculation"""
    return (func_at_x_plus - func_at_x) / epsilon

def newton_solver_optimized(func, x0, args=(), tol=1e-8, maxiter=100, epsilon=1e-5):
    """
    Optimized Newton solver v.1.0
    """
    x = float(x0)
    
    for i in range(maxiter):
        f_val = func(x, *args)
        
        if within_tol(f_val, 0.0, tol, 0.0):
            return x
            
        # numerical derivative
        f_val_plus = func(x + epsilon, *args)
        f_prime = numerical_derivative(f_val, f_val_plus, epsilon)
        
        # step
        newton_step = newton_step_numba(f_val, f_prime)
        
        if np.isinf(newton_step):
            raise RuntimeError(f"Derivative is zero at iteration {i}")
            
        x = x - newton_step
        
        if np.abs(newton_step) < tol: # check if within tol
            return x
    
    raise RuntimeError(f"Failed to converge after {maxiter} iterations")

# Batch processing module
# @njit("f8[:](f8[:], f8, f8, i8)")
# def batch_newton_steps(f_vals: np.ndarray, f_primes: np.ndarray, 
#                       current_x: float, maxiter: int) -> np.ndarray:
#     """
#     Batch processing of Newton steps
#     """
#     n = len(f_vals)
#     results = np.empty(n, dtype=np.float64)
    
#     for i in range(n):
#         x = current_x
#         for j in range(maxiter):
#             if within_tol(f_vals[i], 0.0, 1e-8, 0.0):
#                 results[i] = x
#                 break
#             step = newton_step_numba(f_vals[i], f_primes[i])
#             if np.isinf(step):
#                 results[i] = np.inf
#                 break
#             x = x - step
#         else:
#             results[i] = np.inf

#     return results
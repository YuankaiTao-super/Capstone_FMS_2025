# 创建 fast_bond_pricing.py
import functools
import numpy as np
from numba import njit, types
from numba.typed import Dict

ngjit = functools.partial(njit, cache=True, nogil=True)

@ngjit("f8(f8, f8, f8, f8, f8)")
def simple_bond_price_numba(
    yield_rate: float, 
    coupon_rate: float, 
    years_to_maturity: float, 
    face_value: float = 100.0,
    freq: float = 2.0
) -> float:
    """
    fully Numba-optimized bond pricing function
    """
    if years_to_maturity <= 0.5:
        # 短期债券
        return face_value + coupon_rate * face_value * years_to_maturity
    
    periods = years_to_maturity * freq
    coupon_payment = coupon_rate * face_value / freq
    discount_rate = yield_rate / freq
    
    if abs(discount_rate) < 1e-10:
        # 收益率接近0
        return face_value + coupon_payment * periods
    
    # 标准债券定价公式
    pv_coupons = coupon_payment * (1.0 - (1.0 + discount_rate)**(-periods)) / discount_rate
    pv_principal = face_value * (1.0 + discount_rate)**(-periods)
    
    return pv_coupons + pv_principal

@ngjit("f8(f8, f8, f8, f8, f8)")
def bond_price_root_numba(y: float, target_price: float, coupon: float, 
                         years_to_maturity: float, freq: float) -> float:
    """完全Numba化的根函数"""
    price = simple_bond_price_numba(y, coupon, years_to_maturity, 100.0, freq)
    return price - target_price

def newton_solver_full_numba(target_price: float, coupon: float, years_to_maturity: float,
                            initial_guess: float = 0.05, freq: float = 2.0) -> float:
    """
    完全Numba化的Newton求解器 - 这里才能看到真正的加速
    """
    @ngjit
    def newton_iterations(guess, target, cpn, years, frequency):
        x = guess
        tol = 1e-8
        max_iter = 50
        
        for i in range(max_iter):
            f_val = simple_bond_price_numba(x, cpn, years, 100.0, frequency) - target
            
            if abs(f_val) < tol:
                return x
            
            # 数值导数
            epsilon = 1e-6
            f_plus = simple_bond_price_numba(x + epsilon, cpn, years, 100.0, frequency) - target
            f_prime = (f_plus - f_val) / epsilon
            
            if abs(f_prime) < 1e-12:
                return np.inf
            
            step = f_val / f_prime
            x = x - step
            
            if abs(step) < tol:
                return x
        
        return np.inf
    
    return newton_iterations(initial_guess, target_price, coupon, years_to_maturity, freq)
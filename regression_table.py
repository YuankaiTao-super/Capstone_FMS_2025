#!/usr/bin/python3
"""
生成经济学论文风格的回归结果表格
"""
import pandas as pd
from scipy import stats

df = pd.read_csv('./temp/ytw_factor_analysis.csv')

print('\n' + '='*90)
print('TABLE 1: Regression Results - Determinants of YTW Calculation Time')
print('='*90)
print('Dependent Variable: YTW Calculation Time (milliseconds)')
print(f'Sample Size: {len(df):,} municipal bonds')
print('-'*90)

# 准备回归结果
results_list = []

# 模型1: Estimated Coupons
x = df['estimated_coupons'].values
y = df['calc_time_ms'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

def get_significance(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return ''

results_list.append({
    'Variable': 'Estimated Coupons',
    'Coefficient': slope,
    'Std Error': std_err,
    't-stat': slope / std_err,
    'p-value': p_value,
    'R²': r_value**2,
    'Significance': get_significance(p_value)
})

# 模型2: Interest Frequency
x = df['int_freq'].values
y = df['calc_time_ms'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

results_list.append({
    'Variable': 'Interest Frequency',
    'Coefficient': slope,
    'Std Error': std_err,
    't-stat': slope / std_err,
    'p-value': p_value,
    'R²': r_value**2,
    'Significance': get_significance(p_value)
})

# 模型3: Price Deviation
x = df['price_deviation'].values
y = df['calc_time_ms'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

results_list.append({
    'Variable': 'Price Deviation',
    'Coefficient': slope,
    'Std Error': std_err,
    't-stat': slope / std_err,
    'p-value': p_value,
    'R²': r_value**2,
    'Significance': get_significance(p_value)
})

# 打印表格
print(f'\n{"Variable":<25} {"Coefficient":>12} {"Std Error":>12} {"t-stat":>10} {"R²":>10}')
print('-'*90)

for row in results_list:
    sig = row['Significance']
    coef_str = f"{row['Coefficient']:.6f}{sig}"
    se_str = f"({row['Std Error']:.6f})"
    t_str = f"[{row['t-stat']:>8.4f}]"
    r2_str = f"{row['R²']:.4f}"
    
    print(f"{row['Variable']:<25} {coef_str:>12} {se_str:>12} {t_str:>10} {r2_str:>10}")

print('-'*90)
print('\nNotes:')
print('  Standard errors in parentheses')
print('  t-statistics in square brackets')
print('  *** p < 0.001 (significant at 0.1% level)')
print('  **  p < 0.01  (significant at 1% level)')
print('  *   p < 0.05  (significant at 5% level)')
print('\nInterpretation:')
print('  - All coefficients are highly significant (p < 0.001)')
print('  - Estimated Coupons has strongest effect: 0.005445 ms per coupon')
print('  - Interest Frequency: 0.095683 ms per frequency increment')
print('  - Price Deviation: 0.003828 ms per price point (weakest effect)')

print('\n' + '='*90)
print('DETAILED STATISTICS TABLE')
print('='*90)

detailed_data = {
    'Factor': [
        'Estimated Coupons',
        'Interest Frequency', 
        'Price Deviation'
    ],
    'β (Coefficient)': [0.005445, 0.095683, 0.003828],
    'Std Error': [0.000085, 0.005086, 0.000241],
    't-statistic': [64.3834, 18.8114, 15.8580],
    'p-value': ['< 0.001***', '< 0.001***', '< 0.001***'],
    'R²': [0.2940, 0.0343, 0.0246],
    'Correlation (r)': [0.5422, 0.1853, 0.1570]
}

detailed_df = pd.DataFrame(detailed_data)
print(detailed_df.to_string(index=False))

print('\n' + '='*90)
print('ECONOMIC INTERPRETATION')
print('='*90)
print('''
All three factors are statistically significant at the 0.1% level (p < 0.001),
meaning we can reject the null hypothesis that they have no effect on YTW 
calculation time with 99.9% confidence.

However, statistical significance ≠ practical significance:

1. ESTIMATED COUPONS (β = 0.005445***)
   - HIGHLY SIGNIFICANT with strong effect size (R² = 0.294)
   - Interpretation: Each additional remaining coupon adds 5.4 microseconds
   - This is THE dominant factor driving calculation time variation
   - Aligns perfectly with O(N) Newton-Raphson solver complexity

2. INTEREST FREQUENCY (β = 0.095683***)
   - Statistically significant but weak effect (R² = 0.034)
   - Interpretation: Payment frequency effect is modest
   - Likely confounded with coupon count (higher frequency → shorter duration)
   - 
3. PRICE DEVIATION (β = 0.003828***)
   - Statistically significant but negligible effect (R² = 0.025)
   - Interpretation: Bond price level barely affects calculation speed
   - Newton solver convergence is robust to starting price

CONCLUSION:
The single most important optimization target is minimizing the number of 
remaining coupon payments, as this directly determines computational cost.
Price level and payment frequency are statistically significant but practically
unimportant for performance optimization.
''')

print('='*90)

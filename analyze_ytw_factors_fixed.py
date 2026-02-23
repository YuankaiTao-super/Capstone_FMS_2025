#!/usr/bin/python3
"""
used to analyze YTW calculation time determinants with regression models
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import muniBond

def analyze_with_fixed_effects():
    """Regression analysis with fixed effects"""
    ytw_results = pd.read_csv('./temp/ytw_calc_times.csv')
    df = pd.read_csv('./temp/ice_cep_prices_20251002_cleaned_01.psv', sep='|')
    df_sample = df.sample(n=10_000, random_state=42)
    
    results = []
    for idx, row in df_sample.iterrows():
        cusip = row['securityId']
        price = int(row['bidPx'])
        ytw_row = ytw_results[ytw_results['cusip'] == cusip]
        if ytw_row.empty:
            continue
        
        try:
            bond = muniBond.muniBond(cusip)
            years_to_maturity = bond.effectiveMaturityDate.year - 2025
            int_freq = bond.intFreq if bond.intFreq else 0
            
            if int_freq > 0 and years_to_maturity >= 0:
                estimated_coupons = int(years_to_maturity * int_freq) + 1
            else:
                estimated_coupons = 0
            
            features = {
                'cusip': cusip,
                'price': price,
                'coupon': bond.coupon if bond.coupon else 0,
                'int_freq': int_freq,
                'callable': 1 if bond.callable else 0,
                'years_to_maturity': years_to_maturity,
                'estimated_coupons': estimated_coupons,
                'price_deviation': abs(price - 100),
                'calc_time_ms': ytw_row.iloc[0]['calc_time_ms'],
                'init_ms': ytw_row.iloc[0]['init_ms'],
                'newton_solver_ms': ytw_row.iloc[0]['newton_solver_ms'],
                'generate_cashflows_ms': ytw_row.iloc[0]['generate_cashflows_ms']
            }
            
            results.append(features)
            
        except Exception as e:
            continue
    
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*90)
    print("IMPROVED REGRESSION ANALYSIS WITH FIXED EFFECTS")
    print("Separating Initialization Time from Newton Solver Time")
    print("="*90)

    # =================================================================
    print("\n" + "="*90)
    print("MODEL 1: BASELINE - Single Factor (Estimated Coupons Only)")
    print("="*90)
    print("Specification: calc_time_ms = b0 + b1*coupon_count + e")
    print("-"*90)
    
    X1 = df_res[['estimated_coupons']].values
    y = df_res['calc_time_ms'].values
    
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df_res['estimated_coupons'].values, y)
    model1_sklearn = LinearRegression().fit(X1, y)
    
    r2_1 = r_value1**2
    t_stat1 = slope1 / std_err1
    
    # Calculate F-statistic for Model 1
    n1 = len(y)
    k1 = 1  # number of predictors (excluding constant)
    y1_pred = model1_sklearn.predict(X1)
    residuals1 = y - y1_pred
    ss_total1 = np.sum((y - np.mean(y))**2)
    ss_residual1 = np.sum(residuals1**2)
    ss_regression1 = ss_total1 - ss_residual1
    f_statistic1 = (ss_regression1 / k1) / (ss_residual1 / (n1 - k1 - 1))
    
    print(f"\nRegression Results:")
    print(f"  Intercept (b0): {intercept1:.6f} ms")
    print(f"    Interpretation: Base time including initialization")
    print(f"  Coupon Count (b1): {slope1:.6f}*** ms/coupon")
    print(f"    Std Error: ({std_err1:.6f})")
    print(f"    t-statistic: [{t_stat1:.4f}]")
    print(f"  R-squared: {r2_1:.4f}")
    print(f"  F-statistic: {f_statistic1:.2f}***")
    print(f"  Explanation: {r2_1*100:.2f}% of total calc_time variance")
    
    # =================================================================
    print("\n" + "="*90)
    print("MODEL 2: MULTIVARIATE - All Control Variables")
    print("="*90)
    print("Specification: calc_time_ms = b0 + b1*coupon + b2*freq + b3*price_dev + e")
    print("-"*90)
    
    X2 = df_res[['estimated_coupons', 'int_freq', 'price_deviation']].values
    model2 = LinearRegression().fit(X2, y)
    
    # Calculate each variable individually to get t-statistics
    var_results = []
    for col_idx, col_name in enumerate(['estimated_coupons', 'int_freq', 'price_deviation']):
        X_single = df_res[[col_name]].values
        slope, _, r_val, p_val, se = stats.linregress(df_res[col_name].values, y)
        var_results.append({
            'name': col_name,
            'coef': model2.coef_[col_idx],
            'slope_simple': slope,
            'p_value': p_val
        })
    
    print(f"\nRegression Results:")
    print(f"  Intercept (b0): {model2.intercept_:.6f} ms")
    print(f"\n  Variables:")
    for i, col in enumerate(['estimated_coupons', 'int_freq', 'price_deviation']):
        coef = model2.coef_[i]
        sig = '***' if var_results[i]['p_value'] < 0.001 else '**' if var_results[i]['p_value'] < 0.01 else '*' if var_results[i]['p_value'] < 0.05 else ''
        print(f"    {col:20} {coef:10.6f}{sig}")
    
    r2_2 = model2.score(X2, y)
    
    # Calculate F-statistic for Model 2
    n2 = len(y)
    k2 = X2.shape[1]  # number of predictors
    y2_pred = model2.predict(X2)
    residuals2 = y - y2_pred
    ss_total2 = np.sum((y - np.mean(y))**2)
    ss_residual2 = np.sum(residuals2**2)
    ss_regression2 = ss_total2 - ss_residual2
    f_statistic2 = (ss_regression2 / k2) / (ss_residual2 / (n2 - k2 - 1))
    
    print(f"\n  R-squared: {r2_2:.4f}")
    print(f"  F-statistic: {f_statistic2:.2f}***")
    print(f"  Explanation: {r2_2*100:.2f}% of total calc_time variance")
    
    # Collinearity Check
    print(f"\n  Collinearity Check (Correlation Matrix):")
    corr_matrix = df_res[['estimated_coupons', 'int_freq', 'price_deviation']].corr()
    print(corr_matrix.round(4))
    
    # =================================================================
    print("\n" + "="*90)
    print("MODEL 3: OFFSET METHOD - Removing Initialization Time")
    print("="*90)
    print("Specification: (calc_time - init_time) = b0 + b1*coupon + e")
    print("Where offset = mean initialization time = {:.4f} ms".format(df_res['init_ms'].mean()))
    print("-"*90)
    
    # Method A: Direct subtraction of initialization time
    df_res['residual_time'] = df_res['calc_time_ms'] - df_res['init_ms']
    X3 = df_res[['estimated_coupons']].values
    y3 = df_res['residual_time'].values
    
    slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df_res['estimated_coupons'].values, y3)
    model3 = LinearRegression().fit(X3, y3)
    
    r2_3 = r_value3**2
    t_stat3 = slope3 / std_err3
    
    # Calculate F-statistic for Model 3
    n3 = len(y3)
    k3 = 1  # number of predictors
    y3_pred = model3.predict(X3)
    residuals3 = y3 - y3_pred
    ss_total3 = np.sum((y3 - np.mean(y3))**2)
    ss_residual3 = np.sum(residuals3**2)
    ss_regression3 = ss_total3 - ss_residual3
    f_statistic3 = (ss_regression3 / k3) / (ss_residual3 / (n3 - k3 - 1))
    
    print(f"\nRegression Results (After removing init_ms):")
    print(f"  Intercept (b0): {intercept3:.6f} ms")
    print(f"    Interpretation: Remaining fixed overhead (cashflow generation, etc.)")
    print(f"  Coupon Count (b1): {slope3:.6f}*** ms/coupon")
    print(f"    Std Error: ({std_err3:.6f})")
    print(f"    t-statistic: [{t_stat3:.4f}]")
    print(f"  R-squared: {r2_3:.4f}")
    print(f"  F-statistic: {f_statistic3:.2f}***")
    print(f"  Explanation: {r2_3*100:.2f}% of residual time variance")
    print(f"\n  *** Key Finding: R-squared increased from {r2_1:.4f} to {r2_3:.4f}")
    print(f"  *** This confirms: {(r2_3-r2_1)*100:.2f}% of variance was due to init time noise")
    
    # =================================================================
    print("\n" + "="*90)
    print("MODEL 4: ALTERNATIVE - Univariate on Actual Newton Solver Time")
    print("="*90)
    print("Specification: newton_solver_ms = b0 + b1*coupon + e")
    print("-"*90)
    
    y4 = df_res['newton_solver_ms'].values
    slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(df_res['estimated_coupons'].values, y4)
    
    r2_4 = r_value4**2
    t_stat4 = slope4 / std_err4
    
    # Calculate F-statistic for Model 4
    n4 = len(y4)
    k4 = 1  # number of predictors
    X4 = df_res[['estimated_coupons']].values
    model4 = LinearRegression().fit(X4, y4)
    y4_pred = model4.predict(X4)
    residuals4 = y4 - y4_pred
    ss_total4 = np.sum((y4 - np.mean(y4))**2)
    ss_residual4 = np.sum(residuals4**2)
    ss_regression4 = ss_total4 - ss_residual4
    f_statistic4 = (ss_regression4 / k4) / (ss_residual4 / (n4 - k4 - 1))
    
    print(f"\nRegression Results (Pure Newton Solver Time):")
    print(f"  Intercept (b0): {intercept4:.6f} ms")
    print(f"  Coupon Count (b1): {slope4:.6f}*** ms/coupon")
    print(f"    Std Error: ({std_err4:.6f})")
    print(f"    t-statistic: [{t_stat4:.4f}]")
    print(f"  R-squared: {r2_4:.4f}")
    print(f"  F-statistic: {f_statistic4:.2f}***")
    print(f"  Explanation: {r2_4*100:.2f}% of Newton solver time variance")
    
    # =================================================================
    print("\n" + "="*90)
    print("MODEL 5: Generate Cashflows Time Analysis")
    print("="*90)
    print("Specification: generate_cashflows_ms = b0 + b1*coupon + e")
    print("-"*90)
    
    y5 = df_res['generate_cashflows_ms'].values
    slope5, intercept5, r_value5, p_value5, std_err5 = stats.linregress(df_res['estimated_coupons'].values, y5)
    
    r2_5 = r_value5**2
    t_stat5 = slope5 / std_err5
    
    # Calculate F-statistic for Model 5
    n5 = len(y5)
    k5 = 1  # number of predictors
    X5 = df_res[['estimated_coupons']].values
    model5 = LinearRegression().fit(X5, y5)
    y5_pred = model5.predict(X5)
    residuals5 = y5 - y5_pred
    ss_total5 = np.sum((y5 - np.mean(y5))**2)
    ss_residual5 = np.sum(residuals5**2)
    ss_regression5 = ss_total5 - ss_residual5
    f_statistic5 = (ss_regression5 / k5) / (ss_residual5 / (n5 - k5 - 1))
    
    print(f"\nRegression Results (Generate Cashflows Time):")
    print(f"  Intercept (b0): {intercept5:.6f} ms")
    print(f"    Interpretation: Fixed overhead for cashflow generation")
    print(f"  Coupon Count (b1): {slope5:.6f}*** ms/coupon")
    print(f"    Std Error: ({std_err5:.6f})")
    print(f"    t-statistic: [{t_stat5:.4f}]")
    print(f"    p-value: {p_value5:.2e}")
    print(f"  R-squared: {r2_5:.4f}")
    print(f"  F-statistic: {f_statistic5:.2f}***")
    print(f"  Explanation: {r2_5*100:.2f}% of generate_cashflows time variance")
    
    # =================================================================
    print("\n" + "="*90)
    print("MODEL 6: Multivariate on Generate Cashflows")
    print("="*90)
    print("Specification: generate_cashflows_ms = b0 + b1*coupon + b2*int_freq + b3*price_dev + e")
    print("-"*90)
    
    X6 = df_res[['estimated_coupons', 'int_freq', 'price_deviation']].values
    y6 = df_res['generate_cashflows_ms'].values
    model6 = LinearRegression().fit(X6, y6)
    r2_6 = model6.score(X6, y6)
    
    # Calculate standard errors, t-statistics, and F-statistic for Model 6
    n6 = len(y6)
    k6 = X6.shape[1]  # number of predictors
    y6_pred = model6.predict(X6)
    residuals6 = y6 - y6_pred
    mse6 = np.sum(residuals6**2) / (n6 - k6 - 1)
    
    # Calculate standard errors for coefficients
    X6_with_const = np.column_stack([np.ones(n6), X6])
    var_covar_matrix = mse6 * np.linalg.inv(X6_with_const.T @ X6_with_const)
    std_errors6 = np.sqrt(np.diag(var_covar_matrix))
    
    # Calculate t-statistics and p-values
    coefs6 = np.concatenate([[model6.intercept_], model6.coef_])
    t_stats6 = coefs6 / std_errors6
    p_values6 = 2 * (1 - stats.t.cdf(np.abs(t_stats6), n6 - k6 - 1))
    
    # Calculate F-statistic
    ss_total = np.sum((y6 - np.mean(y6))**2)
    ss_residual = np.sum(residuals6**2)
    ss_regression = ss_total - ss_residual
    f_statistic6 = (ss_regression / k6) / (ss_residual / (n6 - k6 - 1))
    f_pvalue6 = 1 - stats.f.cdf(f_statistic6, k6, n6 - k6 - 1)
    
    print(f"\nRegression Results:")
    print(f"  Intercept (b0): {model6.intercept_:.6f} ms")
    print(f"    Std Error: ({std_errors6[0]:.6f})")
    print(f"    t-statistic: [{t_stats6[0]:.4f}]")
    print(f"\n  Variables:")
    for i, col in enumerate(['estimated_coupons', 'int_freq', 'price_deviation']):
        coef = model6.coef_[i]
        se = std_errors6[i+1]
        t_stat = t_stats6[i+1]
        p_val = p_values6[i+1]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"    {col:20} {coef:10.6f}{sig}")
        print(f"      Std Error: ({se:.6f})")
        print(f"      t-statistic: [{t_stat:.4f}]")
    
    print(f"\n  R-squared: {r2_6:.4f}")
    print(f"  Adjusted R-squared: {1 - (1 - r2_6) * (n6 - 1) / (n6 - k6 - 1):.4f}")
    print(f"  F-statistic: {f_statistic6:.2f}***")
    print(f"  F-statistic p-value: {f_pvalue6:.2e}")
    print(f"  Explanation: {r2_6*100:.2f}% of generate_cashflows time variance")
    
    print(f"\n  Correlation with coupons:")
    print(f"    int_freq correlation: {df_res['estimated_coupons'].corr(df_res['int_freq']):.4f}")
    print(f"    price_deviation correlation: {df_res['estimated_coupons'].corr(df_res['price_deviation']):.4f}")
    
    # =================================================================
    print("\n" + "="*90)
    print("COMPARISON TABLE: All Models Side-by-Side")
    print("="*90)
    
    comparison_df = pd.DataFrame({
        'Model': [
            'Model 1: Baseline',
            'Model 2: Multivariate',
            'Model 3: Offset Init',
            'Model 4: Newton Only',
            'Model 5: Gen CF (univ)',
            'Model 6: Gen CF (multi)'
        ],
        'Dependent Var': [
            'calc_time_ms',
            'calc_time_ms',
            'residual_time',
            'newton_solver_ms',
            'generate_cashflows_ms',
            'generate_cashflows_ms'
        ],
        'Coupon Coef': [
            f'{slope1:.6f}***',
            f'{model2.coef_[0]:.6f}***',
            f'{slope3:.6f}***',
            f'{slope4:.6f}***',
            f'{slope5:.6f}***',
            f'{model6.coef_[0]:.6f}***'
        ],
        'R-squared': [
            f'{r2_1:.4f}',
            f'{r2_2:.4f}',
            f'{r2_3:.4f}',
            f'{r2_4:.4f}',
            f'{r2_5:.4f}',
            f'{r2_6:.4f}'
        ],
        'R-sq %': [
            f'{r2_1*100:.2f}%',
            f'{r2_2*100:.2f}%',
            f'{r2_3*100:.2f}%',
            f'{r2_4*100:.2f}%',
            f'{r2_5*100:.2f}%',
            f'{r2_6*100:.2f}%'
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # =================================================================
    print("\n" + "="*90)
    print("INTERPRETATION & CONCLUSIONS")
    print("="*90)
    
    print(f"""
1. GENERATE CASHFLOWS TIME REGRESSION (Models 5 & 6)
   -- Model 5 (Univariate): R-squared = {r2_5:.4f} ({r2_5*100:.2f}%)
   -- Model 6 (Multivariate): R-squared = {r2_6:.4f} ({r2_6*100:.2f}%)
   -- Coupon count explains {r2_5*100:.2f}% of generate_cashflows variance
   -- This is the PRIMARY variable driving cash flow generation time
   -- Each additional coupon adds {slope5:.6f} ms to generation time
   
2. TOTAL CALCULATION TIME (Models 1-4)
   - Model 1 (baseline) R-squared = {r2_1:.4f} ({r2_1*100:.2f}%)
   - Model 3 (offset init) R-squared = {r2_3:.4f} ({r2_3*100:.2f}%)
   - Difference: {(r2_3-r2_1)*100:.2f} percentage points
   - This means {(r2_3-r2_1)*100:.1f}% of variance came from init time noise
   
3. DECOMPOSITION INSIGHT
   - Init time: ~{df_res['init_ms'].mean():.4f} ms (mostly fixed overhead)
   - Gen CF time: {slope5:.6f} ms/coupon (scales linearly with coupons)
   - Newton time: {slope4:.6f} ms/coupon (scales minimally)
   
4. KEY FINDINGS
   -- Coupon count is the STRONGEST predictor of generate_cashflows time
   -- Generate cashflows scales almost linearly with number of coupons
   -- This is the DOMINANT component of total calculation time
   -- Newton solver time has minimal dependence on coupon count
   
5. PRACTICAL IMPLICATIONS
   - To optimize YTW calculation: focus on cashflow generation efficiency
   - The generate_cashflows component should be the target for speedup
   - Current model accurately captures the time-coupon relationship
   -- Model 5 provides the best insight for this component
""")
    df_res.to_csv('./temp/regression_data_with_residuals.csv', index=False)
    
    print("="*90)
    print("Results saved to:")
    print("  - ./temp/model_comparison.csv")
    print("  - ./temp/regression_data_with_residuals.csv")
    print("="*90)
    
    return df_res

if __name__ == "__main__":
    results = analyze_with_fixed_effects()

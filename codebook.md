# Municipal Bond System Variables Codebook

## 📋 Variable Dictionary for muniBond.py

This document provides a comprehensive mapping of all variables used in the Municipal Bond pricing and risk management system.

---

## 🔑 Core Identifiers

| Variable | Type | Description |
|----------|------|-------------|
| `cusip` | string | Committee on Uniform Securities Identification Procedures code - unique 9-character identifier for US securities |
| `isin` | string | International Securities Identification Number - global unique identifier |
| `ticker` | string | Trading symbol or abbreviated identifier |
| `bref` | DataFrame row | Bond reference data row from security master |

---

## 📅 Date Variables

### Maturity & Lifecycle Dates
| Variable | Type | Description |
|----------|------|-------------|
| `maturityDate` | date | Final maturity date of the bond |
| `accrualDate` | date | Date from which interest begins to accrue |
| `firstCouponDate` | date | Date of the first coupon payment |
| `effectiveMaturityDate` | date | Actual maturity considering refundings (refundDate if applicable, else maturityDate) |

### Coupon Payment Dates
| Variable | Type | Description |
|----------|------|-------------|
| `priorCouponDate` | date | Previous coupon payment date (alias: prevCouponDate) |
| `nextCouponDate` | date | Next scheduled coupon payment date |
| `penultCouponDate` | date | Penultimate (second to last) coupon date (alias: lastPeriodAccrualDate) |
| `calcPrevCouponDate` | date | Calculated previous coupon date for pricing calculations |
| `calcNextCouponDate` | date | Calculated next coupon date for pricing calculations |

### Call & Put Dates
| Variable | Type | Description |
|----------|------|-------------|
| `nextCallDate` | date | Next available call date |
| `parCallDate` | date | Date when bond becomes callable at par (100) |
| `nextPutDate` | date | Next available put date |
| `refundDate` | date | Date when bond is scheduled to be refunded |
| `refundAnnounceDate` | date | Date when refunding was announced |

### Calculation Dates
| Variable | Type | Description |
|----------|------|-------------|
| `currentCalcSettleDate` | date | Settlement date for current calculation |
| `settleDate` | date | Trade settlement date (method parameter) |
| `startDate` | date | Start date for period calculations |
| `endDate` | date | End date for period calculations |

---

## 💰 Financial Variables

### Coupon & Interest
| Variable | Type | Description |
|----------|------|-------------|
| `coupon` | float | Annual coupon rate (as decimal, e.g., 0.05 for 5%) |
| `couponType` | string | Type of coupon (Fixed rate, Zero coupon, Variable, etc.) |
| `intFreqDesc` | string | Interest frequency description (Semiannually, Annually, etc.) |
| `intFreq` | float | Interest payment frequency per year (2.0 for semiannual) |
| `intPeriodMonths` | int | Months between interest payments |
| `intPeriodDays` | float | Days in each interest period |

### Pricing
| Variable | Type | Description |
|----------|------|-------------|
| `effectiveMaturityPrice` | float | Price at effective maturity (100.0 or refundPrice) |
| `nextCallPrice` | float | Price if called at next call date |
| `parCallPrice` | float | Price if called at par call date (usually 100.0) |
| `nextPutPrice` | float | Price if put at next put date |
| `refundPrice` | float | Price at which bond will be refunded |

### Yield & Risk
| Variable | Type | Description |
|----------|------|-------------|
| `y` | float | Yield to maturity (as decimal) |
| `yld` | float | Calculated yield result |
| `p` | float | Bond price |
| `prc` | float | Calculated price result |

---

## 📊 Rating Variables

### S&P Ratings
| Variable | Type | Description |
|----------|------|-------------|
| `spLongRating` | string | Standard & Poor's long-term rating (AAA, AA+, AA, etc.) |
| `spUnderlyingRating` | string | S&P underlying credit rating (without insurance enhancement) |
| `spShortRating` | string | S&P short-term rating (A-1+, A-1, etc.) |
| `spIssuerRating` | string | S&P issuer credit rating |
| `spLongOutlook` | string | S&P rating outlook (Stable, Positive, Negative, Developing) |

### Moody's Ratings
| Variable | Type | Description |
|----------|------|-------------|
| `mdyLongRating` | string | Moody's long-term rating (Aaa, Aa1, Aa2, etc.) |
| `mdyEnhancedLongRating` | string | Moody's enhanced long-term rating (with insurance) |
| `mdyInsuredLongRating` | string | Moody's insured long-term rating |
| `mdyShortRating` | string | Moody's short-term rating (P-1, P-2, etc.) |
| `mdyEnhancedShortRating` | string | Moody's enhanced short-term rating |
| `mdyIssuerLongRating` | string | Moody's issuer long-term rating |
| `mdyIssuerShortRating` | string | Moody's issuer short-term rating |
| `mdyShortOutlook` | string | Moody's short-term outlook |
| `mdyLongOutlook` | string | Moody's long-term outlook |

---

## 🔧 Technical Calculation Variables

### Day Count & Calendar
| Variable | Type | Description |
|----------|------|-------------|
| `dayCount` | string | Day count convention (30/360, Actual/365, etc.) |
| `daysInYear` | int | Number of days in year for calculations (360 or 365) |
| `A` | int | Days from accrual date to settlement date |
| `B` | int | Days in year (same as daysInYear) |
| `DIR` | int | Days from accrual date to effective maturity |
| `E` | int | Days between previous and next coupon dates |

### Mathematical Variables
| Variable | Type | Description |
|----------|------|-------------|
| `M` | float | Payment frequency per year (same as intFreq) |
| `N` | int | Number of remaining coupon payments |
| `R` | float | Annual coupon rate (same as coupon) |
| `RV` | float | Redemption value (usually 1.0 or workout price/100) |
| `PV` | float | Present value calculation intermediate |
| `df` | float | Discount factor |
| `DF(t)` | float | Discount factor function at time t |

### Iteration Variables
| Variable | Type | Description |
|----------|------|-------------|
| `k` | int | Loop counter for coupon periods |
| `t` | float | Time parameter for discount factor |
| `current` | date | Current date in cashflow generation loop |

---

## 🎯 Feature Flags & Options

### Call/Put Features
| Variable | Type | Description |
|----------|------|-------------|
| `callable` | boolean | Whether bond is callable before maturity |
| `putType` | string | Type of put feature (if any) |
| `refundType` | string | Type of refunding (Called, Pre-refunded, ETM, etc.) |

### Special Categories
| Variable | Type | Description |
|----------|------|-------------|
| `specialRefundTypes` | list | List of special refund types that modify workout logic |

---

## 💼 Data Structures

### Workout Scenarios
| Variable | Type | Description |
|----------|------|-------------|
| `workouts` | list of dict | All possible maturity scenarios with date, price, and type |
| `workout` | dict | Single workout scenario {date, price, type} |

### Cash Flows
| Variable | Type | Description |
|----------|------|-------------|
| `cashflows` | dict | Dictionary containing coupon and principal cash flows |
| `cashflows['coupon']` | list | List of coupon payment cash flows |
| `cashflows['principal']` | dict | Principal repayment cash flow |

### Black Lists (Compliance)
| Variable | Type | Description |
|----------|------|-------------|
| `blockedCusips` | list | List of blocked CUSIP identifiers |
| `blockedIssuerTokens` | list | List of blocked issuer name keywords |
| `blockedObligorTokens` | list | List of blocked obligor name keywords |
| `blockedStates` | list | List of blocked state codes |

---

## 🏷️ Classification Variables

### Rating Buckets
| Variable | Type | Description |
|----------|------|-------------|
| `ratingBucket` | string | Credit rating classification (AAA, AA, A, BBB, etc.) |
| `compositeRating` | string | Combined S&P and Moody's rating |
| `compositeRatingRank` | int | Numeric rank of composite rating |

### Tenor Buckets
| Variable | Type | Description |
|----------|------|-------------|
| `tenorBucket` | string | Maturity classification (short, front, intermediate, long) |
| `callBucket` | string | Call feature classification |
| `tenor` | int | Years to maturity |
| `callTenor` | int | Years to next call |

### Size & Coupon Buckets
| Variable | Type | Description |
|----------|------|-------------|
| `maturitySizeBucket` | string | Issuance size classification (micro, small, medium, round, large) |
| `cpnBucket` | string | Coupon rate classification (3, 4, 5, 6) |

---

## 🔢 Mathematical Functions & Constants

### MSRB Functions
| Function | Description |
|----------|-------------|
| `msrbDayCount(start, end)` | MSRB 30/360 day count calculation |
| `msrbRoundPrice(prc)` | MSRB price rounding to nearest 1/8 of 1/32 |
| `msrbRoundYield(yld)` | MSRB yield rounding to nearest 0.001% |

### Frequency Mapping
| Variable | Value | Description |
|----------|-------|-------------|
| `regSettleDays` | 1 | Regular settlement days |
| Semiannually | 2.0 | Payment frequency |
| Annually | 1.0 | Payment frequency |
| Monthly | 12.0 | Payment frequency |
| Quarterly | 4.0 | Payment frequency |

---

## 📊 Reference Data Variables

### Bond Characteristics
| Variable | Type | Description |
|----------|------|-------------|
| `issuerName` | string | Name of bond issuer |
| `obligor` | string | Ultimate obligor/guarantor |
| `stateCode` | string | State of issuance |
| `useOfProceeds` | string | Purpose of bond proceeds |
| `fedTaxStatus` | string | Federal tax treatment |
| `stateTaxStatus` | string | State tax treatment |
| `assetStatus` | string | Current status (Live, Matured, Called, etc.) |
| `isDefaulted` | boolean | Whether bond is in default |
| `isBQ` | boolean | Whether bond is bank qualified |

### Trading Parameters
| Variable | Type | Description |
|----------|------|-------------|
| `minPiece` | int | Minimum trading increment |
| `minIncrement` | int | Minimum size increment |
| `principalFactor` | float | Outstanding principal factor (1.0 = full principal) |

---

## 🎯 Eligibility & Compliance Codes

| Code | Meaning |
|------|---------|
| `ELIGIBLE` | Bond passes all eligibility checks |
| `MISSING_REF_DATA` | Required reference data not available |
| `MAKE_WHOLE_CALL` | Bond has make-whole call provision |
| `BANK_QUALIFIED` | Bond is bank qualified (restricted) |
| `IRREGULAR_INTEREST_FREQUENCY` | Non-standard payment frequency |
| `IMMINENT_CALL` | Call date within 1 year |
| `BELOW_RATING_CUTOFF` | Rating below minimum requirement |
| `OUTSIDE_COUPON_RANGE` | Coupon outside acceptable range |
| `BLOCKED_CUSIP` | CUSIP on blocked list |
| `BLOCKED_ISSUER` | Issuer on blocked list |
| `BLOCKED_STATE` | State on blocked list |

---

*This codebook serves as a comprehensive reference for understanding all variables used in the Municipal Bond pricing and risk management system.*
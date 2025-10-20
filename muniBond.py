#!/usr/bin/python3

from unittest import result
import bmaCalendar
import muniRefData
from math import floor
from scipy.optimize import newton
import pandas as pd
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time

_timing_data = {}

secMaster = muniRefData.loadEnrichedMaster(muniRefData.getEnrichedMasterPath())
specialRefundTypes = ['Called','Called by muni-forward','Pre-refunded','Escrowed to maturity','ETM by muni-forward']
blockedCusipDf = pd.read_csv('cusipBlackList.csv')
blockedCusips = list(blockedCusipDf['cusip'])
blockedIssuerDf = pd.read_csv('issuerBlackList.csv')
blockedIssuerTokens = list(blockedIssuerDf['issuerToken'])
blockedObligorDf = pd.read_csv('obligorBlackList.csv')
blockedObligorTokens = list(blockedObligorDf['obligorToken'])
blockedStateDf = pd.read_csv('stateBlackList.csv')
blockedStates = list(blockedStateDf['stateCode'])

def reloadHardBlocks():
    global blockedCusips
    global blockedIssuerTokens
    global blockedStates
    global blockedObligorTokens
    try:
        blockedCusipDf = pd.read_csv('cusipBlackList.csv')
        blockedCusips = list(blockedCusipDf['cusip'])
        blockedIssuerDf = pd.read_csv('issuerBlackList.csv')
        blockedIssuerTokens = list(blockedIssuerDf['issuerToken'])
        blockedStateDf = pd.read_csv('stateBlackList.csv')
        blockedStates = list(blockedStateDf['stateCode'])
        blockedObligorDf = pd.read_csv('obligorBlackList.csv')
        blockedObligorTokens = list(blockedObligorDf['obligorToken'])
    except Exception:
        pass
    return

class muniBond:
    def __init__(self,cusip):
        init_start = time.time()

        self.cusip = cusip
        bref = secMaster.loc[cusip]
        self.maturityDate = None if pd.isna(bref.maturityDate) else bref.maturityDate.date()
        self.accrualDate = None if pd.isna(bref.accrualDate) else bref.accrualDate.date()
        self.firstCouponDate = None if pd.isna(bref.firstCouponDate) else bref.firstCouponDate.date()
        self.spLongRating = bref.spLongRating
        self.spUnderlyingRating = bref.spUnderlyingRating
        self.spShortRating = bref.spShortRating
        self.spIssuerRating = bref.spIssuerRating
        self.spLongOutlook = bref.spLongOutlook
        self.mdyLongRating = bref.mdyLongRating
        self.mdyEnhancedLongRating = bref.mdyEnhancedLongRating
        self.mdyInsuredLongRating = bref.mdyInsuredLongRating
        self.mdyShortRating = bref.mdyShortRating
        self.mdyEnhancedShortRating = bref.mdyEnhancedShortRating
        self.mdyIssuerLongRating = bref.mdyIssuerLongRating
        self.mdyIssuerShortRating = bref.mdyIssuerShortRating
        self.mdyShortOutlook = bref.mdyShortOutlook
        self.mdyLongOutlook = bref.mdyLongOutlook
        self.priorCouponDate = bref.prevCouponDate.date() if not pd.isna(bref.prevCouponDate) else None
        self.nextCouponDate = bref.nextCouponDate.date() if not pd.isna(bref.nextCouponDate) else None
        self.penultCouponDate = bref.lastPeriodAccrualDate.date() if not pd.isna(bref.lastPeriodAccrualDate) else None
        self.couponType = bref.couponType if not pd.isna(bref.couponType) else None
        # self.interestFrequency as a column in the dataframe or defaultly set as Semiannually
        self.intFreqDesc = bref.interestFrequency if (not self.couponType == 'Zero coupon') else 'Semiannually'
        self.intFreq = getIntFreq(self.intFreqDesc)
        self.intPeriodMonths = int(12.0 / self.intFreq) if self.intFreq is not None else None
        self.coupon = bref.coupon/100.0 if not pd.isna(bref.coupon) else None
        self.dayCount = bref.dayCount
        self.daysInYear = 365 if '365' in self.dayCount else 360
        self.intPeriodDays = self.daysInYear/self.intFreq if self.intFreq is not None else None
        self.callable = bref.isCallable if not pd.isna(bref.isCallable) else False
        if (not pd.isna(bref.nextCallDate)) or (not pd.isna(bref.parCallDate)):
            self.callable = True
        self.refundType = bref.refundType if not pd.isna(bref.refundType) else None
        self.refundDate = bref.refundDate.date() if not pd.isna(bref.refundDate) else None
        self.refundAnnounceDate = bref.refundAnnounceDate.date() if not pd.isna(bref.refundAnnounceDate) else None
        self.refundPrice = bref.refundPrice if not pd.isna(bref.refundPrice) else None
        self.effectiveMaturityDate = self.refundDate if ((not self.refundPrice is None) and (self.refundDate>datetime.now().date())) else self.maturityDate
        self.effectiveMaturityPrice = 100.0 if self.refundPrice is None else self.refundPrice
        self.nextCallDate = bref.nextCallDate.date() if not pd.isna(bref.nextCallDate) else None
        self.nextCallPrice = bref.nextCallPrice if not pd.isna(bref.nextCallPrice) else None
        self.parCallDate = bref.parCallDate.date() if not pd.isna(bref.parCallDate) else None
        self.parCallPrice = bref.parCallPrice if not pd.isna(bref.parCallPrice) else None
        self.putType = bref.putType if not pd.isna(bref.putType) else None
        self.nextPutDate = bref.nextPutDate.date() if not pd.isna(bref.nextPutDate) else None
        self.nextPutPrice = bref.nextPutPrice if not pd.isna(bref.nextPutPrice) else None
        # Note: probably should refactor to generate workouts based on settle date instead of datetime.now
        self.workouts = [{'date':self.effectiveMaturityDate,'price':self.effectiveMaturityPrice,'type':'EFFMAT'}]
        
        # Removed self.refundType is None and now that needs to be inspected separately below to do inclusive refund types instead of excluding all
        # Progressive notes on refund types: 
        # 1. Called - should use eff mat = refund date and ignore call data to allow for refundings to dates that aren't coupon dates e.g. 010831CC7
        # 2. Called by muni-forward - e.g. 592481HZ1 (note bbg appears to handle this bond wrong, working out to maturity on discount price input)
        if not self.refundType in specialRefundTypes:
            if ((self.callable) and (self.nextCallDate is not None) and (self.nextCallDate>datetime.now().date())):
                self.workouts.append({'date':self.nextCallDate,'price':self.nextCallPrice,'type':'NEXTCALL'})
            if ((self.callable) and (self.parCallDate is not None) and (self.parCallDate>datetime.now().date())):
                self.workouts.append({'date':self.parCallDate,'price':self.parCallPrice,'type':'PARCALL'})
            if ((self.putType is not None) and (self.nextPutDate is not None) and (self.nextPutDate>datetime.now().date())):
                self.workouts.append({'date':self.nextPutDate,'price':self.nextPutPrice,'type':'PUT'})
        self.currentCalcSettleDate = None
        self.calcPrevCouponDate = None
        self.calcNextCouponDate = None
        self.cashflows = dict.fromkeys(['coupon','principal'])
        self.cashflows['coupon'] = []
        
        self._cashflow_cache = {}

        init_end = time.time()
        init_elapsed = (init_end - init_start) * 1_000 # convert to ms
        key = f"{self.cusip}_init"
        _timing_data[key] = _timing_data.get(key, 0) + init_elapsed

    def numCouponPeriods(self,startDate,endDate):
        return msrbDayCount(startDate, endDate)/(self.daysInYear/self.intFreq)

    def generate_cashflows(self,settleDate,workout):
        cache_key = (settleDate, workout['date'], workout.get('price', None))
        
        if cache_key in self._cashflow_cache:
            # Restore cached data
            cached = self._cashflow_cache[cache_key]
            self.currentCalcSettleDate = cached['settleDate']
            self.cashflows = cached['cashflows']
            self.calcPrevCouponDate = cached['prevCoupon']
            self.calcNextCouponDate = cached['nextCoupon']
            return
        
        self.currentCalcSettleDate = settleDate
        self.cashflows = dict.fromkeys(['coupon','principal'])
        self.cashflows['coupon'] = []
        self.cashflows['principal'] = {'date':workout['date'],'amount':workout['price'],'type':'principal'}
        if (not self.intFreqDesc == 'Interest at maturity'):
            current = workout['date']
            while (current > settleDate): 
                self.cashflows['coupon'].append({'date':current,'amount':100.0*self.coupon/self.intFreq,'type':'coupon'})
                current = current + relativedelta(months=-self.intPeriodMonths)
            #if ((self.penultCouponDate is not None) and (not self.penultCouponDate in [cf['date'] for cf in self.cashflows])):
            #    self.cashflows.append({'date':self.penultCouponDate,'amount':100.0*self.coupon/self.intFreq,'type':'coupon'})
            self.calcPrevCouponDate = current
            self.calcNextCouponDate = min([cf['date'] for cf in self.cashflows['coupon'] if cf['date']>settleDate])
        self.cashflows['coupon'].sort(key=lambda x: x['date'])
        
        # store the cached data
        self._cashflow_cache[cache_key] = {
            'settleDate': self.currentCalcSettleDate,
            'cashflows': self.cashflows.copy(),
            'prevCoupon': self.calcPrevCouponDate,
            'nextCoupon': self.calcNextCouponDate
        }

    # region Example cashflows structure
    # Example cashflows for a bond with 5% coupon, semiannual payments, maturing on 2027-05-15, and settle date of 2025-03-01:
    # {
    #     'coupon': [
    #         {'date': datetime.date(2025, 5, 15), 'amount': 2.5, 'type': 'coupon'},   # 5%/2 = 2.5%
    #         {'date': datetime.date(2025, 11, 15), 'amount': 2.5, 'type': 'coupon'},
    #         {'date': datetime.date(2026, 5, 15), 'amount': 2.5, 'type': 'coupon'},
    #         {'date': datetime.date(2026, 11, 15), 'amount': 2.5, 'type': 'coupon'},
    #         {'date': datetime.date(2027, 5, 15), 'amount': 2.5, 'type': 'coupon'},
    #     ],
    #     'principal': {
    #         'date': datetime.date(2027, 5, 15),  # Maturity date
    #         'amount': 100.0,                     # Face value repayment
    #         'type': 'principal'
    #     }
    # }
    # endregion

    def coupon_count(self):
        return len(self.cashflows['coupon'])

    def bond_price_atmaturity(self,y,settleDate):
        A = msrbDayCount(self.accrualDate,settleDate)
        B = self.daysInYear
        DIR = msrbDayCount(self.accrualDate,self.effectiveMaturityDate)
        R = self.coupon
        RV = 1.0
        P = ((RV+(DIR*R/B))/(1+((DIR-A)/B)*y))-(A*R/B)
        return msrbRoundPrice(100.0*P)
    
    def bond_price_periodic(self,y,workout,settleDate):
        gen_cf_start = time.time()
        self.generate_cashflows(settleDate, workout)
        gen_cf_end = time.time()
        gen_cf_elapsed = (gen_cf_end - gen_cf_start) * 1_000 # convert to ms
        key = f"{self.cusip}_generate_cashflows"
        _timing_data[key] = _timing_data.get(key, 0) + gen_cf_elapsed

        A = msrbDayCount(self.calcPrevCouponDate,settleDate)
        B = self.daysInYear
        E = msrbDayCount(self.calcPrevCouponDate,self.calcNextCouponDate)
        M = self.intFreq
        N = self.coupon_count()
        R = self.coupon
        RV = workout['price']
        if N<=1:
            P = ((RV/100.0)+(R/M))/(1+((E-A)/E)*(y/M)) - (R*A/B)
            P = P*100.0
        else:
            PV = 0.0
            for k in range(1,N+1):
                df = discountFactor(y, M, k-1+((E-A)/E))
                PV = PV + df*self.cashflows['coupon'][k-1]['amount']
            P = RV*discountFactor(y, M, N-1+((E-A)/E)) + PV - (100.0*R*A/B)
        return msrbRoundPrice(P)

    def price(self,y,settleDate=None):
        prc = 0.0
        if settleDate is None:
            if self.currentCalcSettleDate is None:
                settleDate = bmaCalendar.getNthBusinessDay(datetime.now().date(),bmaCalendar.regSettleDays)
            else: 
                settleDate = self.currentCalcSettleDate
        if self.intFreqDesc == 'Interest at maturity':
            prc = self.bond_price_atmaturity(y,settleDate)
        else:
            prc = min([self.bond_price_periodic(y,w,settleDate) for w in self.workouts])
        return prc
    
    def yieldWorkout(self,p,workout,settleDate=None,overrideGuess=None):
        yld = 0.0
        defaultGuess = self.coupon - 0.0001 if p>100 else self.coupon + 0.0001
        guess = overrideGuess if overrideGuess is not None else defaultGuess
        if settleDate is None:
            if self.currentCalcSettleDate is None:
                settleDate = bmaCalendar.getNthBusinessDay(datetime.now().date(),bmaCalendar.regSettleDays) 
                # regSettleDate is the next business day plus the regular settlement days
            else: 
                settleDate = self.currentCalcSettleDate
            
        newton_start = time.time()
        if p>100:
            yld = newton(bond_price_root,guess,tol=0.0001,maxiter=100,args=(p,self,settleDate,))
        else: 
            yld = newton(bond_price_root,guess,tol=0.0001,maxiter=100,args=(p,self,settleDate,))
        
        newton_end = time.time()
        newton_elapsed = (newton_end - newton_start) * 1_000 # convert to ms

        key = f"{self.cusip}_newton_solver"
        _timing_data[key] = _timing_data.get(key, 0) + newton_elapsed

        return msrbRoundYield(100*yld)

    def ytw(self,p,settleDate=None,overrideGuess=None):
        return min([self.yieldWorkout(p,w,settleDate,overrideGuess) for w in self.workouts])

# DV01 measures how much the bond price will increase if the market interest rate decreases by 1 basis point.
    def dv01_px(self,p,settleDate=None):
        yld = self.ytw(p,settleDate)
        blipYld = yld - 0.01
        blipPx = self.price(blipYld/100.0,settleDate)
        dv01 = blipPx - p
        return dv01
# empty function placeholder
    def dv01_yld(self,y,settleDate=None):
        return

def bond_price_root(y,p,bond,settleDate):
    return bond.price(y,settleDate)-p

def clear_timing_data():
    global _timing_data
    _timing_data.clear()
    
def get_cusip_timing(cusip):
    init_key = f"{cusip}_init"
    cf_key = f"{cusip}_generate_cashflows"
    newton_key = f"{cusip}_newton_solver"
    return {'init_ms': _timing_data.get(init_key, 0), 
            'generate_cashflows_ms': _timing_data.get(cf_key, 0),
            'newton_solver_ms': _timing_data.get(newton_key, 0)}

def getIntFreq(ifd):
    mapper = {
        'Semiannually' : 2.0,
        'Monthly' : 12.0,
        'Annually': 1.0,
        'Interest at maturity' : None,
        'Daily' : 360.0,
        'Quarterly' : 4.0,
        'Weekly' : 52.0
        }
    return mapper.get(ifd,2.0)

def discountFactor(y,m,t):
    return 1.0/(1+y/m)**t

def msrbDayCount(startDate,endDate):
    D1 = 30 if (startDate.day==31) else startDate.day
    D2 = 30 if ((endDate.day==31) and ((D1==30) or (D1==31))) else endDate.day
    M1 = startDate.month
    M2 = endDate.month
    Y1 = startDate.year
    Y2 = endDate.year    
    days = (Y2 - Y1)*360 + (M2-M1)*30 + (D2-D1)
    return days

def traderYield(yld):
    trunc = str(floor(yld*1000.0))
    rounder = int(trunc[-1])
    if rounder>=5:
        result = str(int(trunc[:-1])+1)
    else:
        result = str(int(trunc[:-1]))
    return float(result)/100.0

def msrbRoundYield(yld):
    trunc = str(floor(yld*10000.0))
    if int(trunc[-1])>=5:
        result = (int(trunc[:-1])+1)/1000.0
    else:
        result = int(trunc[:-1])/1000.0
    return result

def msrbRoundPrice(prc):
    rprc = None
    if prc is not None:
        rprc = floor(prc*1000.0)/1000.0
    else:
        pass
    return rprc

def getShortCallBucket(cusip):
    bucket = 'noncallable'
    nowYear = datetime.now().year
    try:
        bondData = secMaster.loc[cusip]
    except KeyError:
        return 'nonmuni'
    if ((not pd.isna(bondData.isCallable)) and (not pd.isna(bondData.nextCallDate))):
        callYear = bondData.nextCallDate.year
        callTenor = callYear - nowYear
        if (callTenor==0):
            bucket = 'imminent'
        elif (callTenor>0) and (callTenor<=3):
            bucket = 'short'
        elif ((callTenor>3) and (callTenor<=5)):
            bucket = 'intermediate'
        else:
            bucket = 'long'
    return bucket

def getCallTenorBucket(cusip):
    bucket = 'noncallable'
    nowYear = datetime.now().year
    try:
        bondData = secMaster.loc[cusip]
    except KeyError:
        return 'nonmuni'
    if ((not pd.isna(bondData.isCallable)) and (not pd.isna(bondData.nextCallDate))):
        callYear = bondData.nextCallDate.year
        callTenor = callYear - nowYear
        if (callTenor==0):
            bucket = 'imminent'
        elif (callTenor>0) and (callTenor<=2):
            bucket = 'short'
        elif ((callTenor>2) and (callTenor<=5)):
            bucket = 'intermediate'
        else:
            bucket = 'long'
    return bucket

def getTenorBucket(cusip,tenorYears=None):
    nowYear = datetime.now().year
    bucket = 'refdataerror'
    useTenor = None
    if tenorYears is not None:
        useTenor = tenorYears
    else:
        try:
            bnd = muniBond(cusip)
            tenorYear = bnd.effectiveMaturityDate.year
            useTenor = tenorYear-nowYear
        except Exception:
            useTenor = None
    if useTenor is not None:
        if (useTenor<=5):
            bucket = 'short'
        elif (useTenor>5) and (useTenor<=9):
            bucket = 'front'
        elif ((useTenor>9) and (useTenor<=19)):
            bucket = 'intermediate'
        else:
            bucket = 'long'
    return bucket

def getNonCallBucket(cusip):
    bucket = 'callable'
    nowYear = datetime.now().year
    try:
        bnd = muniBond(cusip)
    except Exception:
        return 'refdataerror'
    if ((bnd.effectiveMaturityDate is not None) and (not bnd.callable)):
        tenorYear = bnd.effectiveMaturityDate.year
        tenor = tenorYear - nowYear
        if (tenor<=2):
            bucket = 'short'
        elif (tenor>2) and (tenor<=8):
            bucket = 'front'
        elif ((tenor>8) and (tenor<=20)):
            bucket = 'intermediate'
        else:
            bucket = 'long'
    return bucket

def getRatingBucket(cusip):
    try:
        compositeRating = getAlgoCompositeRating(cusip)
        compositeRatingRank = getRatingRank(compositeRating)
        if compositeRatingRank == 1:
            ratingBucket = 'AAA'
        elif compositeRatingRank <= 4:
            ratingBucket = 'AA'
        elif compositeRatingRank <= 7:
            ratingBucket = 'A'
        elif compositeRatingRank <= 10:
            ratingBucket = 'BBB'
        elif compositeRatingRank <= 13:
            ratingBucket = 'BB'
        elif compositeRatingRank <= 16:
            ratingBucket = 'B'
        elif compositeRatingRank <= 19:
            ratingBucket = 'CCC'
        elif compositeRatingRank <= 20:
            ratingBucket = 'CC'
        elif compositeRatingRank <= 21:
            ratingBucket = 'C'
        elif compositeRatingRank <= 22:
            ratingBucket = 'D'
        else:
            ratingBucket = 'NR'
    except Exception:
        ratingBucket = 'MISSING'
    return ratingBucket

def getMaturitySizeBucket(cusip):
    bucket = 'micro'
    try:
        bondData = secMaster.loc[cusip]
        mtySize = bondData.maturitySize if not pd.isna(bondData.maturitySize) else 0
        if mtySize <= 1e6:
            bucket = 'micro'
        elif mtySize < 5e6:
            bucket = 'small'
        elif mtySize < 7.5e6:
            bucket = 'medium'
        elif mtySize < 10e6:
            bucket = 'round'
        else:
            bucket = 'large'
    except KeyError:
        bucket = 'micro'
    return bucket

def getCouponBucket(cusip):
    bucket = 'missing'
    try:
        bondData = secMaster.loc[cusip]
        cpn = bondData.coupon
    except KeyError:
        cpn = None
    if (not pd.isna(cpn)) and (cpn is not None):
        if cpn < 4.0:
            bucket = '3'
        elif cpn < 5.0:
            bucket = '4'
        elif cpn < 6.0:
            bucket = '5'
        else:
            bucket = '6'
    else:
        bucket = 'missing'
    return bucket

def getRatingRank(rating):
    ratingRank = 99
    mapper = {'AAA':1,'Aaa':1,
              'AA+':2,'Aa1':2,'AA':3,'Aa2':3,'AA-':4,'Aa3':4,
              'A+':5,'A1':5,'A':6,'A2':6,'A-':7,'A3':7,
              'BBB+':8,'Baa1':8,'BBB':9,'Baa2':9,'BBB-':10,'Baa3':10,
              'BB+':11,'Ba1':11,'BB':12,'Ba2':12,'BB-':13,'Ba3':13,
              'B+':14,'B1':14,'B':15,'B2':15,'B-':16,'B3':16,
              'CCC+':17,'Caa1':17,'CCC':18,'Caa2':18,'CCC-':19,'Caa3':19,
              'CC':20,'Ca':20,
              'C':21,
              'D':22,
              'NR':99
              }
    return mapper.get(rating,99)

def getRankRating(ratingRank):
    rating = 'NR'
    mapper = {1:'AAA',2:'AA+',3:'AA',4:'AA-',5:'A+',6:'A',7:'A-',8:'BBB+',9:'BBB',10:'BBB-',11:'BB+',12:'BB',13:'BB-',14:'B+',15:'B',16:'B-',17:'CCC+',18:'CCC',19:'CCC-',20:'CC',21:'C',22:'D',99:'NR'}
    return mapper.get(ratingRank,'NR')

def getAlgoCompositeRating(cusip):
    try:
        bondData = secMaster.loc[cusip]
        if not pd.isna(bondData.spUnderlyingRating):
            spRating = bondData.spUnderlyingRating
        elif not pd.isna(bondData.spIssuerRating):
            spRating = bondData.spIssuerRating
        else: 
            spRating = bondData.spLongRating if not pd.isna(bondData.spLongRating) else 'NR'
        if not pd.isna(bondData.mdyIssuerLongRating):
            mdyRating = bondData.mdyIssuerLongRating
        else:
            mdyRating = bondData.mdyLongRating if not pd.isna(bondData.mdyLongRating) else 'NR'
        if (not spRating == 'NR') and (not mdyRating == 'NR'):
            compositeRatingRank = max(getRatingRank(spRating),getRatingRank(mdyRating))
        elif (not spRating == 'NR'):
            compositeRatingRank = getRatingRank(spRating)
        elif (not mdyRating == 'NR'):
            compositeRatingRank = getRatingRank(mdyRating)
        else:
            compositeRatingRank = 99
        compositeRating = getRankRating(compositeRatingRank)
    except KeyError:
        compositeRating = 'NR'
    return compositeRating

def getInsurance(cusip):
    insurance = 'uninsured'
    try:
        bondData = secMaster.loc[cusip]
        if (not pd.isna(bondData.bondInsurance)):
            if not bondData.bondInsurance == 'NONE':
                insurance = 'insured'
    except KeyError:
        pass
    return insurance

def getPurposeClass(cusip):
    purposeClass = 'none'
    try:
        bondData = secMaster.loc[cusip]
        if (not pd.isna(bondData.purposeClass)):
            purposeClass = bondData.purposeClass
    except KeyError:
        pass
    return purposeClass

def getAssetClaim(cusip):
    assetClaim = 'none'
    try:
        bondData = secMaster.loc[cusip]
        if (not pd.isna(bondData.assetClaim)):
            assetClaim = bondData.assetClaim
    except KeyError:
        pass
    return assetClaim

def getDeminimisCutoff(refdata):
    cutoff = None
    try:
        sd = bmaCalendar.regSettleDate
        ed = refdata.maturityDate.date() if (not pd.isna(refdata.maturityDate)) and (refdata.maturityDate is not None) else None
        if ed is not None:
            cutoff = 100.0 - 0.25*(((ed-sd).days)//365)
    except Exception:
        pass
    return cutoff

def getAlgoRefDataDict(cusip):
    nowYear = datetime.now().year
    nowMonth = datetime.now().month
    rdct = dict.fromkeys(['cusip','hasRefData','issuerName','obligor','stateCode','ratingBucket','spWatch','mdyWatch','callTenor','sinkTenor','tenor','issueTenorMonths','mtyMonth','principalFactor','couponType','assetStatus','isDefaulted','isBQ','fedTaxStatus','stateTaxStatus','useOfProceeds','purposeClass','assetClaim','maturitySize','coupon','putType','refundType','xoRedemption','mandRedemption','optRedemption','minPiece','minIncrement','firstSettleDate','makeWholeCall','interestFrequency','deminimisPx'])
    rdct['cusip'] = cusip
    try:
        bondData = secMaster.loc[cusip]
        rdct['hasRefData'] = 'Y'
        rdct['issuerName'] = bondData.issuerName if not pd.isna(bondData.issuerName) else None
        rdct['obligor'] = bondData.obligor if not pd.isna(bondData.obligor) else None
        rdct['stateCode'] = bondData.stateCode if not pd.isna(bondData.stateCode) else None
        rdct['compositeRating'] = getAlgoCompositeRating(cusip)
        rdct['callTenor'] = bondData.nextCallDate.year - nowYear if not pd.isna(bondData.nextCallDate) else None
        rdct['issueTenorMonths'] = 12*(nowYear - bondData.issueDate.year)+(nowMonth - bondData.issueDate.month) if not pd.isna(bondData.issueDate) else 0
        rdct['sinkTenor'] = bondData.nextSinkDate.year - nowYear if not pd.isna(bondData.nextSinkDate) else None
        if not pd.isna(bondData.refundDate):
            rdct['tenor'] = bondData.refundDate.year - nowYear
            rdct['mtyMonth'] = bondData.refundDate.month
        elif not pd.isna(bondData.maturityDate):
            rdct['tenor'] = bondData.maturityDate.year - nowYear
            rdct['mtyMonth'] = bondData.maturityDate.month
        else:
            rdct['tenor'] = None
        rdct['spWatch'] = bondData.spLongOutlook if not pd.isna(bondData.spLongOutlook) else None
        rdct['mdyWatch'] = bondData.mdyLongOutlook if not pd.isna(bondData.mdyLongOutlook) else None
        rdct['principalFactor'] = bondData.principalFactor if not pd.isna(bondData.principalFactor) else None
        rdct['couponType'] = bondData.couponType if not pd.isna(bondData.couponType) else None
        rdct['assetStatus'] = bondData.assetStatus if not pd.isna(bondData.assetStatus) else None
        if pd.isna(bondData.isDefaulted):
            rdct['isDefaulted'] = 'yes'
        elif bondData.isDefaulted:
            rdct['isDefaulted'] = 'yes'
        else:
            rdct['isDefaulted'] = 'no'
        if pd.isna(bondData.isBQ):
            rdct['isBQ'] = 'missing'
        elif bondData.isBQ:
            rdct['isBQ'] = 'yes'
        else:
            rdct['isBQ'] = 'no'
        rdct['fedTaxStatus'] = bondData.fedTaxStatus if not pd.isna(bondData.fedTaxStatus) else None
        rdct['stateTaxStatus'] = bondData.stateTaxStatus if not pd.isna(bondData.stateTaxStatus) else None
        rdct['useOfProceeds'] = bondData.useOfProceeds if not pd.isna(bondData.useOfProceeds) else None
        rdct['purposeClass'] = bondData.purposeClass if not pd.isna(bondData.purposeClass) else 'none'
        rdct['assetClaim'] = bondData.assetClaim if not pd.isna(bondData.assetClaim) else 'none'
        rdct['maturitySize'] = bondData.maturitySize if not pd.isna(bondData.maturitySize) else 0
        rdct['coupon'] = bondData.coupon if not pd.isna(bondData.coupon) else None
        rdct['putType'] = bondData.putType if not pd.isna(bondData.putType) else None
        rdct['refundType'] = bondData.refundType if not pd.isna(bondData.refundType) else None
        rdct['xoRedemption'] = bondData.extraordinaryRedemption if not pd.isna(bondData.extraordinaryRedemption) else None
        rdct['mandRedemption'] = bondData.mandatoryRedemption if not pd.isna(bondData.mandatoryRedemption) else None
        rdct['optRedemption'] = bondData.optionalRedemption if not pd.isna(bondData.optionalRedemption) else None
        rdct['minPiece'] = bondData.minPiece if not pd.isna(bondData.minPiece) else None
        rdct['minIncrement'] = bondData.minIncrement if not pd.isna(bondData.minIncrement) else None
        rdct['firstSettleDate'] = bondData.firstSettleDate.date() if not pd.isna(bondData.firstSettleDate) else None
        rdct['makeWholeCall'] = bondData.makeWholeCall if not pd.isna(bondData.makeWholeCall) else None
        rdct['interestFrequency'] = bondData.interestFrequency if not pd.isna(bondData.interestFrequency) else None
        rdct['deminimisPx'] = getDeminimisCutoff(bondData)
    except KeyError:
        rdct['hasRefData'] = 'N'
    return rdct

def getAlgoRefDataBuckets(cusip):
    bdct = dict.fromkeys(['ratingBucket','tenorBucket','callBucket','cpnBucket','insurance','maturitySizeBucket'])
    bdct['ratingBucket'] = getRatingBucket(cusip)
    bdct['tenorBucket'] = getTenorBucket(cusip)
    bdct['callBucket'] = getCallTenorBucket(cusip)
    bdct['cpnBucket'] = getCouponBucket(cusip)
    bdct['insurance'] = getInsurance(cusip)
    bdct['maturitySizeBucket'] = getMaturitySizeBucket(cusip)
    return bdct

def getAlgoRefAqEligibility(cusip):
    eligCode = 'ELIGIBLE'
    refdata = getAlgoRefDataDict(cusip)
    buckets = getAlgoRefDataBuckets(cusip)
    if not (refdata['hasRefData']=='Y'):
        return 'MISSING_REF_DATA'
    if refdata['makeWholeCall'] is not None:
        if refdata['makeWholeCall']:
            eligCode = 'MAKE_WHOLE_CALL'
    if refdata['isBQ'] in ['missing','yes']:
        eligCode = 'BANK_QUALIFIED'
    if refdata['interestFrequency'] is None:
        eligCode = 'MISSING_INTEREST_FREQUENCY'
    else:
        if not refdata['interestFrequency'] == 'Semiannually':
            eligCode = 'IRREGULAR_INTEREST_FREQUENCY'
    if (refdata['callTenor'] is None) and (refdata['sinkTenor'] is not None):
        eligCode = 'NONCALLABLE_SINKER'
    if (refdata['callTenor'] is not None) and (refdata['sinkTenor'] is not None):
        if refdata['sinkTenor'] < refdata['callTenor']:
            eligCode = 'NEXT_SINK_BEFORE_CALL'
    if refdata['tenor'] is None:
        eligCode = 'MISSING_MATURITY_DATE'
    if refdata['issueTenorMonths'] <= 3:
        eligCode = 'RECENT_ISSUE_DATE'
    if refdata['principalFactor'] is None:
        eligCode = 'MISSING_PRINCIPAL_FACTOR'
    if not refdata['principalFactor'] == 1.0:
        eligCode = 'PRINCIPAL_FACTOR'
    if (refdata['spWatch'] in ['NEGATIVE','NM']) or (refdata['mdyWatch'] == 'ON'):
        eligCode = 'RATING_NEGATIVE_OUTLOOK'
    if (refdata['callTenor'] is not None):
        if refdata['callTenor']<=1:
            eligCode = 'IMMINENT_CALL'
    if refdata['xoRedemption'] is not None:
        if refdata['xoRedemption']:
            eligCode = 'XO_REDEMPTION'
    if (refdata['mandRedemption'] is not None) and (not refdata['mandRedemption']=='No'):
        eligCode = 'MANDATORY_REDEMPTION'
    if (refdata['optRedemption'] is not None) and (not refdata['optRedemption']=='No'):
        eligCode = 'OPTIONAL_REDEMPTION'
    if (refdata['tenor'] is not None):
        if refdata['tenor']<=1:
            eligCode = 'IMMINENT_REDEMPTION'
    if not refdata['couponType'] in ['Fixed rate','Original issue discount']:
        eligCode = 'NON_FIXED_COUPON'
    if not refdata['assetStatus'] == 'Live':
        eligCode = 'DEAD_ASSET'
    if not refdata['isDefaulted'] == 'no':
        eligCode = 'DEFAULTED_BOND'
    if (not refdata['fedTaxStatus'] == 'Tax-exempt'):
        eligCode = 'TAX_STATUS'
    if refdata['stateTaxStatus'] == 'Taxable':
        eligCode = 'STATE_TAX_STATUS'
    if refdata['useOfProceeds'] is None:
        eligCode = 'MISSING_SECTOR'
    if refdata['useOfProceeds'] in ["Gas","Hospitals","Charter school","Single-family housing","Single/multi-family housing","Multi-family housing","Senior housing independent living","Nursing homes","Lifecare/retirement centers"]:
        eligCode = 'SECTOR_DISABLED'
    if refdata['coupon'] is None:
        eligCode = 'MISSING_COUPON'
    elif ((refdata['coupon']<4.0) or (refdata['coupon']>5.9)):
        eligCode = 'OUTSIDE_COUPON_RANGE'
    if refdata['putType'] is not None:
        eligCode = 'PUTABLE'
    if refdata['refundType'] is not None:
        if not refdata['refundType']=='Pre-refunded':
            eligCode = 'NON_PRERE_REFUNDING'
    if (not buckets['ratingBucket'] in ['AAA','AA','A']):
        eligCode = 'BELOW_RATING_CUTOFF'
    if (not refdata['minPiece']==5000) or (not refdata['minIncrement']==5000):
        eligCode = 'NONSTANDARD_MIN_PIECE_OR_INCREMENT'
    if cusip in blockedCusips:
        eligCode = 'BLOCKED_CUSIP'
    if any([x in refdata['issuerName'] for x in blockedIssuerTokens]):
        eligCode = 'BLOCKED_ISSUER'
    if refdata['obligor'] is not None:
        if any([x in refdata['obligor'] for x in blockedObligorTokens]):
            eligCode = 'BLOCKED OBLIGOR'
    if refdata['stateCode'] in blockedStates:
        eligCode = 'BLOCKED_STATE'
    return eligCode

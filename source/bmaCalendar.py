#!/usr/bin/python3

from datetime import datetime
import pandas_market_calendars as mcal

todayDate = datetime.now().date()
startDate = datetime(1900,1,1)
endDate = datetime(2200,12,31)

bma = mcal.get_calendar('SIFMA_US')
bmaBusinessDays = [x.date() for x in bma.valid_days(startDate,endDate)]

forceHolidays = [datetime(2025,4,18).date()]

for h in forceHolidays:
    if h in bmaBusinessDays:
        bmaBusinessDays.remove(h)

regSettleDays = 1

def isBusinessDay(date):
    return date in bmaBusinessDays

def getNthBusinessDay(date,offset):
    if date in bmaBusinessDays:
        idx = bmaBusinessDays.index(date)
    else:
        useDate = max([d for d in bmaBusinessDays if d<date])
        idx = bmaBusinessDays.index(useDate)
    bd = bmaBusinessDays[idx+offset]
    return bd 

def getBusinessDaysWithin(sd,ed):
    return [d for d in bmaBusinessDays if ((d>=sd) and (d<=ed))]

regSettleDate = getNthBusinessDay(datetime.now().date(),regSettleDays)
regSettleDateStr = regSettleDate.strftime('%Y%m%d')
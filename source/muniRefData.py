#!/usr/bin/python3

import pandas as pd 

def getRefDataPath():
    return '//fmsops/users/Trading/ALGO/'

def getSecMasterConfig():
    return pd.read_csv(getRefDataPath() + 'securityMasterConfig.csv')

def getRawMasterConfig():
    cfg = getSecMasterConfig()
    return cfg.loc[cfg['enrichedField']=='N']

def getEnrichedMasterConfig():
    return getSecMasterConfig()    

def getRawMasterFieldList():
    cfg = getRawMasterConfig()
    return list(cfg['fieldName'])

def getRawMasterDateFields():
    cfg = getRawMasterConfig()
    return list(cfg[cfg['dateField']=='Y']['fieldName'])

def getEnrichedMasterDateFields():
    cfg = getEnrichedMasterConfig()
    return list(cfg[cfg['dateField']=='Y']['fieldName'])

def getFieldDataType(charType,boolField):
    dtype = pd.StringDtype()
    if charType == 'F':
        dtype = pd.Float64Dtype()
    elif charType == 'J':
        dtype = pd.Int64Dtype()
    elif boolField == 'Y':
        dtype = pd.BooleanDtype()
    return dtype

def getRawMasterDataTypes():
    dtypes = {}
    cfg = getRawMasterConfig()
    for index,row in cfg.iterrows():
        dtypes[row.fieldName] = getFieldDataType(row.qTypeChar,row.boolField)
    return dtypes

def getEnrichedMasterDataTypes():
    dtypes = {}
    cfg = getEnrichedMasterConfig()
    for index,row in cfg.iterrows():
        dtypes[row.fieldName] = getFieldDataType(row.qTypeChar,row.boolField)
    return dtypes

def getRawMasterPath():
    return getRefDataPath() + 'muniSecurityMasterCurrent.csv'

def getEnrichedMasterPath():
    return  getRefDataPath() + 'muniSecMaster.csv'

def getRawMasterHeader():
    cfg = getRawMasterConfig()
    fields = list(cfg['fieldName'])
    delim = '|'
    header = delim.join([f for f in fields])
    return header + '\n'

def getRawMasterRowDict():
    cfg = getRawMasterConfig()
    fields = list(cfg['fieldName'])
    return dict.fromkeys(fields)

# Helper functions
def loadMaster(inputFilePath):
    refMaster = pd.read_csv(inputFilePath,sep = '|',index_col = 'cusip',
                            dtype = getRawMasterDataTypes())
    refMaster[getRawMasterDateFields()] = refMaster[getRawMasterDateFields()].apply(pd.to_datetime,format='%Y-%m-%d',errors='coerce')
    refMaster = refMaster[refMaster.index.notnull()]
    return refMaster

def loadEnrichedMaster(inputFilePath):
    refMaster = pd.read_csv(inputFilePath,sep = '|',index_col = 'cusip',
                            dtype = getEnrichedMasterDataTypes())
    refMaster[getEnrichedMasterDateFields()] = refMaster[getEnrichedMasterDateFields()].apply(pd.to_datetime,format='%Y-%m-%d',errors='coerce')
    refMaster = refMaster[refMaster.index.notnull()]
    return refMaster


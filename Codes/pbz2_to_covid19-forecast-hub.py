import pandas as pd
import io
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.stats.proportion as smp
import pylab
from scipy.optimize import curve_fit
import niddk_covid_sicr as ncs
import requests
import bz2
import pickle
import _pickle as cPickle

# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data 
    
def preparedf(USNwkaheadinccase,forecasttype,forecastdate):
    USNwkaheadinccaseOUT = pd.melt(USNwkaheadinccase.reset_index(level=0),id_vars="index",var_name='target_end_date',value_name="value")
    USNwkaheadinccaseOUT['location']='US'
    USNwkaheadinccaseOUT['type']='quantile'
    USNwkaheadinccaseOUT['forecast_date']=forecastdate
    x=USNwkaheadinccaseOUT['target_end_date']-pd.to_datetime('today')    
    USNwkaheadinccaseOUT['target']= (np.ceil(x / np.timedelta64(1, 'W')).astype(int)).apply(str) +' Wk ahead ' + forecasttype
    USNwkaheadinccaseOUT.rename(columns = {'index':'quantile'}, inplace = True)
    USNwkaheadinccaseOUT = USNwkaheadinccaseOUT.reindex(columns=['location','target','type','quantile','forecast_date','target_end_date','value'])
    return USNwkaheadinccaseOUT
    
try:
    del USNwkaheadinccase
    del USNwkaheadincdeath
    del USNwkaheadcumdeath
except:
    print('files do not exist')

forecastdate='2020-09-01'

projloc = 'C:\\Users\\osman\\Documents\\fitUS\\'
fitloc = 'C:\\Users\\osman\\Documents\\US06-27-20\\fitUS06-27-20'
rois = ncs.list_rois(fitloc, 'SICRMQC2R2DX2', '.pkl')
df_roi_t0 = pd.read_csv('n_data.csv')
qus = np.r_[(0.01, 0.025, np.arange(0.05,1,0.05).tolist(), 0.975, 0.99)]

for current_roi in rois:  
    #current_roi = 'US_MD'
    df_withprojonly = decompress_pickle(projloc+current_roi+'.pbz2')
    numprojections = int(len(df_withprojonly.columns)/3)
    numweekstoreport = 31

    t0extracted = df_roi_t0[df_roi_t0['roi']==current_roi].t0.values
    newcolswithdates = pd.date_range(start=t0extracted[0], periods=numprojections, closed='left')
    df_withprojonly.columns = np.tile(newcolswithdates,3)

    PROTOCases = df_withprojonly.iloc[:, :numprojections ]
    PROTODeaths = df_withprojonly.iloc[:, -numprojections:]

    Cases = PROTOCases.transpose().resample('W-SAT').sum()
    Deaths = PROTODeaths.transpose().resample('W-SAT').sum()
    CumDeaths = Deaths.cumsum(axis=0)

    Nwkaheadinccase  = Cases.quantile(qus, axis=1).iloc[:,-numweekstoreport:]
    Nwkaheadincdeath = Deaths.quantile(qus, axis=1).iloc[:,-numweekstoreport:]
    Nwkaheadcumdeath = CumDeaths.quantile(qus, axis=1).iloc[:,-numweekstoreport:]
    
    #print(Nwkaheadinccase.iloc[:,-1].name)
    #print(Nwkaheadincdeath.iloc[:,-1].name)
    #print(Nwkaheadcumdeath.iloc[:,-1].name)
    
    if 'USNwkaheadinccase' in locals():
        USNwkaheadinccase  = USNwkaheadinccase.add(Nwkaheadinccase)
        USNwkaheadincdeath = USNwkaheadincdeath.add(Nwkaheadincdeath)
        USNwkaheadcumdeath = USNwkaheadcumdeath.add(Nwkaheadcumdeath)
    else:
        USNwkaheadinccase = Nwkaheadinccase.copy()
        USNwkaheadincdeath = Nwkaheadincdeath.copy()
        USNwkaheadcumdeath = Nwkaheadcumdeath.copy()

    #compressed_pickle(current_roi,df_withprojonly)
    print(current_roi,numprojections)
    USNwkaheadinccaseOUT = preparedf(USNwkaheadinccase,'inc case',forecastdate)
    USNwkaheadincdeathOUT = preparedf(USNwkaheadincdeath,'inc death',forecastdate)
    USNwkaheadcumdeathOUT = preparedf(USNwkaheadcumdeath,'cum death',forecastdate)

    #USNwkaheadinccase.to_csv('NIH-FDA_US_inccase.csv')
    #USNwkaheadincdeath.to_csv('NIH-FDA_US_incdeath.csv')
    #USNwkaheadcumdeath.to_csv('NIH-FDA_US_cumdeath.csv')
    #USNwkaheadinccaseOUT.to_csv('NIH-FDA_US_inccase.csv', index=False)
    #USNwkaheadincdeathOUT.to_csv('NIH-FDA_US_incdeath.csv', index=False)
    #USNwkaheadcumdeathOUT.to_csv('NIH-FDA_US_cumdeath.csv', index=False)
    USNIHFDAproj = pd.concat([USNwkaheadinccaseOUT, USNwkaheadincdeathOUT, USNwkaheadcumdeathOUT])
    USNIHFDAproj.to_csv(forecastdate+'-NIHandFDA-SICR.csv', index=False)    

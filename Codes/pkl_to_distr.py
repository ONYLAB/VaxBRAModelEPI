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

fitloc = 'C:\\Users\\osman\\Documents\\fitUS06-27-20'
rois = ncs.list_rois(fitloc, 'SICRMQC2R2DX2', '.pkl')
df_roi_t0 = pd.read_csv('n_data.csv')

for current_roi in rois:  
    df = ncs.extract_samples(fitloc, '../models/', 'SICRMQC2R2DX2', current_roi, 1)
    df_withprojonly = df[['chain','draw','warmup','f1','f2','sigmar','sigmad','sigmar1','sigmad1','sigmau','mbase','mlocation','trelax','extra_std','extra_std_R','extra_std_D','q','cbase','clocation','n_pop','phi[1]','phi[2]','phi[3]','u_init[1]','u_init[2]','u_init[3]','u_init[4]','u_init[5]','u_init[6]','u_init[7]','u_init[8]','sigmac','beta','sigma','R0']]
    df_withprojonly.to_csv(current_roi+'.csv')

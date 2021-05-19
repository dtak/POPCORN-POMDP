#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to pull vasopressors out of raw mimic data cut for use in eg 
  downstream RL modeling or to predict intervention onsets.

As for fluids, only bothering to use MV data bc of issues in timing of fluids for CV

@author: josephfutoma
"""

import sys
import numpy as np
import pickle
import pandas as pd
np.set_printoptions(threshold=1000)

PATH_TO_REPO = "XXX"


#####
##### MV input events
#####

inputdat_mv = pd.read_csv(PATH_TO_REPO+"raw-data/INPUTEVENTS_MV.csv",
	usecols=['ICUSTAY_ID','STARTTIME','ENDTIME','ITEMID','AMOUNT',
	'AMOUNTUOM','RATE','RATEUOM','STATUSDESCRIPTION'])

inputdat_mv = inputdat_mv.dropna(axis=0,how='any',
	subset=['ICUSTAY_ID','STARTTIME','ITEMID','AMOUNT','RATE'])

inputdat_mv = inputdat_mv.astype(
	{'ICUSTAY_ID':int,
		'STARTTIME':str,
		'ENDTIME':str,	
		'ITEMID':int,
		'AMOUNT':float,
		'AMOUNTUOM':str,
		'STATUSDESCRIPTION':str,
		'RATE':float,
		'RATEUOM':str
	})

inputdat_mv['STARTTIME'] = pd.to_datetime(inputdat_mv['STARTTIME'])
inputdat_mv['ENDTIME'] = pd.to_datetime(inputdat_mv['ENDTIME'])

inputdat_mv = inputdat_mv[inputdat_mv['AMOUNT']>0]
inputdat_mv = inputdat_mv[inputdat_mv['RATE']>0]
inputdat_mv = inputdat_mv[inputdat_mv['STATUSDESCRIPTION']!='Rewritten']

###
### NOTE: per Leo & Ryan, only use Dopamine, Epinephrine, Norepinephrine, Vasopressin, Phenylephrine
###    For other applications, you may want other drugs as well -> check with a clinician!
###

##### VASOPRESSORS; total 91388
vaso_item_ids = [
# 221653, #Dobutamine; 1512,  CUT??
221662, #Dopamine; 8470, 
221289, #Epinephrine; 4625, 
# 221986, #Milrinone; 3525,  CUT??
221906, #Norepinephrine; 69114 times, 
222315, #Vasopressin; 4142, 
221749 #Phenylephrine; 72892
]

vaso_names = [ 
# 'Dobutamine',
'Dopamine',
'Epinephrine',
# 'Milrinone',
'Norepinephrine',
'Vasopressin',
'Phenylephrine'
]

inputdat_mv = inputdat_mv[inputdat_mv['ITEMID'].isin(vaso_item_ids)]
inputdat_mv = inputdat_mv.sort_values(by=["ICUSTAY_ID","STARTTIME"])
inputdat_mv['EVENT_TIME'] = (inputdat_mv['ENDTIME']-inputdat_mv['STARTTIME']).astype('timedelta64[m]').astype(int)

inputdat_mv.loc[inputdat_mv['RATEUOM']=='units/hour','RATE'] /= 60
inputdat_mv.loc[inputdat_mv['RATEUOM']=='units/hour','RATEUOM'] = 'units/min'

inputdat_mv['RATE_NORMED_NOREPI'] = inputdat_mv['RATE'] #final dose to use; modify to norepi

### Do dose normalization following Matthieu's query
# https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_Data_extract_MIMIC3_140219.ipynb

#Vasopressin
inputdat_mv.loc[(inputdat_mv['ITEMID']==222315) & (inputdat_mv['RATE_NORMED_NOREPI']>.2),'RATE_NORMED_NOREPI'] = 0.2 #filter extremes (Matthieu did...)
inputdat_mv.loc[inputdat_mv['ITEMID']==222315,'RATE_NORMED_NOREPI'] *= 5 

#Phenylephrine
inputdat_mv.loc[inputdat_mv['ITEMID']==221749,'RATE_NORMED_NOREPI'] *= 0.45

#Dopamine
inputdat_mv.loc[inputdat_mv['ITEMID']==221662,'RATE_NORMED_NOREPI'] *= 0.01

#Epinephrine: none (same scale as norepi)
#Norepinephrine: none (baseline)


inputdat_mv.to_csv(PATH_TO_REPO+"query-data/vasopressors_mv.csv",index=False)


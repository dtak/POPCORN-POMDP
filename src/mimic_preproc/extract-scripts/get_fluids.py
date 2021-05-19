#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 4 2018

File to pull fluids out of raw mimic data cut for use in downstream RL modeling.

Takes data from the raw mimic csv's in raw-data:
	path on odyssey:  /n/dtak/mimic-iii-v1-4/raw-data 
    
Most of what we need is in INPUTEVENTS_CV & INPUTEVENTS_MV


@author: josephfutoma
"""

import sys
import numpy as np
import pickle
import pandas as pd
import pickle 
np.set_printoptions(threshold=1000)
pd.set_option("display.max_columns",101)

PATH_TO_REPO = "XXX"

#####
##### MV input events
#####

inputdat_mv = pd.read_csv(PATH_TO_REPO+"raw-data/INPUTEVENTS_MV.csv",
	usecols=['ICUSTAY_ID','STARTTIME','ENDTIME','ITEMID','AMOUNT',
	'AMOUNTUOM','RATE','RATEUOM','STATUSDESCRIPTION'])

inputdat_mv = inputdat_mv.dropna(axis=0,how='any',
	subset=['ICUSTAY_ID','STARTTIME','ITEMID','AMOUNT'])

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

#filters
inputdat_mv = inputdat_mv[inputdat_mv['AMOUNT']>0]
inputdat_mv = inputdat_mv[inputdat_mv['STATUSDESCRIPTION']!='Rewritten'] #admin error, ignore these

###
### NOTE: per Leo's suggestion, just using most common crystalloids along with blood products for bleeds
###

##### CRYSTALLOIDS
mv_crystal_items = {
225158: 'NaCl 0.9%',
225828: 'LR',
# 225944: 'Sterile Water',
# 225797: 'Free Water',
# 225159: 'NaCl 0.45%',
# 225161: 'NaCl 3% (Hypertonic Saline)',
# 225823: 'D5 1/2NS',
# 225825: 'D5NS',
# 225827: 'D5LR',
# 225941: 'D5 1/4NS',
# 226089: 'Piggyback'
}

##### COLLOIDS
mv_coll_items = {
   #  220864: 'Albumin 5%',
   #  220862: 'Albumin 25%',
   #  225174: 'Hetastarch (Hespan) 6%',
   #  225795: 'Dextran 40',
   #  225796: 'Dextran 70',
   #  # -- below ITEMIDs not in use
   # # -- 220861 | Albumin (Human) 20%
   # # -- 220863 | Albumin (Human) 4%
   
   # 220949: 'Dextrose 5%',
   # 220950: 'Dextrose 10%',
   # 220952: 'Dextrose 50%'
}


############ BLOOD PRODUCTS

mv_rbc_items = {
  225168: 'Packed Red Blood Cells',
  226368: 'PACU Packed RBC Intake',
  226370: 'OR Packed RBC Intake',
  227070: 'OR Autologous Blood Intake'
}

mv_ffp_items = {
  220970: 'PACU FFP Intake',
  226367: 'Fresh Frozen Plasma',
  227072: 'OR FFP Intake'
}

mv_platelet_items = {
225170: 'Platelets',
226369: 'OR Platelet Intake'
}

#####

mv_item_ids = list(mv_crystal_items.keys())
mv_item_ids.extend(list(mv_coll_items.keys()))
mv_item_ids.extend(list(mv_rbc_items.keys()))
mv_item_ids.extend(list(mv_ffp_items.keys()))
mv_item_ids.extend(list(mv_platelet_items.keys()))

mv_items = list(mv_crystal_items.values())
mv_items.extend(list(mv_coll_items.values()))
mv_items.extend(list(mv_rbc_items.values()))
mv_items.extend(list(mv_ffp_items.values()))
mv_items.extend(list(mv_platelet_items.values()))


inputdat_mv = inputdat_mv[inputdat_mv['ITEMID'].isin(mv_item_ids)]
inputdat_mv = inputdat_mv.sort_values(by=["ICUSTAY_ID","STARTTIME"])
inputdat_mv.loc[inputdat_mv['AMOUNTUOM']=='L','AMOUNT'] *= 1000
inputdat_mv.loc[inputdat_mv['AMOUNTUOM']=='L','AMOUNTUOM'] = 'ml'

#how long admin was, in mins
inputdat_mv['EVENT_TIME'] = (inputdat_mv['ENDTIME']-inputdat_mv['STARTTIME']).astype('timedelta64[m]').astype(int)

#NOTE: may want to play around with this, as a very large fraction are 1, so more of an "instant" fluids

BOLUS_TIME_THRESH = 60 #time window over which fluid was administered for us to include

### filter out to get final fluids dataset, for now

inputdat_mv = inputdat_mv[inputdat_mv['EVENT_TIME'] <= BOLUS_TIME_THRESH]


inputdat_mv.to_csv(PATH_TO_REPO+"query-data/allfluids_and_bloodproducts_mv.csv",index=False)

#NOTE: we intentionally do NOT filter on volume in this script, so very small fluid amounts are kept
#   it is expected that this will be handled in a downstream data cleaning script that will aggregate
#   fluid amounts and filter out periods where very small amounts (eg < 250ml) are given

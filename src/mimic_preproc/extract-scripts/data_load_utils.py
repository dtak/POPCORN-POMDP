#####
#
# data_load_utils.py
#    Contains some helper funcs to load in the relevant files in query-data
#	 This may be useful for several of the preprocessing scripts to create model-data
#    for the RL-hypotension task or intervention onset tasks
#####

import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
# import matplotlib.pyplot as plt
from time import time
import sys

### helper funcs to convert inds at hourly level to other levels
def convert_1inds_to_kinds(ind_vec,k=8):
    """
    takes in a vector of indicators, ind_vec, denoting whether a measurement was taken in past hour
    returns an aggregate form of this for past k hours, eg was value X taken in past k hours
    """
    res = np.cumsum(np.array(ind_vec))
    res[k:] -= res[:-k]
    return np.array(res>=1,"float")

def convert_1inds_to_everinds(ind_vec):
    """
    takes in a vector of indicators, ind_vec, denoting whether a measurement was taken in past hour
    returns an aggregate form for whether this variable has ever been taken so far
    """
    res = np.cumsum(np.array(ind_vec))
    return np.array(res>=1,"float")

def load_labs_and_vitals(PATH_TO_QUERY_DATA,cohort_dat,
	get_arterial_BPs=False):
	"""
	giant function that takes in & loads all labs & vitals 
	in their current format in csv's within query data.

	verbose: spit out extra info about # zero vals tossed
	"""
		  
	all_ts_dats = {}
	all_ICU_IDs = np.array(cohort_dat['ICUSTAY_ID'])

	ICU_ADM_dat = cohort_dat.loc[:,['ICUSTAY_ID','HADM_ID']]


	###
	### start with vitals, which are all coming from CE
	###
	ts_vars_vitals = [
		'dbp',
		'fio2',
		'GCS',
		'hr',
		'map',
		'sbp',
		'spontaneousrr',
		'spo2',
		'temp',
		'urine',
		'weight'
	]

	#NOTE: fix VALUE & VALUENUM if we're doing any manual edits to data bugs...
	for v in ts_vars_vitals:
		print(v)
		if v=='urine':
			dat = pd.read_csv(PATH_TO_QUERY_DATA+v+".csv",
				usecols=["ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUEUOM"])
		else:
			dat = pd.read_csv(PATH_TO_QUERY_DATA+v+".csv",
				usecols=["ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"])

		#filter out any NAs or 0 values
		dat = dat.dropna(axis=0,how='any',subset=['ICUSTAY_ID','CHARTTIME','VALUE'])
		### EDIT: retain the 0's for urine and RR! these are likely real! 
		# useful to know this, as indicates someone pretty sick
		if v=='urine':
			dat = dat[dat['VALUE']>=0]
		elif v=='spontaneousrr':
			dat = dat[dat['VALUENUM']>=0]
		else:
			dat = dat[dat['VALUENUM']>0]

		#presort for easy access later during discretization
		dat['CHARTTIME'] = pd.to_datetime(dat['CHARTTIME'])
		dat = dat.sort_values(by=["ICUSTAY_ID","CHARTTIME"])

		if get_arterial_BPs and v in ['dbp','map','sbp']: 
			#NOTE: if needed, also have item ids allowing flags for which BPs are arterial (invasive)
			# this may be useful info, as someone who has an arterial line in is more sick...
			
			art_item_ids = [
			220051,225310, #dbp
			220050,225309, #sbp
			220052,225312 #map
			]
			nonart_item_ids = [
			227242,224643,220180, #dbp
			224167,227243,220179, #sbp
			220181 #map
			]
			# split out arterial vs non-arterial
			nonart_dat = dat.loc[np.in1d(dat['ITEMID'],nonart_item_ids),:]
			all_ts_dats[v] = nonart_dat
			art_dat = dat.loc[np.in1d(dat['ITEMID'],art_item_ids),:]
			all_ts_dats['art_'+v] = art_dat
			
			continue

		if v == 'GCS': 
			#split out into 3 separate types; combine after doing carry-forward imputation
			#not all 3 types are necessarily taken at identical time stamps, so this is necessary...s
			eye_item_ids = [227011,226756,220739]
			motor_item_ids = [227012,226757,223901]
			verbal_item_ids = [227014,228112,226758,223900]

			dat_eye = dat.loc[np.in1d(dat['ITEMID'],eye_item_ids),:]
			dat_motor = dat.loc[np.in1d(dat['ITEMID'],motor_item_ids),:]
			dat_verbal = dat.loc[np.in1d(dat['ITEMID'],verbal_item_ids),:]

			all_ts_dats[v+'_eye'] = dat_eye
			all_ts_dats[v+'_motor'] = dat_motor
			all_ts_dats[v+'_verbal'] = dat_verbal

			continue

		#some of these are messed up and a decimal in [0,1] rather than 21-100%; 
		if v=='fio2':
			MV_id = 223835 #other item ids for FiO2 are for CV, and CV had weird units
			dat = dat.loc[dat['ITEMID']==MV_id,:]
			dat.loc[dat['VALUE']<=1,'VALUE'] *= 100 
			dat.loc[np.logical_and(dat['VALUE']>1,dat['VALUE']<=10),'VALUE'] *= 10 #assuming this is right...
			dat.loc[dat['VALUENUM']<=1,'VALUENUM'] *= 100 
			dat.loc[np.logical_and(dat['VALUENUM']>1,dat['VALUENUM']<=10),'VALUENUM'] *= 10 #assuming this is right...

		if v=='spontaneousrr':
			MV_ids = [220210,224690]
			MV_inds = np.in1d(dat['ITEMID'],MV_ids)
			dat = dat.loc[MV_inds,:]
			#filtering to just MV fixes weird text values in VALUE column...

		#convert all F to C 
		if v=='temp':
			F_ids = [223761,678]
			F_inds = np.in1d(dat['ITEMID'],F_ids)
			dat.loc[F_inds,'VALUE'] = (dat.loc[F_inds,'VALUE'] - 32.) / 1.8
			dat.loc[F_inds,'VALUENUM'] = (dat.loc[F_inds,'VALUENUM'] - 32.) / 1.8

		all_ts_dats[v] = dat


	###
	### get all labs next. slightly trickier as need to merge in to get the admission ID associated with each ICU stay
	###
	ts_vars_labs_single_file = [
		'bun',
		'magnesium',
		'platelets',
		'sodium',
		'alt',
		# 'co2', #NOTE: exclude for now, due to considerable overlap with bicarb (this is one of the several itemids for bicarb)
		'hct',
		'po2',
		'ast',
		'potassium',
		'wbc',
		'bicarbonate',
		'creatinine',
		'lactate',
		'pco2',
	]

	for v in ts_vars_labs_single_file:
		print(v)

		dat = pd.read_csv(PATH_TO_QUERY_DATA+v+".csv",
			usecols=['HADM_ID','ITEMID','CHARTTIME','VALUE','VALUENUM','VALUEUOM'])
		dat = dat.dropna(axis=0,how='any',subset=['HADM_ID','CHARTTIME','VALUENUM'])
		
		dat = dat[dat['VALUENUM']>0]
		dat['CHARTTIME'] = pd.to_datetime(dat['CHARTTIME'])
		dat = dat.merge(ICU_ADM_dat,'left','HADM_ID')
		dat = dat.dropna(axis=0,how='any',subset=['ICUSTAY_ID'])
		dat = dat.sort_values(by=["ICUSTAY_ID","CHARTTIME"])

		all_ts_dats[v] = dat

	###
	### these labs exist in both CE & LE so need an extra merge
	###
	ts_vars_multiple_files = [
		'bilirubin_total',
		'glucose',
		'inr',
		'hgb',
	]

	for v in ts_vars_multiple_files:
		print(v)

		#read labevents data in first, then chartevents
		lab_dat = pd.read_csv(PATH_TO_QUERY_DATA+v+"_labs.csv",
			usecols=['HADM_ID','ITEMID','CHARTTIME','VALUE','VALUENUM','VALUEUOM'])
		lab_dat = lab_dat.dropna(axis=0,how='any',subset=['HADM_ID','CHARTTIME','VALUENUM'])
		
		lab_dat = lab_dat[lab_dat['VALUENUM']>0]
		lab_dat = lab_dat.merge(ICU_ADM_dat,'left','HADM_ID')
		lab_dat = lab_dat.dropna(axis=0,how='any',subset=['ICUSTAY_ID'])
		lab_dat['CHARTTIME'] = pd.to_datetime(lab_dat['CHARTTIME'])
		lab_dat = lab_dat.sort_values(by=["ICUSTAY_ID","CHARTTIME"])
		lab_dat = lab_dat.loc[:,['HADM_ID', 'ICUSTAY_ID', 'ITEMID', 'CHARTTIME',
			'VALUE', 'VALUENUM', 'VALUEUOM']]

		ce_dat = pd.read_csv(PATH_TO_QUERY_DATA+v+"_ce.csv",
			usecols=["ICUSTAY_ID","ITEMID","CHARTTIME","VALUE","VALUENUM","VALUEUOM"])

		ce_dat = ce_dat.dropna(axis=0,how='any',subset=['ICUSTAY_ID','CHARTTIME','VALUENUM'])
		
		ce_dat = ce_dat[ce_dat['VALUENUM']>0]
		ce_dat['CHARTTIME'] = pd.to_datetime(ce_dat['CHARTTIME'])
		ce_dat = ce_dat.sort_values(by=["ICUSTAY_ID","CHARTTIME"])

		dat = pd.concat([lab_dat,ce_dat])
		dat = dat.sort_values(by=["ICUSTAY_ID","CHARTTIME"])

		all_ts_dats[v] = dat

	print('time series variables all loaded!')

	#later on we'll impute any missing values with the median as a first quick and dirty method
	VAR_REFERENCE_IMPUTE = {
	'art_dbp': 60,
	'dbp': 60, 
	'alt': 34,
	'art_map': 85, 
	'map': 85, 
	'art_sbp': 120, 
	'sbp': 120,
	'ast': 40,
	'hr': 86,
	'bun': 23,
	'bilirubin_total': 0.9,
	'creatinine': 1,
	'glucose': 120,
	'hct': 30.2, 
	'hgb': 10.2, 
	'fio2': 21,
	'bicarbonate': 25,
	'pco2': 40,
	'po2': 95, 
	'potassium': 4.1,
	'sodium': 140, 
	'lactate': 1, 
	'magnesium': 2,
	'platelets': 208,
	'spontaneousrr': 18,
	'spo2': 95, 
	'wbc': 9, 
	'weight': 82,
	'GCS': 15, 
	'GCS_eye': 4,
	'GCS_verbal': 5,
	'GCS_motor': 6,
	'temp': 37,
	'urine': 80,
	'inr': 1.1
	}

	VARS_CLIP_LOWS_HIGHS = {
	'art_dbp': [30,180] ,
	'dbp': [30,180], 
	'alt': [8,3000],
	'art_map': [40,200], 
	'map': [40,200], 
	'art_sbp': [60,220], 
	'sbp': [60,220],
	'ast': [7,6000],
	'hr': [25,150],
	'bun': [2,150],
	'bilirubin_total': [0.1,30],
	'creatinine': [0.3,15],
	'glucose': [50,500],
	'hct': [18,50], 
	'hgb': [5,16], 
	'fio2': [21,100],
	'bicarbonate': [9,40],
	'pco2': [19,100],
	'po2': [30,500], 
	'potassium': [2,6.5],
	'sodium': [120,160], 
	'lactate': [0.5,21], 
	'magnesium': [1,4],
	'platelets': [25,600],
	'spontaneousrr': [1,40],
	'spo2': [80,100], 
	'wbc': [1,50], 
	'weight': [40,200], 
	'GCS_eye': [1,4],
	'GCS_verbal': [1,5],
	'GCS_motor': [1,6],
	'GCS': [3,15], 
	'temp': [32,40],
	'urine': [0,2500],
	'inr': [0.5,20]
	}

	all_ts_vars = list(all_ts_dats.keys())

	#clip outliers outside reasonable physiologic values...
	for v in all_ts_vars:
		dat = all_ts_dats[v]

		if v=='urine':
			val_str = 'VALUE'
		else:
			val_str = 'VALUENUM'

		vals = np.array(dat[val_str],"float")

		lq = VARS_CLIP_LOWS_HIGHS[v][0]
		uq = VARS_CLIP_LOWS_HIGHS[v][1]

		dat.loc[vals<=lq,val_str] = lq
		dat.loc[vals>=uq,val_str] = uq

	print('time series variables clipped to plausible ranges, outliers removed')

	return all_ts_dats, VAR_REFERENCE_IMPUTE, all_ts_vars




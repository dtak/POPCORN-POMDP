#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

File to build out a cohort using the ADMISSIONS, ICUSTAYS, and PATIENTS tables in raw mimic data.

@author: josephfutoma
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle

from time import time

PATH_TO_REPO = "XXX"

### options for building out cohort.
limit_first_ICU = True # do we limit to first ICU stay if there are several in an admission?
limit_first_admission = False #limit to first admission only?
age_thresh = 18 #min age to be included
LOS_lower_thresh = 12#LOS must be at least this long to be included; exclude very short stays

# load in stuff
mv_icuids_dat = pd.read_csv(PATH_TO_REPO+'query-data/mv_icuids.csv')

#may be many icu stays for one admission
icustays_dat = pd.read_csv(PATH_TO_REPO+'raw-data/ICUSTAYS.csv',
		usecols=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','FIRST_CAREUNIT',
		'INTIME','OUTTIME'])

#may be many admissions for one patient
admissions_dat = pd.read_csv(PATH_TO_REPO+'raw-data/ADMISSIONS.csv',	
	usecols=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DEATHTIME','HOSPITAL_EXPIRE_FLAG',
	'ADMISSION_TYPE','ADMISSION_LOCATION','DISCHARGE_LOCATION',
	'INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY'])

patients_dat = pd.read_csv(PATH_TO_REPO+'raw-data/PATIENTS.csv',
	usecols=['SUBJECT_ID','GENDER','DOB','DOD','DOD_HOSP','DOD_SSN'])

### merge these all together
cohort_dat = patients_dat.merge(admissions_dat,'inner',on='SUBJECT_ID')
cohort_dat = cohort_dat.merge(icustays_dat,'inner',on=['SUBJECT_ID','HADM_ID'])
cohort_dat = mv_icuids_dat.merge(cohort_dat,'left',on='ICUSTAY_ID')

# NOTE: not entirely sure what the difference is between these death dates...
# mimic documentation claims HOSPITAL_EXPIRE_FLAG marks in-hospital mortality, not 
# sure what concordance among these deaths is...
cohort_dat = cohort_dat.loc[:,['ICUSTAY_ID','HADM_ID','SUBJECT_ID',
	'INTIME','OUTTIME','ADMITTIME','DISCHTIME','DOB','DOD','DOD_HOSP',
	'DOD_SSN','DEATHTIME','HOSPITAL_EXPIRE_FLAG','GENDER','ADMISSION_TYPE','ADMISSION_LOCATION',
	'DISCHARGE_LOCATION','INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS',
	'ETHNICITY','FIRST_CAREUNIT']]

cohort_dat['INTIME'] = pd.to_datetime(cohort_dat['INTIME'])
cohort_dat['OUTTIME'] = pd.to_datetime(cohort_dat['OUTTIME'])
cohort_dat['ADMITTIME'] = pd.to_datetime(cohort_dat['ADMITTIME'])
cohort_dat['DISCHTIME'] = pd.to_datetime(cohort_dat['DISCHTIME'])
cohort_dat['DOB'] = pd.to_datetime(cohort_dat['DOB'])
cohort_dat['DOD'] = pd.to_datetime(cohort_dat['DOD'])
cohort_dat['DOD_HOSP'] = pd.to_datetime(cohort_dat['DOD_HOSP'])
cohort_dat['DOD_SSN'] = pd.to_datetime(cohort_dat['DOD_SSN'])
cohort_dat['DEATHTIME'] = pd.to_datetime(cohort_dat['DEATHTIME'])
cohort_dat['LOS'] = (cohort_dat['OUTTIME']-cohort_dat['INTIME']).dt.total_seconds()/60/60 #in hrs
#get age in years...super janky, i will never figure out datetimes...
cohort_dat['AGE'] = [x.total_seconds()/60/60/24/365.2422 for x in (np.array(cohort_dat['ADMITTIME'].dt.date) - np.array(cohort_dat['DOB'].dt.date))] 

cohort_dat = cohort_dat.sort_values(by=["ICUSTAY_ID"]) 

# should be 23,386 ICU stays so far

#get the ICU with earliest INTIME for each HADM_ID
if limit_first_ICU:
	first_icu_stays_dat = cohort_dat.loc[:,['HADM_ID','INTIME']].groupby(['HADM_ID']).min().reset_index()
	cohort_dat = cohort_dat.merge(first_icu_stays_dat,'right',on=['HADM_ID','INTIME'])
	#if doing this, should be at 21,876 now

if limit_first_admission:
	first_adm_dat = cohort_dat.loc[:,['SUBJECT_ID','ADMITTIME']].groupby(['SUBJECT_ID']).min().reset_index()
	cohort_dat = cohort_dat.merge(first_adm_dat,'right',on=['SUBJECT_ID','ADMITTIME'])
	#if doing this and also did first ICU stay, should be at 17,678 now.
	#however, i'm not sure how necessary it is to do this....ask Leo...?
	#TODO may be worth doing some EDA here


#FILTER TO ADULTS 
cohort_dat = cohort_dat.loc[cohort_dat['AGE']>=age_thresh,:] 

#FILTER ON LOS 
cohort_dat = cohort_dat.loc[cohort_dat['LOS']>=LOS_lower_thresh,:]

#should be at 21583 ICU stays, if limit to first ICU stay, do not limit to first admission, age>=18, LOS>=12

cohort_dat.to_csv(PATH_TO_REPO+"query-data/cohort.csv",index=False)

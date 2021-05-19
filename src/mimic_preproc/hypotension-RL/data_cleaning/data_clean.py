#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Main file to put together a coherent dataset for RL modeling of hypotension management 

UPDATE: significantly changing how we setup discretization of time for all this
to be based on when actions are *actually* taken, and also when it seems like 
decisions to not treat were made...logic TBD here...
	- first pass: every hour or two hours when patient is "very sick"
	- otherwise, decision every 8 hours during periods of stability

@author: josephfutoma
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
from time import time

# from data_load_utils import load_labs_and_vitals

#when you play around with data locally set this to path where everything is stored
PATH_TO_QUERY_DATA = "/n/dtak/mimic-pipeline-tools/mimic-iii-v1-4/query-data/"
PATH_TO_CLEANED_DATA = "/n/dtak/mimic-pipeline-tools/hypotension/model-data/"

pd.set_option("display.max_columns",101)
pd.set_option("display.max_rows",101)

###########
#params for dataset construction

#default thresholds we'll use throughout, so not explicitly in filenames (for brevity)
LOS_LOWER_THRESH = 12 #exclude if LOS is shorter than this many hours
LOS_CAP = 48 #last time point considered, if longer than this cap here & end traj 
#(will take last decision a bit before this final time, eg an hour prior)
MIN_MAP_THRESH = 12 #must have at least this many measured MAP values to be retained

#this is a much harsher filter, may want to play around with this one...
# this filter is only applied to files with 'Xbpleq65' in the filename
MAP_FILTER_THRESH = 65
MAP_NUM_BELOW_THRESH = 3 #must have at least this many MAP values below MAP_FILTER_THRESH to be retained

#NOTE: in this newer version of the data cleaning, there is 
#  no longer a hard start on first decision time to be at hour 1, or on hourly spacing
#
# DANGER! If you are expecting your data to be on an hourly grid, this does NOT do that!

#TODO: eventually later on might implement different args for some of these design choices
BP_AGG_FUNC = 'min' #'min'; how to aggregate BP values to get a single value...
BASELINE_IMPUTE_METHOD = 'median' #for now the only option is population median, but maybe improve later
TIMESERIES_IMPUTE_METHOD = 'last' #for now the only option is sample-and-hold or last-one-carry-forward

SEED = 8675309 #seed for reproducing any results that involve randomness
np.random.seed(SEED)

##########
########## load in the baseline/static data for the cohort
##########

cohort_dat = pd.read_csv(PATH_TO_QUERY_DATA+'cohort.csv')

# do any additional filtering beyond cohort construction, if desired...

cohort_dat = cohort_dat.loc[cohort_dat['LOS'] >= LOS_LOWER_THRESH,:]

print('cohort loaded')

##########
########## LOAD all time series data
##########

all_ts_dats, ts_medians, all_ts_vars = load_labs_and_vitals(
	PATH_TO_QUERY_DATA,cohort_dat,verbose=False)

print('time series variables all loaded')

###########
########### all vital/lab values have been loaded; preprocessing of most extreme values
###########



for v in all_ts_vars:
	dat = all_ts_dats[v]
	if 'GCS' not in v: 
		if VERBOSE:
			print(v)

		vals = np.array(dat['value'],"float")
		lq,uq = np.percentile(vals,[LOWER_Q,UPPER_Q])
		if VERBOSE:
			print(lq,uq)
		dat.loc[vals<=lq,'value'] = lq
		dat.loc[vals>=uq,'value'] = uq

		#some manual clips for certain vars
		if v=='fio2':
			dat.loc[vals<=1,'value'] *= 100 #units off
			dat.loc[vals<21,'value'] = 21 #implausible (and rarely seen) to be lower than room air
		if v=='map':
			dat.loc[vals<=40,'value'] = 40 #implausible
		if v=='sbp':
			dat.loc[vals<=60,'value'] = 60 #implausible
		if v=='dbp':
			dat.loc[vals<=30,'value'] = 30 #implausible		
		if v=='gfr':
			dat.loc[vals>=100,'value'] = 100 #GFR doesn't really mean anything when that high		



###################
################### load in & process all actions. NOTE that times vary across trajectories now!
###################

def total_pressor_normed_amount(start_t,end_t,v_starts,v_ends,v_rates):
	# get total amount of pressor in specified period [t_start,t_end]
	# normalized by size of period
	# TODO: vectorize & speedup...?

	delta_t = end_t - start_t

	#integrate total pressors given this period, catching all edge cases
	pressor_amt = 0
	for v_s, v_e, r in zip(v_starts,v_ends,v_rates):
		if v_s >= start_t and v_e <= end_t:
			pressor_amt += r*60*(v_e-v_s)		
		if v_s < start_t and v_e > end_t:
			pressor_amt += r*60*delta_t
		if v_s < start_t and v_e > start_t and v_e <= end_t:
			pressor_amt += r*60*(v_e-start_t)
		if v_s >= start_t and v_s < end_t and v_e > end_t:
			pressor_amt += r*60*(end_t-v_s)

	pressor_normed_amt = 1/delta_t*pressor_amt
	return pressor_normed_amt


#load in data; actions are already sorted by icu_id & start_time
fluids_dat = pd.read_csv(path_to_query_data+'allfluids_and_bloodproducts_mv.csv')
fluids_dat['STARTTIME'] = pd.to_datetime(fluids_dat['STARTTIME'])
fluids_dat['ENDTIME'] = pd.to_datetime(fluids_dat['ENDTIME'])

#filter & threshold
fluids_dat['AMOUNT'] += 1 #rounding edge cases
fluids_dat = fluids_dat.loc[fluids_dat['AMOUNT'] >= 250, :] #cut tiny fluids
#TODO: why is pandas throwing warning?
fluids_dat['AMOUNT'][fluids_dat['AMOUNT']>=2000] = 2000


vaso_dat = pd.read_csv(path_to_query_data+'vasopressors_mv.csv')
vaso_dat['STARTTIME'] = pd.to_datetime(vaso_dat['STARTTIME'])
vaso_dat['ENDTIME'] = pd.to_datetime(vaso_dat['ENDTIME'])

#threshold; don't filter bc *any* pressor on is a big deal, unlike fluids
vaso_dat['RATE_NORMED_NOREPI'][vaso_dat['RATE_NORMED_NOREPI']>=2.5] = 2.5

#cache for speedup
fluid_starts = fluids_dat['ICUSTAY_ID'].searchsorted(final_ICU_IDs,'left')
fluid_ends = fluids_dat['ICUSTAY_ID'].searchsorted(final_ICU_IDs,'right')
vaso_starts = vaso_dat['ICUSTAY_ID'].searchsorted(final_ICU_IDs,'left')
vaso_ends = vaso_dat['ICUSTAY_ID'].searchsorted(final_ICU_IDs,'right')

###### NOTE: probably not a bad idea to filter to *ONLY* ICU IDs that at least take *some* sort of action...??
###	or at the very least, either have: 
###		- 1+ non-null action, OR
###		- many (5+...?) low MAPs that are < 65...


#first pass we go through and grab all the raw action amounts & times, then discretize
all_rawaction_data = {}

#buffer to give at end of trajetory to assess effect of terminal action 
#UPDATE: this is implicit now by using last MAP as terminating time, and 
#	so last action will be last tx before then...
# END_BUFFER = 1

#buffer around "sick" times, where if no explicit treatment for this long before or after
# a "sick" time, then the "sick" time should be treated as an intentional "dont treat" action
#
# 	define: "very sick" patient to be someone with a MAP of < 60...if MAP is that low and no tx, 
#	we will assume that this was an intentional "dont treat" action...
# TODO: also may want this to depend on lactate...?? or other markers of acuity?
SICK_TIME_BUFFER = 1.01
MAP_SICK_THRESH = 60 #criteria for "sick"...for now...

#if been this many hours or more between other action times (either tx or explicit no-tx), 
#  add an extra action in for explicit no-action, since even "healthy" in ICU is being
#  monitored fairly regularly...
LONG_GAP_TIME = 4.01

loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100 == 99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	# get ICU start time, out time, & LOS 
	this_pop_dat = pop_dat.iloc[ID_ind]
	start_time = this_pop_dat['intime']
	outtime = this_pop_dat['outtime']
	total_time = (outtime-start_time).total_seconds()/60/60

	#either discharge from ICU or capped length (for now 2 days)	
	# NOTE take off some time at very end to give us a buffer,
	# and allow us to assess the effect of the last action taken...
	# we will artifically force last decision time to be here, at very latest
	end_time = min(total_time,LOS_CAP)

	#get MAP values & times to help inform grid...
	s = map_starts[ID_ind]
	e = map_ends[ID_ind]
	this_maps = map_dat[s:e]
	map_times = np.array((this_maps['charttime'] - start_time).astype('timedelta64[m]').astype(float))/60
	map_vals = np.array(this_maps['value'])

	#filter to before end...
	map_inds = np.logical_and(map_times <= end_time, map_times >= 0)
	map_times = map_times[map_inds]
	map_vals = map_vals[map_inds]

	#skip IDs with very few MAPs taken
	if len(map_vals) < MIN_MAP_THRESH:
		continue 

	#UPDATE our terminating time so that last MAP is where we cut things off...
	end_time = np.max(map_times)

	##### get treatment info for this patient

	#fluids
	s = fluid_starts[ID_ind]
	e = fluid_ends[ID_ind]
	this_fdat = fluids_dat[s:e]

	f_starts = np.array((this_fdat['STARTTIME'] - start_time).astype('timedelta64[m]').astype(float))/60
	f_amounts = np.array(this_fdat['AMOUNT'])

	#filter irrelevant fluids
	f_ind = f_starts < end_time
	f_starts = f_starts[f_ind]
	f_amounts = f_amounts[f_ind]

	#pressors
	s = vaso_starts[ID_ind]
	e = vaso_ends[ID_ind]
	this_vdat = vaso_dat[s:e]
	
	v_starts = np.array((this_vdat['STARTTIME'] - start_time).astype('timedelta64[m]').astype(float))/60
	v_ends = np.array((this_vdat['ENDTIME'] - start_time).astype('timedelta64[m]').astype(float))/60
	v_rates = np.array(this_vdat['RATE_NORMED_NOREPI'])

	#filter irrelevant pressors
	v_ind = v_starts < end_time
	v_starts = v_starts[v_ind]
	v_ends = v_ends[v_ind]
	v_rates = v_rates[v_ind]
	v_ends[v_ends > end_time] = end_time #also force pressors to end, at latest, at our artifical end

	### Step 1: we need to get decision times for building this trajectory,
	### 	start by getting all times when treatment decisions were made:
	###			- pressor started
	###			- pressor ended
	###			- fluid started (end irrelevant bc short)

	all_tx_times = np.unique(np.concatenate([f_starts,v_starts,v_ends])) 
	all_tx_times = all_tx_times[all_tx_times < end_time] #only allow tx actions before end buffer

	### Step 2: expressly filter out tx start times where no MAP between treatments: 
	### 	it is assumed safe to combine actions in this window, since action would not have been
	### 	updated by a new MAP, so nothing funny about why follow-up action taken...

	filtered_tx_times = []
	if len(all_tx_times) > 0: #only makes sense if tx actually done...

		#start with first tx start
		filtered_tx_times.append(all_tx_times[0])

		last_t = all_tx_times[0] 
		for t in all_tx_times[1:]: 
			#iterate forward & check if a new MAP btw last tx and next one
			if np.any(np.logical_and(map_times>=last_t, map_times<=t)):
				filtered_tx_times.append(t)
			last_t = t
	filtered_tx_times = np.array(filtered_tx_times)

	### Step 3: get the set of all times where patient is "very sick", and no tx
	###		given either shortly before (i.e. they're working on it already...),
	###		or shortly after (they see this update & change/start tx as appropriate)
	### TODO: better way to get sick times than just MAP <= 60?

	very_sick_times = np.unique(map_times[map_vals <= MAP_SICK_THRESH])
	for t in very_sick_times:
		diffs = filtered_tx_times - t #how far is this sick_time from nearest tx

		#next tx (or sick-time) from this "bad" measurement is "far away"...
		# then treat this new no-action as intentional
		#NOTE: don't need explicit logic to handle edge case where filtered_tx_times is empty
		if np.all(np.abs(diffs) > SICK_TIME_BUFFER):
			filtered_tx_times = np.concatenate([filtered_tx_times,[t]])

	#again filter out from very end & sort
	filtered_tx_times = filtered_tx_times[filtered_tx_times < end_time]
	filtered_tx_times = np.unique(filtered_tx_times)

	### Step 4: Fill in last holes so there's an action 
	### 	(or no-tx) done at least every 4 hours...

	no_sick_gap_times = []

	#add in extra times at very beginning & end to catch edge cases
	tx_sick_times_extended = np.unique(np.concatenate([[0],filtered_tx_times,[end_time]]))
	diffs = tx_sick_times_extended[1:] - tx_sick_times_extended[:-1]
	maxdiff = np.max(diffs)

	while maxdiff > LONG_GAP_TIME:
		### find max diff; 
		max_ind = np.where(diffs==maxdiff)[0][0] #doesn't matter if many just grab first

		### add midpoint time between these two far apart times...
		new_time = (tx_sick_times_extended[max_ind]+tx_sick_times_extended[max_ind+1])/2
		tx_sick_times_extended = np.unique(np.concatenate([tx_sick_times_extended,[new_time]]))
		no_sick_gap_times.append(new_time)

		#keep going until all gaps are small enough...
		diffs = tx_sick_times_extended[1:] - tx_sick_times_extended[:-1]
		maxdiff = np.max(diffs)

	##### yay, we now have a final set of action times for our time discretization.
	#####     now, add up all tx amounts for the action times to actually compute actions...

	#add in 0 & end_time even though not explicitly an action; 
	#need to get the tx amounts per period for edge cases each end,
	# even if not explicitly modeling as action...
	final_action_times = np.unique(np.concatenate([[0],filtered_tx_times,no_sick_gap_times,[end_time]]))

	### ok, now collect up the action amounts...

	#tricky so use helper func
	pressor_normed_amts = []
	for (t0,t1) in zip(final_action_times[:-1],final_action_times[1:]):
		amt = total_pressor_normed_amount(t0,t1,v_starts,v_ends,v_rates)
		pressor_normed_amts.append(amt)
	pressor_normed_amts.append(np.nan) #none at end; not tracking anything after terminal end_time

	#more straightforward; not integrating, just treat as point mass with no timing
	fluid_amts = []
	for (t0,t1) in zip(final_action_times[:-1],final_action_times[1:]):
		fluid_inds = np.logical_and(f_starts >= t0, f_starts < t1)
		fluid_amts.append(np.sum(f_amounts[fluid_inds]))
	fluid_amts.append(np.nan)

	############ 

	actions_dat = pd.DataFrame()
	actions_dat['Times'] = final_action_times
	actions_dat['Vasopressor_normed_amt'] = np.array(pressor_normed_amts)
	actions_dat['Total_fluid_bolus_amt'] = np.array(fluid_amts)

	### SAVE
	all_rawaction_data[ID] = actions_dat


pickle.dump(all_rawaction_data,open(path_to_model_data+'all_raw_actions_times.p','wb'))
all_rawaction_data = pickle.load(open(path_to_model_data+'all_raw_actions_times.p','rb'))

# filtered down cohort slightly...
final_ICU_IDs = np.sort(np.array(list(all_state_data.keys())))
#update pop_dat to account for filtering from stays with few MAPs...
pop_dat = pop_dat.loc[np.in1d(pop_dat['icustay_id'],final_ICU_IDs),:]

assert final_ICU_IDs.shape[0] == pop_dat.shape[0]


### test actions btw 0 & actual first action_time...
# all_fluids = []
# all_pressors = []
# for ID in final_ICU_IDs:
# 	actions_dat = all_rawaction_data[ID]
# 	first_act = actions_dat.iloc[0]

# 	all_fluids.append(first_act['Total_fluid_bolus_amt'])
# 	all_pressors.append(first_act['Vasopressor_normed_amt'])

# all_fluids = np.array(all_fluids)
# all_pressors = np.array(all_pressors)

# print('fluid counts for 0 - first time')
# print(np.unique(all_fluids,return_counts=True))
# print('pressor counts for 0 - first time')
# print(np.unique(all_pressors,return_counts=True))

# ok, so only about 0.5% of time is there an action btw 0 & first flagged action time we have...
# ignore and these won't be modeled explicitly w RL but will at least inform states 
# when we aggregate some features based on past tx amounts given...
# TODO: what the fuck, how/why is this edge case happening...

########## ok, now let's discretize actions into bins...

fluid_cutoffs = [0,500,1000,1e8] #4 bins
vaso_cutoffs = [0,8.1,21.58,1e8] #4 bins based on 33.3/66.7% quantiles of nonzeros

EPS = 1e-8
FLUIDS_BINS = np.array(fluid_cutoffs)+EPS; FLUIDS_BINS[0] = 0
VASO_BINS = np.array(vaso_cutoffs)+EPS; VASO_BINS[0] = 0

NUM_VASO_BINS = len(VASO_BINS) #4
NUM_FLUID_BINS = len(FLUIDS_BINS) #4
#in overall numbering, we'll have 0 = no action; 1 = dose 1 of fluids, 0 vaso; 
# 2 = dose 2 of fluids, 0 vaso; ... ; F = max fluids, 0 vaso; F+1 = 0 fluids, 1 vaso, ... 
NUM_ACTIONS = NUM_VASO_BINS*NUM_FLUID_BINS

def vaso_fluid_to_bin(fluids,vasos):
	f_ids = np.searchsorted(FLUIDS_BINS,fluids)
	v_ids = np.searchsorted(VASO_BINS,vasos)

	overall_ids = f_ids + NUM_FLUID_BINS*v_ids

	return f_ids,v_ids,overall_ids


##### ok group raw action amounts...almost done...

all_actions_disc_data = {}

loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100 == 99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	dat = all_rawaction_data[ID]

	fluids = np.array(dat['Total_fluid_bolus_amt'])
	vasos = np.array(dat['Vasopressor_normed_amt'])

	f_ids,v_ids,overall_ids = vaso_fluid_to_bin(fluids,vasos)
	f_ids[-1] = v_ids[-1] = overall_ids[-1] = -999 #no last action

	dat['OVERALL_ACTION_ID'] = overall_ids
	dat['FLUID_ID'] = f_ids
	dat['VASO_ID'] = v_ids

	all_actions_disc_data[ID] = dat

pickle.dump(all_actions_disc_data,open(path_to_model_data+'all_disc-v4f4_actions_times.p','wb'))
# all_actions_disc_data = pickle.load(open(path_to_model_data+'all_disc-v4f4_actions_times.p','rb'))

#get overall action cts
all_actions_disc = []
for ID in final_ICU_IDs:
	acts = np.array(all_actions_disc_data[ID]['OVERALL_ACTION_ID'])[1:-1]
	all_actions_disc.append(acts)
all_actions_disc = np.concatenate(all_actions_disc)

action_cts = np.unique(all_actions_disc,return_counts=True)
for a,c in zip(action_cts[0],action_cts[1]):
	print(a,c,c/len(all_actions_disc)*100)



##### go back and add in extra vars: total fluids & pressors so far, and in past 8 hours

### TODO: THIS IS BROKEN & NOT REALLY 8 HOURS BC OF TIME GAPS...ITS LAST 8 TIME PTS...

def rolling_sum(vec,k=8):
	res = np.cumsum(np.array(vec,"float"))
	res[k:] -= res[:-k]
	return res

loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100==99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	actions_dat = all_actions_disc_data[ID]

	### off by 1 here; the prev/last vars should be 0 initially, and then drop the last one...
	z = np.zeros(actions_dat.shape[0])

	actions_dat['total_all_prev_vasos'] = z
	actions_dat['total_all_prev_vasos'][1:] = np.cumsum(np.array(actions_dat['Vasopressor_normed_amt'],"float"))[:-1]
	actions_dat['total_all_prev_fluids'] = z
	actions_dat['total_all_prev_fluids'][1:] = np.cumsum(np.array(actions_dat['Total_fluid_bolus_amt'],"float"))[:-1]
	actions_dat['total_last_8hrs_vasos'] = z
	actions_dat['total_last_8hrs_vasos'][1:] = rolling_sum(actions_dat['Vasopressor_normed_amt'],8)[:-1]
	actions_dat['total_last_8hrs_fluids'] = z
	actions_dat['total_last_8hrs_fluids'][1:] = rolling_sum(actions_dat['Total_fluid_bolus_amt'],8)[:-1]

	all_actions_disc_data[ID] = actions_dat

pickle.dump(all_actions_disc_data,open(path_to_model_data+'all_disc-v4f4_actions_times_prevactions.p','wb'))
# all_actions_disc_data = pickle.load(open(path_to_model_data+'all_disc-v4f4_actions_times_prevactions.p','rb'))


########## yay lets do states already


all_state_data = {}

baseline_cov_names = ['age','is_F','surg_ICU','is_not_white',
	'is_emergency','is_urgent','hrs_from_admit_to_icu']

#cache cutpoints in sorted dataframes in advance, much faster
starts_ts = {}; ends_ts = {}
for v in all_ts_vars:
	dat = all_ts_dats[v]
	starts_ts[v] = dat['icustay_id'].searchsorted(final_ICU_IDs,'left')
	ends_ts[v] = dat['icustay_id'].searchsorted(final_ICU_IDs,'right')


loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100==99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	###############
	#static vars first
	this_pop_dat = pop_dat.iloc[ID_ind]
	start_time = this_pop_dat['intime']
	outtime = this_pop_dat['outtime']
	total_time = (outtime-start_time).total_seconds()/60/60

	#build out all static vars
	age = float(this_pop_dat['age'])
	if age > 90: age = 90 #BUG: tons of ages are 300!! will screw up standardization
	if np.isnan(age): age = base_medians['age']

	is_F = int(this_pop_dat['gender']=='F')
	if np.isnan(is_F): is_F = 0

	icu_type = this_pop_dat['first_careunit']
	surg_ICU = int(icu_type=='CSRU' or icu_type=='SICU' or icu_type=='TSICU')

	ethn = this_pop_dat['ethnicity']
	is_not_white = 0
	if 'WHITE' not in ethn: is_not_white = 1

	admit_type = this_pop_dat['admission_type']
	is_emergency = 0
	is_urgent = 0
	if admit_type=='EMERGENCY': is_emergency=1
	if admit_type=='URGENT': is_urgent=1

	hrs_from_admit_to_icu = (this_pop_dat['intime']-this_pop_dat['admittime']).total_seconds()/60/60
	if hrs_from_admit_to_icu<0: hrs_from_admit_to_icu = 0 #weird edge case

	baseline_covs = np.array([age,is_F,surg_ICU,
		is_not_white,is_emergency,is_urgent,hrs_from_admit_to_icu])

	###############
	##### figure out time stamps

	this_act_dat = all_actions_disc_data[ID]

	#drop initial action time; just so we can get (rare) initial actions
	grid_times = np.array(this_act_dat['Times'])[1:] 
	n_t = len(grid_times)
	n_dec = n_t - 1 #no action taken at very last time point; used for final transition & reward


	###############
	### state variables

	states_dat = pd.DataFrame()
	states_dat['Times'] = grid_times
	states_dat['normed_time'] = grid_times/LOS_CAP

	#add baseline data in first
	for var,val in zip(baseline_cov_names,baseline_covs):
		states_dat[var] = val


	#####
	#now build out the time series
	#also build out indicator vector at each time, noting whether var was imputed or not
	for v in all_ts_vars:
		s = starts_ts[v][ID_ind]
		e = ends_ts[v][ID_ind]
		this_dat = all_ts_dats[v][s:e]
		this_t = np.array((this_dat['charttime'] - start_time).astype('timedelta64[m]').astype(float))/60
		if 'GCS' in v:
			this_vals = np.array(this_dat['valuenum'])
		else:
			this_vals = np.array(this_dat['value'])

		#impute anything initially missing with pop median, then
		#fill this in with observed values via LOCF
		imputed_vals = ts_medians[v]*np.ones(n_t)

		### Now get LOCF for cts labs/vitals
		tt = np.searchsorted(grid_times+1e-8,this_t)
		
		for i in range(len(tt)):
			if i!=len(tt)-1:
				imputed_vals[tt[i]:tt[i+1]] = this_vals[i] 
			else:
				imputed_vals[tt[i]:] = this_vals[i] 
		
		#EXCEPT: for MAP/SBP/DBP, we fill in with the worst in the window
		#	this only applies to windows in which *more* than 1 MAP is measured.
		#	after these windows, the LOCF kicks in with the most recent value from them
		if v in ['map','sbp','dbp']:
			u_tt = np.unique(tt)
			starts = np.searchsorted(tt,u_tt,'left')
			ends = np.searchsorted(tt,u_tt,'right')
			for t,s,e in zip(u_tt,starts,ends):
				if t>=0 and t<n_t: 
					imputed_vals[t] = np.min(this_vals[s:e])

		#get indicators for at which times the variable was actually sampled
		inds_samples_vals = np.zeros(n_t+1) #edge case when values past endtime
		inds_samples_vals[tt] = 1.0
		inds_samples_vals = inds_samples_vals[:-1] #edge case when values past endtime

		states_dat[v] = imputed_vals
		states_dat[v+'_ind'] = inds_samples_vals

	#last, combine the 3 GCS vars & then toss the individuals
	states_dat['GCS'] = states_dat['GCS_eye']+states_dat['GCS_motor']+states_dat['GCS_verbal']
	states_dat['GCS_ind'] = states_dat['GCS_eye_ind']+states_dat['GCS_motor_ind']+states_dat['GCS_verbal_ind']
	states_dat['GCS_ind'] = np.minimum(1,states_dat['GCS_ind'])

	#also drop GFR indicator since same as creat
	states_dat = states_dat.drop(['GCS_eye','GCS_motor','GCS_verbal','GCS_eye_ind',
		'GCS_motor_ind','GCS_verbal_ind','gfr_ind'],axis=1)

	all_state_data[ID] = states_dat


pickle.dump(all_state_data,open(path_to_model_data+'all_states.p','wb'))
# all_state_data = pickle.load(open(path_to_model_data+'all_states.p','rb'))


############
############ setup rewards for each ICU stay
############

def lin_reward_func(bps,cutoffs=[40,55,60,65],vals=[-1,-0.15,-0.05,0]): 
	return np.interp(bps,cutoffs,vals)
# xx = np.linspace(40,75,1000); plt.plot(xx,np.log(1+lin_reward_func(xx)+1e-8)); plt.show()
# xx = np.linspace(40,75,1000); plt.plot(xx,lin_reward_func(xx)); plt.show()

all_reward_data = {}

URINE_OUTPUT_THRESH = 30 #if urine is above this, we're not too worried about BP if it's above 55
MAP_UO_THRESH = 55 #as long as MAP is above this, give max reward as long as UO is ok

for ID in final_ICU_IDs:

	s_dat = all_state_data[ID]

	times = np.array(s_dat['Times'])[:-1] #last state doesn't have reward, no action (just use state to compute reward)
	
	bps = np.array(s_dat['map'])[1:] #first bp reading not used for reward; bp_t, take a_t, r_t = f(bp_t+1)
	rewards = lin_reward_func(bps)

	#extra mask to ensure UO measured, if not measured yet, treat as if bad.
	# don't want to give a free pass to slightly low MAPs that we got
	# before a UO was measured
	uos = np.array(s_dat['urine'])
	uo_inds = np.array(s_dat['urine_ind'])
	if np.all(uo_inds==0):
		uos *= 0
	else:
		first_meas_uo = np.where(uo_inds==1)[0][0]
		uos[:first_meas_uo] = 0
	#ok, now that fixed initial UOs, cut to one ahead for rewards...
	uos = uos[1:]

	good_inds = np.logical_and(uos >= URINE_OUTPUT_THRESH, bps >= MAP_UO_THRESH)

	rewards[good_inds] = 0 #moderate hypotension but good UO = ok

	rewards_dat = pd.DataFrame()
	rewards_dat['Times'] = times
	rewards_dat['Rewards'] = rewards

	all_reward_data[ID] = rewards_dat

pickle.dump(all_reward_data,open(path_to_model_data+'rewards.p','wb'))
# all_reward_data = pickle.load(open(path_to_model_data+'rewards.p','rb'))


##### extend the state space...
##### build out additional indicator variables for labs: 8 hour and ever-measured inds
#####	(no vitals since super frequent as is)


### helper funcs to convert inds at hourly level to other levels
def convert_1inds_to_kinds(ind_vec,times,k=8):
	#NOTE: old func breaks here bc times not hourly...
	n = len(ind_vec)
	assert len(times)==n

	res = np.zeros(n,"float")
	res[0] = ind_vec[0]
	for i in range(1,n):
		this_t = times[i]
		rel_inds = np.logical_and(times >= this_t-k, times <= this_t)
		res[i] = np.any(ind_vec[rel_inds]==1)

	return res

def convert_1inds_to_everinds(ind_vec):
	res = np.cumsum(np.array(ind_vec))
	return np.array(res>=1,"float")

#add in 8hour and ever-measured indicators for these variables (mostly labs, not measured as often)
extra_ind_vars = np.array(['bicarbonate', 'bun', 'creatinine', 'fio2',
       'glucose', 'hct', 'lactate', 'magnesium','platelets', 'potassium', 'sodium', 
       'wbc', 'alt','ast', 'bilirubin_total', 'co2', 'hgb','inr','pco2', 'po2', 'weight'])

all_extended_states_dat = {}

loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100==99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	states_dat = all_state_data[ID]
	this_times = np.array(states_dat['Times'])
	for v in extra_ind_vars:
		# SLOW!!
		# states_dat[v+'_8ind'] = convert_1inds_to_kinds(states_dat[v+'_ind'],this_times,8)
		states_dat[v+'_everind'] = convert_1inds_to_everinds(states_dat[v+'_ind'])

	acts_dat = all_actions_disc_data[ID]

	#TODO check for off by 1's
	for act_v in range(1,NUM_VASO_BINS):
		states_dat['last_vaso_'+str(act_v)] = np.array(np.array(acts_dat['VASO_ID'])[:-1]==act_v,"float")

	for act_f in range(1,NUM_FLUID_BINS):
		states_dat['last_fluid_'+str(act_f)] = np.array(np.array(acts_dat['FLUID_ID'])[:-1]==act_f,"float")

	states_dat['total_all_prev_vasos'] = np.array(acts_dat['total_all_prev_vasos'])[1:]
	states_dat['total_all_prev_fluids'] = np.array(acts_dat['total_all_prev_fluids'])[1:]
	states_dat['total_last_8hrs_vasos'] = np.array(acts_dat['total_last_8hrs_vasos'])[1:]
	states_dat['total_last_8hrs_fluids'] = np.array(acts_dat['total_last_8hrs_fluids'])[1:]

	all_extended_states_dat[ID] = states_dat

#list of all vars
state_vars = np.array(all_extended_states_dat[final_ICU_IDs[0]].columns)

pickle.dump(all_extended_states_dat,open(path_to_model_data+'all_states_extravars.p','wb'))
# all_state_data = pickle.load(open(path_to_model_data+'all_states_extravars.p','rb'))





#############
############# write out as npz for safe loading
#############

all_states_np = []
all_actions_np = []
all_rewards_np = []

for ID in final_ICU_IDs:

	this_s = all_extended_states_dat[ID]
	this_a = all_actions_disc_data[ID]
	this_r = all_reward_data[ID]

	states_np = np.array(this_s)[:,1:] # dim: T+1 x n_S
	act_np = np.array(this_a)[1:-1,3] #dim: T
	rew_np = np.array(this_r)[:,1] # dim: T

	all_states_np.append(states_np)
	all_actions_np.append(act_np)
	all_rewards_np.append(rew_np)


np.savez(path_to_model_data+'states_actions_rewards_IDs.npz',
	all_states=all_states_np,all_actions=all_actions_np,
	all_rewards=all_rewards_np,all_IDs=final_ICU_IDs,
	state_var_names=state_vars[1:])


















#### log, then standardize appropriate columns after filtering 







all_states = []
for ID in final_ICU_IDs:
	this_states = all_extended_states_dat[ID]
	all_states.append(np.array(this_states))
all_states = np.vstack(all_states)

all_state_vars = np.array(this_states.columns)

vars_no_processing = ['Times','normed_time','is_F','surg_ICU','is_not_white','is_emergency','is_urgent','last_reward']
inds_to_process = [] #normed_time (already scaled); anything with ind; last_vaso_X or last_fluid_X;
for i,v in enumerate(all_state_vars):
	if v in vars_no_processing or 'ind' in v or 'last_fluid' in v or 'last_vaso' in v:
		pass
	else:
		inds_to_process.append(i)
		print(all_state_vars[i])
inds_to_process = np.array(inds_to_process)

# all_state_means = np.mean(all_states,0)
# all_state_sds = np.std(all_states,0)
# for v,m,s in zip(all_state_vars,all_state_means,all_state_sds):
# 	print(v,m,s)

all_logstate_means = np.mean(np.log(all_states+.1),0)
all_logstate_sds = np.std(np.log(all_states+.1),0)



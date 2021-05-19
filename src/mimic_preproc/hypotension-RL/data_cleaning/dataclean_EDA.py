#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Dump of some old portions of data cleaning script related to simple EDA & visualizations,
in part to setup state vars and in part to setup action space to discretize

In general can ignore, but may be useful for reassessing how to discretize actions

@author: josephfutoma
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from time import time


### a little EDA to figure out how we want to discretize things now...



### fluids

all_fluid_units = np.array(fluids_dat['AMOUNTUOM'])
unq_units = np.unique(all_fluid_units,return_counts=True)

all_fluid_vals = np.array(fluids_dat['AMOUNT'])
all_fluid_vals += 1 #fix weird edge cases due to rounding / data entry

#filter out low values not useful, and cap high values (since discretizing anyways)
all_fluid_vals = all_fluid_vals[all_fluid_vals>=250]
all_fluid_vals[all_fluid_vals>=2000] = 2000

qs = [0,.1,1,2.5,5,10,25,50,75,90,95,97.5,99,99.9,100]
fluid_percs = np.percentile(all_fluid_vals,qs)
for q,f in zip(qs,fluid_percs):
	print(q,f)


plt.close('all')
plt.figure(figsize=(6,6))
plt.hist(all_fluid_vals,100)
plt.show()

### seems like we should do 1000mL+, 500-1000mL, and 250-500. 
# TODO: how do we want to aggregate multiple smaller fluids given in short period that add
#   to something substantial?

# HACK: for now, just filter and only keep fluids that are 250ml or more...

fluid_IDs = np.unique(fluids_dat['ICUSTAY_ID'])

def plot_fluids(ID):

	this_start_t = pop_dat.loc[pop_dat['icustay_id']==ID,'intime'].iloc[0]
	this_end_t = pop_dat.loc[pop_dat['icustay_id']==ID,'LOS'].iloc[0]

	this_fluids = fluids_dat.loc[fluids_dat['ICUSTAY_ID']==ID,:]
	this_doses = np.array(this_fluids['AMOUNT'])
	this_times = np.array(this_fluids['STARTTIME']-this_start_t).astype('timedelta64[m]').astype(float)/60
	keep_inds = np.logical_and(this_doses>100,this_times<48)
	this_doses = this_doses[keep_inds]
	this_times = this_times[keep_inds]

	# get rolling sum of fluid amounts in X hours
	LOOK_AHEAD_WINDOW = 15/60

	fluid_totals = []
	for t in this_times :
		fluid_totals.append(np.sum(this_doses[np.logical_and(this_times>=t,this_times<=t+LOOK_AHEAD_WINDOW)]))
	fluid_totals = np.array(fluid_totals)

	# plt.close('all')
	plt.figure(figsize=(8,8))
	plt.scatter(this_times,this_doses)
	plt.plot(this_times,fluid_totals)
	plt.show()

try:
	ID = fluid_IDs[np.random.randint(len(fluid_IDs))]
	plot_fluids(ID)
except:
	pass








### vaso EDA

all_vaso_rates = np.array(vaso_dat['RATE_NORMED_NOREPI'])

all_vaso_rates[all_vaso_rates>2.5] = 2.5

qs = [0,.1,1,2.5,5,10,25,50,75,90,95,97.5,99,99.9,100]
vaso_percs = np.percentile(all_vaso_rates,qs)
for q,v in zip(qs,vaso_percs):
	print(q,v)


plt.close('all')
plt.figure(figsize=(6,6))
plt.hist(all_vaso_rates,250)
plt.show()



"""
vaso logic a lot trickier...need to figure out times when things start AND end,
since no given grid anymore...

X = 15 min...??

- if multiple drugs started at exact same time or within next X min, add up total norepi
	drug rates across types to get an overall pressor "rate"
- if multiple rates for a single drug in next X min, take max of them all

- always keep track of the current rate

- any time a drug is turned off or rate changes (outside X min blocks), then 
	refigure out current rate

- if rate has been dropped to 0, then we're back to a no-pressor action


"""








####### MAP EDA: what should discretization be??

map_starts = map_dat['icustay_id'].searchsorted(final_ICU_IDs,'left')
map_ends = map_dat['icustay_id'].searchsorted(final_ICU_IDs,'right')

all_map_timings = [] #get total diffs in times btw MAP measurements

loop_t = time()
for ID_ind,ID in enumerate(final_ICU_IDs):
	if ID_ind % 100 == 99:
		print("processing %d/%d, took %.2f" %(ID_ind+1,len(final_ICU_IDs),time()-loop_t))
		loop_t = time()

	this_pop_dat = pop_dat.iloc[ID_ind]

	start_time = this_pop_dat['intime']
	outtime = this_pop_dat['outtime']
	total_time = (outtime-start_time).total_seconds()/60/60

	#either discharge from ICU or capped length (for now 2 days)	
	# NOTE take off some time at very end to give us a buffer,
	# and allow us to assess the effect of the last action taken...
	# we will artifically force this to be a decision time as traj is ending
	end_time = min(total_time,LOS_CAP)

	s = map_starts[ID_ind]
	e = map_ends[ID_ind]
	this_maps = map_dat[s:e]
	map_times = np.array((this_maps['charttime'] - start_time).astype('timedelta64[m]').astype(float))/60
	map_vals = np.array(this_maps['value'])
	map_inds = map_times <= end_time+1
	map_times = map_times[map_inds]
	map_vals = map_vals[map_inds]

	map_diff_timings = map_times[1:] - map_times[:-1] #time between consecutive MAPs

	all_map_timings.append(map_diff_timings)



all_map_timings_stacked = np.concatenate(all_map_timings,0)

tmp = np.array(all_map_timings_stacked)
tmp = tmp[tmp<=3]
tmp = tmp[tmp>=.001]

plt.close('all')
plt.figure(figsize=(8,8))
plt.hist(tmp,100)
plt.show()

## about 1.2% of MAPs are taken 2 or more hours apart
## about 63.7% of MAPs are spread out by 0.9-1.1 hours apart
## about 67.5% of MAPs spread out by 0.9+ hours apart
## about 32.5% of MAPs taken less than .9 hours apart
## about 27% of MAPs taken within 0.5 hours 
## about 21% of MAPs taken within 0.25 hours (15 min)
## about 7.6% of MAPs taken within 1 min (explicitly excluding measurements at exact same time)



### DECISION: break nonzero fluid amounts down into 3 bins: [250,500), [500,1000), [1000,MAX]
	#TODO figure out logic to catch many small fluids in short time that may add up...?

### DECISION: figure out vaso discretization *after* we aggregate across all drug types...


#first test...did anything weird happen with actions btw 

all_fluids = []
all_pressors = []
for ID in final_ICU_IDs:
	actions_dat = all_rawaction_data[ID]
	first_act = actions_dat.iloc[0]

	all_fluids.append(first_act['Total_fluid_bolus_amt'])
	all_pressors.append(first_act['Vasopressor_normed_amt'])

all_fluids = np.array(all_fluids)
all_pressors = np.array(all_pressors)

np.unique(all_fluids,return_counts=True)
np.unique(all_pressors,return_counts=True)

### hmm, ok...pretty much always no action btw 0 & 
# first flagged tx time...

all_fluids = []
all_pressors = []
for ID in final_ICU_IDs:
	actions_dat = all_rawaction_data[ID]
	all_fluids.append(np.array(actions_dat['Total_fluid_bolus_amt'])[1:-1])
	all_pressors.append(np.array(actions_dat['Vasopressor_normed_amt'])[1:-1])

all_fluids = np.concatenate(all_fluids)
all_pressors = np.concatenate(all_pressors)

# all_fluids[all_fluids>2000] = 2000
all_fluids_nz = all_fluids[all_fluids>0]

plt.hist(all_fluids_nz,100)
plt.show()

# all_pressors[all_pressors>50] = 50
all_pressors_nz = all_pressors[all_pressors>0]


plt.close('all')

plt.subplot(4,1,1)


plt.hist(np.log(all_pressors_nz),100)

plt.vlines(np.log(np.percentile(all_pressors_nz,33)),0,1000)
plt.vlines(np.log(np.percentile(all_pressors_nz,67)),0,1000)
# plt.vlines(np.log(np.percentile(all_pressors_nz,60)),0,1000)
# plt.vlines(np.log(np.percentile(all_pressors_nz,80)),0,1000)

plt.show()








plt.subplot(4,1,2)
plt.hist(np.log(all_pressors_nz),100)

plt.subplot(4,1,3)
plt.hist(all_pressors_nz[all_pressors_nz<50],100)

plt.subplot(4,1,4)
plt.hist(np.log(all_pressors_nz[all_pressors_nz<50]),100)

plt.show()


# qs = [0,.1,.5,1,2.5,5,10,20,25,40,50,60,75,80,90,95,97.5,99,99.5,99.9,100]
qs = [33.333, 66.666]
pressor_qs = np.percentile(all_pressors_nz,qs)

for q,p in zip(qs,pressor_qs):
	print(q,p)






	# plt.close('all')
	# plt.figure(figsize=(6,6))
	# plt.plot(map_times,map_vals)
	# plt.plot([0,end_time+END_BUFFER],[65,65],color='grey',alpha=.5)
	# plt.plot(query_times,pressor_rates*500)
	# # plt.scatter(v_starts,v_rates*250)
	# # plt.scatter(v_ends,v_rates*250)
	# plt.show()

################################################# OLD









# just BP
plt.hist(vals[np.logical_and(vals>=28,vals<=190)],40); 
plt.vlines(55,0,500000s,color='red')
plt.vlines(75,0,500000,color='red')
plt.title('BP'); 
plt.show()

# histogram of all vals
sns.set(font_scale=.5)
plt.close('all')
plt.figure(figsize=(25,25))
for i,v in enumerate(all_ts_vars):
	if 'GCS' not in v:
		vals = np.array(all_ts_dats[v]['value'],"float")
	else:
		vals = np.array(all_ts_dats[v]['valuenum'],"float")
	print(v)
	print(np.round(np.percentile(vals,[0,.1,1,5,10,25,50,75,90,95,99,99.9,100]),2))

	plt.subplot(8,4,i+1)

	# plt.hist(vals,50)

	tmp = np.log(vals+.1)
	tmp = (tmp-np.mean(tmp))/np.std(tmp)
	tmp = sigmoid_squash(tmp)
	plt.hist(tmp,30)

	plt.title(v)

plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/all_vars_hists.pdf')
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_all_vars_hists.pdf')
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_std_all_vars_hists.pdf')
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_std_squash_all_vars_hists.pdf')

plt.show() 



##############s


# get hist of how long each traj is
nt = []
n_map_inds = []
for dat in all_state_data.values():
	nt.append(dat.shape[0])
	n_map_inds.append(np.sum(dat,0)['map_ind'])
np.percentile(nt,[0,5,25,50,75,95,100])
np.percentile(n_map_inds,[0,5,25,50,75,95,100])
n_MAP_meas = np.array(n_MAP_meas) #number of actual MAP values




states_dat = filtered_states_data
actions_dat = filtered_actions_data
rewards_dat = filtered_rewards_data
#[10142], for 1 hour...
final_ICU_IDs = np.array(list(states_dat.keys())) 


all_vaso = []
all_fluids = []

# pt_v_ct = 0 # 4958 / 13636 = 36% (1hr)
# pt_f_ct = 0 # 10815 / 13636 = 79% (1hr)
pt_v_ct = 0 # 4073 / 10142 = 40%
pt_f_ct = 0 # 8231 / 10142 = 81%

for dat in actions_dat.values():
	all_vaso.extend(dat['Vasopressors'])
	all_fluids.extend(dat['Fluids'])

	if np.any(dat['Vasopressors']>0):
		pt_v_ct += 1
	if np.any(dat['Fluids']>0):
		pt_f_ct += 1		


all_vaso = np.array(all_vaso)
all_fluids = np.array(all_fluids)


qs = [0,1,2.5,5,10,25,50,75,90,95,97.5,99,100]
qs = np.linspace(0,100,21)



print('vaso % nonzero')
print(np.mean(all_vaso>0))
print('-----')

vaso_q = np.percentile(all_vaso[all_vaso>0],qs)
for i,j in zip(qs,vaso_q):
	print(np.round(i,3),np.round(j,3))

all_vaso_nz = all_vaso[all_vaso>0]

all_vaso_nz[all_vaso_nz > 150] = 150 #1 hour


vaso_q = np.percentile(all_vaso_nz,qs)
for i,j in zip(qs,vaso_q):
	print(np.round(i,3),np.round(j,3))



plt.hist(all_vaso_nz,50)
plt.title('vaso doses, %d hr spaced' %int(TIME_WINDOW))
# plt.show()
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/vaso_hist_%dhr_%dbpleq65.pdf' %(int(TIME_WINDOW),MAP_NUM_BELOW_THRESH))
plt.close()


#OK let's do cutoff of 40 as a super high dose; to add more bins, just increase granularity in <40  (1 hour)

all_vaso_nz_low = all_vaso_nz[all_vaso_nz<40] #1 hour
# all_vaso_nz_low = all_vaso_nz[all_vaso_nz<240] #6 hour 

plt.hist(all_vaso_nz_low,50)
plt.title('vaso doses, %d hr spaced' %int(TIME_WINDOW))
# plt.show()
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/lowvaso_hist_%dhr_%dbpleq65.pdf' %(int(TIME_WINDOW),MAP_NUM_BELOW_THRESH))
plt.close()

np.percentile(all_vaso_nz_low,[50]) #9
np.percentile(all_vaso_nz_low,[33.3,66.7]) #5, 13.5
np.percentile(all_vaso_nz_low,[25,50,75]) #3.5, 9, 17
#Let's just separate by sub-dividing the 0-40 range up into equal bins?


cutoffs = [0,40,150] #3 vaso bins
cutoffs = [0,20,40,150] #4 vaso bins: 
cutoffs = [0,13,26,40,150] # 5 vaso bins: 
cutoffs = [0,10,20,30,40,150] #6 vaso bins

cutoffs = [0,5,15,40,150] #RUN WITH THIS
# 0.0 5.0
# 0.2825776154448213 27078
# 5.0 15.0
# 0.3230472214975215 30956
# 15.0 40.0
# 0.26024523871641014 24938
# 40.0 150.0
# 0.13412992434124707 12853

for c1,c2 in zip(cutoffs[:-1],cutoffs[1:]):
	c1 *= TIME_WINDOW
	c2 *= TIME_WINDOW
	print(c1,c2)
	if c2 != cutoffs[-1]:
		tmp = np.logical_and(all_vaso_nz>=c1,all_vaso_nz<c2)
		print(np.mean(tmp),np.sum(tmp))
	else:
		tmp = np.logical_and(all_vaso_nz>=c1,all_vaso_nz<=c2)
		print(np.mean(tmp),np.sum(tmp))




#########


print('fluids % nonzero')
print(np.mean(all_fluids>0))
print('-----')

fluid_q = np.percentile(all_fluids[all_fluids>0],qs)
for i,j in zip(qs,fluid_q):
	print(np.round(i,3),np.round(j,3))

all_fluids_nz = all_fluids[all_fluids>0]
all_fluids_nz[all_fluids_nz < 200] = 0
all_fluids_nz[all_fluids_nz > 2000] = 2000
all_fluids_nz = all_fluids_nz[all_fluids_nz>0]


fluid_q = np.percentile(all_fluids_nz,qs)
for i,j in zip(qs,fluid_q):
	print(np.round(i,3),np.round(j,3))


##### check on most common fluid doses...
tmp = np.histogram(all_fluids_nz,100)
for i,j in zip(tmp[0],tmp[1]):
	print(np.round(i),np.round(j))

unqs = np.unique(all_fluids_nz,return_counts=True)
inds = np.argsort(unqs[1])[::-1]
for i in range(25):
	print(unqs[0][inds[i]],unqs[1][inds[i]])
######

plt.hist(all_fluids_nz,100)
plt.title('fluid doses, %d hr spaced (<200 cut)' %int(TIME_WINDOW))
# plt.show()
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/fluid_hist_%dhr_%dbpleq65.pdf' %(int(TIME_WINDOW),MAP_NUM_BELOW_THRESH))
plt.close()

all_fluids_nz += 0.01 #weird edge cases with numeric issues like 499.99998

#1 hour 
cutoffs = [0,1000,2001] # 3 fluid doses: 66.7% / 33.3% split (w/in any fluid). 
cutoffs = [0,500,1000,2001] #4 fluid doses: 30.3 / 36.4 / 33.3 % GO WITH THIS



for c1,c2 in zip(cutoffs[:-1],cutoffs[1:]):
	print(c1,c2)
	if c2 != cutoffs[-1]:
		tmp = np.logical_and(all_fluids_nz>=c1,all_fluids_nz<c2)
		print(np.mean(tmp),np.sum(tmp))
	else:
		tmp = np.logical_and(all_fluids_nz>=c1,all_fluids_nz<=c2)
		print(np.mean(tmp),np.sum(tmp))




"""
most common doses of fluids:
100.0 51301
200.0 28526
50.0 20509
500.0 11060
250.0 7907
1000.0 7553

1.0 5236
300.0 3230
150.0 2895

100
250
500
>1000

"""


####### ACTIONS EDA

NUM_VASO_BINS = 5
NUM_FLUID_BINS = 4
NUM_ACTIONS = NUM_VASO_BINS*NUM_FLUID_BINS
TIME_WINDOW = 1
MAP_NUM_BELOW_THRESH = 3
all_disc_actions = pickle.load(open(MIMIC_DATA_PATH+'model_data/actions_discretized_vaso%d_fluid%d_%dhr_%dbpleq65.p' 
			%(NUM_VASO_BINS,NUM_FLUID_BINS,int(TIME_WINDOW),MAP_NUM_BELOW_THRESH),'rb'))

all_actions = []

for dat in all_disc_actions.values():
	all_actions.extend(dat['OVERALL_ACTION_ID'])

all_actions = np.array(all_actions)

cts = np.unique(all_actions,return_counts=True)

plt.close()
plt.hist(all_actions,NUM_ACTIONS,log=True)
plt.grid(alpha=.2)
plt.title('actions: 1-%d = fluid, 0 pressor; %d = 0 fluid, 1 pressor,...' %(NUM_FLUID_BINS-1,NUM_FLUID_BINS))
plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/allactionshist_%dhr_vaso%d_fluid%d_%dbpleq65.pdf' %(TIME_WINDOW,
	NUM_VASO_BINS,NUM_FLUID_BINS,MAP_NUM_BELOW_THRESH))

# plt.show()

# plt.close()
# plt.hist(all_actions[all_actions>0],NUM_ACTIONS-1)
# plt.title('actions: 1-%d = fluid, 0 pressor; %d = 0 fluid, 1 pressor,...')
# plt.show()


np.mean(all_actions>0) #21.3%



#histogram of all state vals
# sns.set(font_scale=.5)
# plt.close('all')
# plt.figure(figsize=(25,25))
# for num,i in enumerate(inds_to_process):
# 	vals = all_states[:,i]
# 	print(all_state_vars[i])
# 	print(np.round(np.percentile(vals,[0,.1,1,5,10,25,50,75,90,95,99,99.9,100]),2))

# 	plt.subplot(9,4,num+1)

# 	# plt.hist(vals,50)

# 	tmp = np.log(vals+.1)
# 	tmp = (tmp-np.mean(tmp))/np.std(tmp)
# 	tmp = sigmoid_squash(tmp)
# 	plt.hist(tmp,30)

# 	plt.title(all_state_vars[i])
# plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/all_cts_statevars_hists.pdf')
# plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_all_cts_statevars_hists.pdf')
# plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_std_all_cts_statevars_hists.pdf')
# plt.savefig(MIMIC_DATA_PATH+'data_cleaning/figs/log_std_squash_all_cts_statevars_hists.pdf')





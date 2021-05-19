 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 2019

Learn behavior policy for this dataset using annoy (approx kNN)

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
import annoy #approximate kNN 

# TODO: set this to the correct path where your preprocessed mimic data
# for RL for this hypotension task lives
MIMIC_DATA_PATH = "XXX"

pd.set_option("display.max_columns",101)
os.chdir(MIMIC_DATA_PATH)

state_weights = [
5, #normed_time
0, #age
0, #is_F
0, #surg_ICU
0, #is_not_white
0, #is_emergency
0, #is_urgent
0, #hrs_from_admit_to_icu
0, #bicarbonate
0, #bicarbonate_ind
0, #bun
0, #bun_ind
0, #creatinine
0, #creatinine_ind
0, #fio2
0, #fio2_ind
0, #glucose
0, #glucose_ind
0, #hct
0, #hct_ind
0, #hr
0, #hr_ind
5, #lactate
1, #lactate_ind
0, #magnesium
0, #magnesium_ind
0, #platelets
0, #platelets_ind
0, #potassium
0, #potassium_ind
0, #sodium
0, #sodium_ind
0, #spo2
0, #spo2_ind
0, #spontaneousrr
0, #spontaneousrr_ind
0, #temp
0, #temp_ind
5, #urine
1, #urine_ind
0, #wbc
0, #wbc_ind
0, #alt
0, #alt_ind
0, #ast
0, #ast_ind
0, #bilirubin_total
0, #bilirubin_total_ind
0, #co2
0, #co2_ind
0, #dbp
0, #dbp_ind
0, #hgb
0, #hgb_ind
5, #map
1, #map_ind
0, #pco2
0, #pco2_ind
0, #po2
0, #po2_ind
0, #sbp
0, #sbp_ind
0, #weight
0, #weight_ind
0, #gfr
0, #GCS
0, #GCS_ind
1, #lactate_8ind
1, #lactate_everind
0, #po2_8ind
0, #po2_everind
0, #pco2_8ind
0, #pco2_everind
0, #fio2_everind
0, #alt_everind
0, #ast_everind
0, #bilirubin_total_everind
5, #last_vaso_1
5, #last_vaso_2
5, #last_vaso_3
5, #last_vaso_4
5, #last_fluid_1
5, #last_fluid_2
5, #last_fluid_3
5, #total_all_prev_vasos
5, #total_all_prev_fluids
5, #total_last_8hrs_vasos
5, #total_last_8hrs_fluids
0, #last_reward
]

state_weights = np.array(state_weights)
#sqrt because rescale features by this, so weight in squared loss is original wt
state_weights = np.sqrt(state_weights) 

#################

NUM_VASO_BINS = 5
NUM_FLUID_BINS = 4
N_ACTIONS = NUM_VASO_BINS*NUM_FLUID_BINS
TIME_WINDOW = 1
MAP_NUM_BELOW_THRESH = 3

states_dat = pickle.load(open(MIMIC_DATA_PATH+'model_data/processed_finalfeatures_vaso%d_fluid%d_states_%dhr_%dbpleq65.p' %(
	NUM_VASO_BINS,NUM_FLUID_BINS,int(TIME_WINDOW),MAP_NUM_BELOW_THRESH),'rb'))
actions_dat = pickle.load(open(MIMIC_DATA_PATH+'model_data/actions_discretized_vaso%d_fluid%d_%dhr_%dbpleq65.p' 
			%(NUM_VASO_BINS,NUM_FLUID_BINS,int(TIME_WINDOW),MAP_NUM_BELOW_THRESH),'rb'))
rewards_dat = pickle.load(open(MIMIC_DATA_PATH+'model_data/rewards_%dhr_%dbpleq65.p' %(int(TIME_WINDOW),MAP_NUM_BELOW_THRESH),'rb'))

### print out all weights along with state vars...




final_ICU_IDs = np.array(list(states_dat.keys()))
n_ids_tot = len(final_ICU_IDs)

### train/test split
tr_perc = 1.0
n_tr_pat = int(n_ids_tot*tr_perc)
n_te_pat = n_ids_tot - n_tr_pat

seed = 12345
np.random.seed(seed)

tr_ids = final_ICU_IDs
# perm = np.random.permutation(n_ids_tot)
# tr_ids = final_ICU_IDs[perm[:n_tr_pat]]
# te_ids = final_ICU_IDs[perm[n_tr_pat:]]
#TODO: should probably sort to avoid issues later...

### train
all_tr_states = []
all_tr_actions = []
all_tr_ids = [] #tr_id for every single state; to map back and reconstruct later

for ID in tr_ids:
	# drop first column of state_dat (time); 
	# drop last row (no action associated; only used for getting final reward)
	# only keep action cols of actions after discretization (2:4)
	# s_dat = np.array(states_dat[ID])[:-1,1:]
	# a_dat = np.array(actions_dat[ID])[:-1,2:4] #use the last action at last time, along with current state in knn 

	# sa_dat = np.concatenate([s_dat,np.zeros((s_dat.shape[0],2))],1)
	# sa_dat[:,-2:] = a_dat #lining up [s0,0],[s1,a0],[s2,a1],...,[s_T-1,a_T-2] 
	# for 1 hour: sa_dat[:,-2:] = a_dat


	#drop first column of state_dat (time); 
	#drop last row (no action associated; only used for getting final reward)
	s_dat = np.array(states_dat[ID])[:-1,1:]
	all_tr_states.append(s_dat)

	next_actions = np.array(actions_dat[ID]['OVERALL_ACTION_ID'][1:])
	all_tr_actions.extend(next_actions)
	all_tr_ids.extend([ID]*len(next_actions))


all_tr_states = np.concatenate(all_tr_states,0)
all_tr_actions = np.array(all_tr_actions)
all_tr_ids = np.array(all_tr_ids)

assert all_tr_states.shape[0] == all_tr_actions.shape[0] == all_tr_ids.shape[0]

#normalize the cts columns (besides actions)
# tr_means = np.mean(all_tr_states,0)
# tr_sds = np.std(all_tr_states,0)

###TODO cache these for easy scaling in future

# pickle.dump([tr_means,tr_sds,state_cts_inds_norm],open('./model_data/state_means_sds_ctsinds_%dhr_bpleq65.p' %TIME_WINDOW,'wb'))

# all_tr_states[:,state_cts_inds_norm] = (all_tr_states[:,state_cts_inds_norm]-
	# tr_means[state_cts_inds_norm])/tr_sds[state_cts_inds_norm]


all_tr_states = all_tr_states*state_weights #reweight

################## ready to run kNN on train!

n_dim = all_tr_states.shape[1]

n_trees = 500 
#built in 6 min, 1 hr, 5/4

t = time()

knn = annoy.AnnoyIndex(n_dim,metric='euclidean')
for i in range(all_tr_states.shape[0]):
	knn.add_item(i, all_tr_states[i,:])

knn.build(n_trees)
print("built in %.1f" %(time()-t))



t = time()
all_action_probs = []
all_nn_actions_cts = []
NUM_NN = 100 #TODO: tune...??? try a few? 50, 100, 250, 500

for i in range(all_tr_states.shape[0]):
	if i%1000==0:
		print("%d / %d, took %.1f so far" %(i,all_tr_states.shape[0],time()-t))
	tmp = np.array(knn.get_nns_by_item(i,NUM_NN+1)[1:]) #exclude yourself. TODO: exclude all from same patient also??
	nn_actions = all_tr_actions[tmp]
	all_action_probs.append(np.mean(nn_actions==all_tr_actions[i]))
	all_nn_actions_cts.append(np.unique(nn_actions,return_counts=True))

print("matched all in %.1f" %(time()-t)) #

all_action_probs = np.array(all_action_probs)
all_nn_actions_cts = np.array(all_nn_actions_cts)

pickle.dump([all_action_probs,all_nn_actions_cts],open(MIMIC_DATA_PATH+'model_data/all_action_probs_%dnn_vaso%d_fluid%d_%dhr_%dbpleq65.p' 
	%(NUM_NN,NUM_VASO,NUM_FLUID,TIME_WINDOW,MAP_NUM_BELOW_THRESH),'wb'))

####### 	
#### convert all_nn_actions_cts into dict with all beh act probs for all acts...
#### 	useful if we want to do viz on the full behavior policy, incorporating *all* action probs
####	and not just action probs for the actions actually taken...
#######

all_behprob_acts = {}

starts = np.searchsorted(all_tr_ids,tr_ids,'left')
ends = np.searchsorted(all_tr_ids,tr_ids,'right')

for i,ID in enumerate(tr_ids):
	# WAT...
	#skip first action for each ID, as this is the action taken at 0, and we only want beh probs for all others
	# this_acts_cts = all_nn_actions_cts[(starts[i]+1):ends[i],:]

	this_acts_cts = all_nn_actions_cts[starts[i]:ends[i],:]

	tmp = np.zeros((this_acts_cts.shape[0],N_ACTIONS))
	for ii in range(this_acts_cts.shape[0]):
		tmp[ii,this_acts_cts[ii,0]] += 1/NUM_NN*this_acts_cts[ii,1]

	all_behprob_acts[ID] = tmp

pickle.dump(all_behprob_acts,open(MIMIC_DATA_PATH+'model_data/all_behprobs_allactions_%dnn_vaso%d_fluid%d_%dhr_%dbpleq65.p' 
	%(NUM_NN,NUM_VASO,NUM_FLUID,TIME_WINDOW,MAP_NUM_BELOW_THRESH),'wb'))

############# write out behavior action probs and use as ground truth

act_probs = pickle.load(open(MIMIC_DATA_PATH+'model_data/all_action_probs_%dnn_vaso%d_fluid%d_%dhr_%dbpleq65.p' 
	%(NUM_NN,NUM_VASO,NUM_FLUID,TIME_WINDOW,MAP_NUM_BELOW_THRESH),'rb'))
probs = act_probs[0]
# a moderate eps for settings with 0 prob to avoid numeric issues
probs[probs==0] = 0.002 #prev: .01 for 50 nn's

all_acts_with_probs = {}
starts = np.searchsorted(all_tr_ids,final_ICU_IDs,'left')
ends = np.searchsorted(all_tr_ids,final_ICU_IDs,'right')

for ii,ID in enumerate(final_ICU_IDs):
	if ii%100==99:
		print('%d/%d' %(ii,len(final_ICU_IDs)))

	inds = np.arange(starts[ii],ends[ii])
	this_probs = probs[inds]
	# this_acts = all_tr_actions[inds]

	act_dat = actions_dat[ID].iloc[1:,:] #cut first row since 
	act_dat = act_dat.assign(ACT_PROBS=this_probs) 

	all_acts_with_probs[ID] = act_dat


pickle.dump(all_acts_with_probs,open(MIMIC_DATA_PATH+'model_data/actions_discretized_withprobs_%dnn_vaso%d_fluid%d_%dhr_%dbpleq65.p'
		%(NUM_NN,NUM_VASO,NUM_FLUID,TIME_WINDOW,MAP_NUM_BELOW_THRESH),'wb'))


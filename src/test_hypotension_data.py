#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 13:41:18 2018

First experiment for joint learning of HMM model with POMDP policy.
Goal of experiment is to test effect of varying number of dimensions in 
our toy environment (for now, tiger and a chain-world).

Tons of tweaks and modifications to try out different bits during learning...

@author: josephfutoma
"""

import os
import sys
import pickle
import copy
import argparse
from time import time

import autograd
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad as vg
from autograd import make_vjp
from autograd.scipy.misc import logsumexp
from autograd.misc.flatten import flatten_func,flatten
import autograd.scipy.stats as stat

from sklearn.cluster import KMeans

#import matplotlib.pyplot as plt

from util import *
from util_hypotension import *
from pbvi_cts import *
from action_hmm_cts import *     
from OPE_funcs import * 

#####
##### helper funcs
#####

def update_and_write_savedict(save_dict):
    #### save out to a dict
    save_dict['objs'] = objs
    save_dict['RL_objs'] = RL_objs
    save_dict['HMM_objs'] = HMM_objs

    save_dict['grad_norms'] = grad_norms
    save_dict['RL_grad_norm'] = RL_grad_norm
    save_dict['HMM_grad_norm'] = HMM_grad_norm
                    
    save_dict['HMM_te_objs'] = HMM_te_objs
    save_dict['grad_norms_HMM_te'] = grad_norms_HMM_te    

    save_dict['te_policy_val'] = te_policy_val
    # save_dict['te_policy_val_noprune'] = te_policy_val_noprune

    save_dict['tr_ESS'] = tr_ESS
    # save_dict['tr_ESS_noprune'] = tr_ESS_noprune
    save_dict['tr_CWPDIS'] = tr_CWPDIS
    save_dict['tr_CWPDIS_obj'] = tr_CWPDIS_obj
    # save_dict['tr_CWPDIS_obj_noprune'] = tr_CWPDIS_obj_noprune

    save_dict['te_ESS'] = te_ESS
    # save_dict['te_ESS_noprune'] = te_ESS_noprune

    save_dict['te_CWPDIS'] = te_CWPDIS

    save_dict['tracked_params'] = tracked_params         
    save_dict['tracked_Vs'] = tracked_Vs            
    save_dict['tracked_Bs'] = tracked_Bs   

    save_dict['params'] = params   

    try:
        print("saving!")
        with open(RESULTS_PATH+model_string+'.p','wb') as f:
            pickle.dump(save_dict, f) 
    except:
        print("save failed!!")
    return save_dict   


#####
##### funcs to get param inits
#####

def params_init_random(alpha_pi=25,alpha_T=25):
    """ 
    random initialization
    """
    T = np.zeros((n_S,n_S,n_A))   
    for s in range(n_S):
        for a in range(n_A):
            T[:,s,a] = np.random.dirichlet(alpha_T*np.ones(n_S))
       
    pi = np.random.dirichlet(alpha_pi*np.ones(n_S))
    
    #random obs model; assumes data already standardized
    O_means = np.random.normal(0,1,(n_dim,n_S,n_A))
    O_sds = np.random.normal(1,.25,(n_dim,n_S,n_A))
    n_inds = np.sum(O_sds<=0.1)
    while n_inds>0:
        O_sds[O_sds<=0.1] = np.random.normal(1,.25,n_inds)
        n_inds = np.sum(O_sds<=0.1)
    O = (O_means,O_sds)
    
    #for now, assume R is normal with unknown means and known, small variances (eg .1)
    nz_rew = rewards_tr[np.logical_not(np.isinf(rewards_tr))]
    R = np.random.normal(np.mean(nz_rew),np.std(nz_rew),(n_S,n_A))
    n_inds = np.sum(R>1) #truncate R to be < 1
    while n_inds>0:
        R[R>1] = np.random.normal(np.mean(nz_rew),np.std(nz_rew),n_inds)
        n_inds = np.sum(R>1) 

    params = (pi,T,O,R)
    return params


def params_init_MAP_sep(map_ind=0,M_step_with_missing=False):

    #helper stuff
    map_log1p_mean = 4.28748298
    map_log1p_sd = 0.1783257 

    def back_transform(maps):
        return np.exp(maps*map_log1p_sd+map_log1p_mean)-1 
    def transform(raw_maps):
        return (np.log(1+raw_maps)-map_log1p_mean)/map_log1p_sd

    ## bin MAPs...
    all_maps = np.reshape(observs_tr[:,:,map_ind],-1)
    all_maps = all_maps[np.logical_not(np.isinf(all_maps))]

    all_raw_maps = np.exp(all_maps*map_log1p_sd + map_log1p_mean) - 1

    # TODO: way to automate this???
    #for now just manually define the n_S-1 bin edges

    if n_S == 5:
        qs = [5,10,15,30]        
    elif n_S == 10:
        qs = [1,2.5,5,7.5,10,15,20,30,60]
    elif n_S == 15:
        qs = [0.5,2,4,6,8,10,12,14,16,18,20,25,30,65]

    map_bins = np.percentile(all_maps,qs)

    #OLD
    # qs = np.linspace(0,100,n_S+1)[1:-1]

    #now use these states defined by MAP separation to filter
    max_T = rewards_tr.shape[1]
    gam = np.zeros((max_T+1,n_S,N)) #sstats for states
    E_Njka = .01*np.ones((n_S,n_S,n_A)) #sstats for trans

    #edge case at beginning when separating on MAP / obs
    this_map = init_observs_tr[:,map_ind]
    this_state = np.searchsorted(map_bins,this_map)
    gam[0,this_state,np.arange(N)] = 1

    for t in range(max_T):
        mask = np.logical_not(np.isinf(observs_tr[:,t,map_ind])) #only get obs actually go this long
        this_map = observs_tr[mask,t,map_ind]
        this_state = np.searchsorted(map_bins,this_map)
        gam[t+1,this_state,mask] = 1

    for n in range(N):
        this_state = np.where(gam[:,:,n])[1]
        for t in range(int(seq_lens_tr[n])):
            E_Njka[this_state[t],this_state[t+1],actions_tr[n,t]] += 1

    #run M step... decide if want to use missing masks or not...
    params = params_init_random()
    nat_params = to_natural_params(params)

    if M_step_with_missing:
        pi,T,O,R = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,E_Njka,
            init_observs = init_observs_tr,
            init_actions = init_actions_tr,
            observs_missing_mask = observs_mask_tr,
            init_observs_missing_mask = init_observs_mask_tr)
    else:
        pi,T,O,R = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,E_Njka)
    
    params = (pi,T,O,R)

    return params



#####
##### funcs for the joint learning
#####


def init_B_and_V(params):
    V_min = 0
    B = initialize_B(params,V_min,gamma,n_expandB_iters=min(int(n_S*2),50)) #max out at...S*4 previously...
    ## TODO: put a check to see if B is too big, else drop the last few rows??
    n_B = B.shape[0]
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]
    return V,B


def params_init(param_init,**kwargs):
    #### get initialization...

    if param_init == 'random':
        params = params_init_random(**kwargs)
    if param_init == 'MAP-sep':
        params = params_init_MAP_sep(**kwargs)

    return params


def get_param_inits(param_init,n_PBVI_iters=20):
    """
    helper func, to get a bunch of inits by different means and then 
    test how well they do, and choose the best to run with
    """
    # inits = np.array(['random','kmeans','EM-random','EM-kmeans','reward-sep','BP-sep'])

    restarts_per_init = {
    'random': 10, 
    'MAP-sep': 10
    }
    n_restarts = restarts_per_init[param_init]

    lls_tr = []
    polvals_tr = []
    ESS_tr = []
    objs = []

    best_obj = np.inf #will select the best init based on PC objective: HMM_obj + lambda*RL_obj
    best_EM_obj = np.inf

    for restart in range(n_restarts):

        if param_init=='MAP-sep':
            #for MAP-sep init, try M step with & without missing data half the time...
            params = params_init(param_init,M_step_with_missing=restart>=n_restarts/2)
        else:
            params = params_init(param_init)


        nat_params = to_natural_params(params)
        pi,T,O,R = params

        print("learning policy for restart %d" %restart,flush=True)
        V,B = init_B_and_V(params)
        for ii in range(n_PBVI_iters):
            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,eps=.001,PBVI_temps=[.01,.01,.01],n_samps=100)

        #check value of init:
        #   - log-lik of HMM on test data 
        #   - val of learned policy

        ll = MAP_objective(nat_params,observs_tr,actions_tr,
            init_observs_tr,init_actions_tr,
            observs_mask_tr,init_observs_mask_tr)
        lls_tr.append(-ll)

        all_beliefs_tr = get_beliefs(params,seq_lens_tr,actions_tr,observs_tr,
            init_observs_tr,init_actions_tr,observs_mask_tr,init_observs_mask_tr)
        RL_obj,(_,CWPDIS_obj,ESS,_,_,_,_) = softmax_policy_value_objective_term(
            nat_params,R,V,B,
            action_probs_tr,beh_probs_tr,actions_tr,init_actions_tr,observs_tr,
            init_observs_tr,observs_mask_tr,init_observs_mask_tr,
            rewards_tr,seq_lens_tr,gamma,
            cached_beliefs=all_beliefs_tr,update_V = False,
            gr_safety_thresh=gr_safety_thresh,prune_num=0,
            ESS_penalty=ESS_penalty)

        polvals_tr.append(CWPDIS_obj)
        ESS_tr.append(ESS)

        ###
        ### based on current lambda, select the best overall objective
        ###

        if lambd == np.inf:
            obj = log_prior(nat_params) + 1e8*RL_obj
        else:
            obj = ll + lambd*RL_obj
        objs.append(obj)

        if obj < best_obj:
            best_obj = obj
            best_nat_params = nat_params
            best_params = params
            best_V = V
            best_B = B

            best_te_ll = -MAP_objective(best_nat_params,observs_te,actions_te,
                init_observs_te,init_actions_te,observs_mask_te,init_observs_mask_te)
            all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                init_observs_te,init_actions_te,observs_mask_te,init_observs_mask_te)
            _,(_,CWPDIS_obj,ESS,_,_,_,_) = softmax_policy_value_objective_term(
                nat_params,R,V,B,
                action_probs_te,beh_probs_te,actions_te,init_actions_te,observs_te,
                init_observs_te,observs_mask_te,init_observs_mask_te,
                rewards_te,seq_lens_te,gamma,
                cached_beliefs=all_beliefs_te,update_V = False,
                gr_safety_thresh=gr_safety_thresh,prune_num=0)

            save_dict['best_init_params'] = best_params
            save_dict['best_init_natparams'] = best_nat_params
            save_dict['best_restart_ind'] = restart
            save_dict['best_obj'] = best_obj
            save_dict['best_V_init'] = best_V
            save_dict['best_B_init'] = best_B
            save_dict['best_init_te_ESS'] = ESS
            save_dict['best_init_te_ll'] = best_te_ll
            save_dict['best_init_te_polval'] = CWPDIS_obj

    #init stuff in case we want to check them later
    save_dict['init_objs'] = objs
    save_dict['init_lls_tr'] = lls_tr
    save_dict['init_polvals_tr'] = polvals_tr
    save_dict['init_ESS_tr'] = ESS_tr

    return best_params,best_V,best_B


if __name__ == "__main__":    

    ##########
    ########## parse args & setup
    ########## 

    parser = argparse.ArgumentParser()
    #paths default to local if not specified
    parser.add_argument('--data_path', default='/Users/josephfutoma/Dropbox/research/mimic_data/hypotension_management/model_data/POPCORN_9obs-logstd-inds_alldata.p')
    parser.add_argument('--results_path')
    parser.add_argument('--logs_path')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--var_set', default='all')
    parser.add_argument('--log_lambd', type=float, default=0)
    parser.add_argument('--param_init', default='random')
    parser.add_argument('--num_states', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)

    parser.add_argument('--prune_num', type=int, default=0)
    parser.add_argument('--ESS_penalty', type=float, default=0)
    parser.add_argument('--gr_safety_thresh', type=float, default=0)

    #now unpack args
    arg_dict = vars(parser.parse_args())

    DATA_PATH = arg_dict['data_path']
    LOGS_PATH = arg_dict['logs_path']
    RESULTS_PATH = arg_dict['results_path']

    seed = arg_dict['seed']
    var_set = arg_dict['var_set']
    lambd = np.power(10,arg_dict['log_lambd'])
    param_init = arg_dict['param_init']
    n_S = arg_dict['num_states']
    fold = arg_dict['fold']
    prune_num = arg_dict['prune_num']
    ESS_penalty = arg_dict['ESS_penalty']
    gr_safety_thresh = arg_dict['gr_safety_thresh']

    env_name = 'hypotension' 
    model_string = '%s_nS%d_lambd%.8f_vars-%s_init-%s_prune%d_ESSpenalty%.2f_gr-safety-thresh-%.2f_seed%d_fold%d' %(
                env_name,n_S,lambd,var_set,param_init,prune_num,ESS_penalty,gr_safety_thresh,seed,fold)

    #redirect stdout/stderr when remote
    if LOGS_PATH is not None:
        sys.stdout = open(LOGS_PATH+model_string+"_out.txt","w")
        sys.stderr = open(LOGS_PATH+model_string+"_err.txt","w")
    print("starting! "+model_string,flush=True)

    ##########
    ########## Load in data, setup job params
    ##########

    #setup params
    n_A = 20
    gamma = 0.999
    n_folds = 5

    np.set_printoptions(threshold=10000)                                                      
    np.random.seed(seed)
    
    #####
    #load data and setup train/test split
    #####
    all_obs_dim = 9 #9 labs/vitals at most
    all_dat = pickle.load(open(DATA_PATH,'rb'))
    all_ids = np.array(list(all_dat.keys()))
    var_names = list(all_dat.values())[0].columns
    var_names = np.array(var_names[:all_obs_dim]) 

    N_tot = len(all_ids)
    fold_ids = np.arange(N_tot) % n_folds

    tr_inds = fold_ids != fold
    te_inds = fold_ids == fold
    N = int(np.sum(tr_inds))
    Nte = int(np.sum(te_inds))

    rng = np.random.RandomState(711) #fixed train/test sets across folds!
    perm = rng.permutation(N_tot)

    ids_tr = all_ids[perm[tr_inds]]
    ids_te = all_ids[perm[te_inds]]

    (init_observs_te,observs_te,init_observs_mask_te,observs_mask_te,
        rewards_te,init_actions_te,actions_te,action_probs_te,
        beh_probs_te,seq_lens_te) = get_padded_databatch_from_IDs(
        ids_te,all_dat,var_names,n_A)
    (init_observs_tr,observs_tr,init_observs_mask_tr,observs_mask_tr,
        rewards_tr,init_actions_tr,actions_tr,action_probs_tr,
        beh_probs_tr,seq_lens_tr) = get_padded_databatch_from_IDs(
        ids_tr,all_dat,var_names,n_A)

    ##### do a bit of clipping on MAP values, limit outlier upper values...
    MAP_THRESH = 2
    init_observs_te[:,0] = np.clip(init_observs_te[:,0],a_min=None,a_max=MAP_THRESH)
    observs_te[:,:,0] = np.clip(observs_te[:,:,0],a_min=None,a_max=MAP_THRESH)
    init_observs_tr[:,0] = np.clip(init_observs_tr[:,0],a_min=None,a_max=MAP_THRESH)
    observs_tr[:,:,0] = np.clip(observs_tr[:,:,0],a_min=None,a_max=MAP_THRESH)
    #####

    #subset vars if only using a subset...
    if var_set == 'map':
        n_dim = 1 

        init_observs_te = init_observs_te[:,[0]]
        observs_te = observs_te[:,:,[0]]
        init_observs_mask_te = init_observs_mask_te[:,[0]]
        observs_mask_te = observs_mask_te[:,:,[0]]

        init_observs_tr = init_observs_tr[:,[0]]
        observs_tr = observs_tr[:,:,[0]]
        init_observs_mask_tr = init_observs_mask_tr[:,[0]]
        observs_mask_tr =  observs_mask_tr[:,:,[0]]

    if var_set == 'map-urine-lactate':
        n_dim = 3

        init_observs_te = init_observs_te[:,:3]
        observs_te = observs_te[:,:,:3]
        init_observs_mask_te = init_observs_mask_te[:,:3]
        observs_mask_te = observs_mask_te[:,:,:3]

        init_observs_tr = init_observs_tr[:,:3]
        observs_tr = observs_tr[:,:,:3]
        init_observs_mask_tr = init_observs_mask_tr[:,:3]
        observs_mask_tr =  observs_mask_tr[:,:,:3]

    if var_set == 'all':
        n_dim = 9

    var_names = np.array(var_names[:n_dim]) 

    #up the 0-NN cases probs by a little bit (pretty rare anyways...)
    action_probs_tr[action_probs_tr==.001] = .01
    action_probs_te[action_probs_te==.001] = .01

    te_returns = []
    for i in range(Nte):    
        te_returns.append(np.sum(np.power(gamma,np.arange(seq_lens_te[i]))*rewards_te[i,:seq_lens_te[i]]))
    te_returns = np.array(te_returns)
    print('fold %d, est test set beh policy value: %.5f' %(fold,np.mean(te_returns)),flush=True)
    # test set avg returns: 48.86755, 48.44530, 48.42580, 47.65438, 48.23278: 48.33 overall 

    ### learning params
    n_epochs = 20000
    batchsize = N 
    lr = 1e-3
    optim = 'rprop'
    PBVI_train_update_iters = 1
    
    save_dict = {}

    ##########
    ########## Given an init, start PC learning
    ##########

    params,V,B = get_param_inits(param_init)
    nat_params = to_natural_params(params)
    pi,T,O,R = params

    ##### setup gradient functions
    ## explicitly split our objective into HMM term and RL term so we can 
    ## track each value and gradients separately
    
    RLobj_V_g = value_and_output_and_grad(softmax_policy_value_objective_term)
    Prior_obj_g = vg(log_prior)
    HMMobj_g = vg(MAP_objective)
        
    flat_nat_params,unflatten = flatten(nat_params)

    #store progress
    #TODO: func to hide this setup...but still keep these in global namespace?? class to cache all this & just save object??
    objs = []
    RL_objs = []
    HMM_objs = []
    
    grad_norms = []
    RL_grad_norm = []
    HMM_grad_norm = []
     
    HMM_te_objs = []
    grad_norms_HMM_te = []
    
    te_policy_val = []
    te_policy_val_noprune = []

    tr_ESS = []
    tr_ESS_noprune = []
    tr_CWPDIS = []
    tr_CWPDIS_obj = []
    tr_CWPDIS_obj_noprune = []

    te_ESS = []
    te_ESS_noprune = []
    te_CWPDIS = []

    tracked_params = []
    tracked_Vs = []
    tracked_Bs = []
    tracked_Bs.append(B)

    #init rprop stuff
    tot_iter = 0
    last_v = np.inf #last objective value
    last_g = np.zeros(len(flat_nat_params))
    step_sizes = lr*np.ones(len(flat_nat_params))
    last_steps = np.zeros(len(flat_nat_params))

    for epoch in range(n_epochs):
        print("starting epoch %d" %epoch,flush=True)

        for n_iter in range(N//batchsize):   
            t = time()

            (this_init_observs,this_observ,this_init_observs_mask,this_observs_mask,
                this_rew,this_init_act,this_act,this_act_probs,this_beh_probs,
                this_seq_lens) = (init_observs_tr,observs_tr,init_observs_mask_tr,observs_mask_tr,
                rewards_tr,init_actions_tr,actions_tr,action_probs_tr,beh_probs_tr,seq_lens_tr)      

            this_nat_params = nat_params

            #####
            ##### HMM objective
            #####

            if lambd == np.inf:
                HMM_obj,HMM_grad = Prior_obj_g(this_nat_params)
            else: 
                HMM_obj,HMM_grad = HMMobj_g(this_nat_params,this_observ,
                    this_act,this_init_observs,this_init_act,
                    this_observs_mask,this_init_observs_mask)
            
            HMM_grad = flatten(HMM_grad)[0]
            
            #####
            ##### RL objective 
            #####
            
            if lambd > 0:

                RL_obj,(V,CWPDIS_obj,ESS,CWPDIS_nums,
                    CWPDIS_denoms,ESS_noprune,
                    CWPDIS_obj_noprune),RL_grad = RLobj_V_g(this_nat_params,R,V,B,
                    this_act_probs,this_beh_probs,this_act,this_init_act,this_observ,
                    this_init_observs,this_observs_mask,this_init_observs_mask,
                    this_rew,this_seq_lens,gamma,
                    gr_safety_thresh=gr_safety_thresh,
                    PBVI_update_iters=PBVI_train_update_iters,
                    update_V=True,V_penalty=1e-6,
                    prune_num=prune_num,ESS_penalty=ESS_penalty) 

                V = [V[0]._value,V[1]._value]

                if lambd == np.inf:
                    RL_obj *= 1e8
                    RL_grad = flatten(RL_grad)[0]*1e8
                else:
                    RL_obj *= lambd
                    RL_grad = flatten(RL_grad)[0]*lambd  

                #save RL stuff if computing during opt anyways
                RL_objs.append(RL_obj)
                RL_grad_norm.append(np.sum(np.abs(RL_grad)))

                tr_ESS.append(ESS._value) 
                # tr_ESS_noprune.append(ESS_noprune._value)
                # tr_CWPDIS.append((CWPDIS_nums._value, CWPDIS_denoms._value))
                # tr_CWPDIS_obj_noprune.append(CWPDIS_obj_noprune._value)
                tr_CWPDIS_obj.append(CWPDIS_obj._value)

            else:
                RL_obj = 0
                RL_grad = np.zeros(HMM_grad.shape)

            g = RL_grad + HMM_grad
            v = RL_obj + HMM_obj

            # g = np.clip(g,-1e4,1e4)
            
            #save stuff
            objs.append(v)
            grad_norms.append(np.sum(np.abs(g)))

            HMM_objs.append(HMM_obj)
            HMM_grad_norm.append(np.sum(np.abs(HMM_grad)))

            #apply gradient!
            flat_nat_params,last_g,step_sizes,last_steps = rprop(flat_nat_params,g,last_g,step_sizes,last_steps,v,last_v)
            last_v = v

            pi,T,O,R = to_params(unflatten(flat_nat_params))
            params = (pi,T,O,R)            
            nat_params = to_natural_params(params)

            #update R separately via E step 
            _,gam = forward_backward_Estep(nat_params,observs_tr,actions_tr,rewards_tr,
                get_xi=False,init_observs=init_observs_tr,
                init_actions=init_actions_tr,
                observs_missing_mask=observs_mask_tr,
                init_observs_missing_mask=init_observs_mask_tr)
            R = M_step_just_reward(nat_params,observs_tr,actions_tr,rewards_tr,gam)

            params = (pi,T,O,R)            
            nat_params = to_natural_params(params)
            flat_nat_params,unflatten = flatten(nat_params)

            #####
            ##### End of learning iteration, now do some checks every so often...
            #####

            tot_iter += 1
            if tot_iter%1==0:
                print("epoch %d, iter %d, RL obj %.4f, HMM obj %.4f, total obj %.4f grad L1-norm %.4f, took %.2f" 
                      %(epoch,tot_iter,RL_obj,HMM_obj,v,np.sum(np.abs(g)),time()-t),flush=True)

            #every so often, check test set
            if tot_iter % 100 == 1 or tot_iter==n_epochs:                              

                ##### check HMM performance on held-out test set
                HMM_obj,HMM_grad = HMMobj_g(nat_params,observs_te,actions_te,
                    init_observs_te,init_actions_te,
                    observs_mask_te,init_observs_mask_te)  
                print("HMM objective on test data %.4f, grad norm %.4f" 
                      %(-HMM_obj,np.sum(np.abs(flatten(HMM_grad)[0]))),flush=True)
                HMM_te_objs.append(-HMM_obj)
                grad_norms_HMM_te.append(np.sum(np.abs(flatten(HMM_grad)[0])))

                ### and check the policy...
                all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                    init_observs_te,init_actions_te,observs_mask_te,init_observs_mask_te)

                if lambd > 0:
                    _,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                        ESS_noprune,CWPDIS_obj_noprune) = softmax_policy_value_objective_term(
                        nat_params,R,V,B,
                        action_probs_te,beh_probs_te,actions_te,init_actions_te,observs_te,
                        init_observs_te,observs_mask_te,init_observs_mask_te,
                        rewards_te,seq_lens_te,gamma,
                        cached_beliefs=all_beliefs_te,
                        update_V = False,
                        gr_safety_thresh=gr_safety_thresh,
                        prune_num=prune_num,
                        ESS_penalty=ESS_penalty)

                    print('iter %d, est value of policy on test data: %.5f' %(tot_iter,CWPDIS_obj),flush=True)
                    te_policy_val.append(CWPDIS_obj)
                    # te_policy_val_noprune.append(CWPDIS_obj_noprune)
                    te_ESS.append(ESS)
                    # te_ESS_noprune.append(ESS_noprune)
                    te_CWPDIS.append((CWPDIS_nums,CWPDIS_denoms))

                    tracked_params.append(params)
                    tracked_Vs.append(V)

                #treat the 2 stage case separately...
                if lambd==0:
                    pi,T,O,R = params

                    V,B = init_B_and_V(params)

                    tracked_Bs.append(B)
                    tracked_params.append(params)

                    this_V = []
                    this_te_policy_val = []
                    this_te_ESS = []

                    for _ in range(3):

                        #first, update V
                        for _ in range(10):
                            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,eps=.001,
                                PBVI_temps=[.01,.01,.01])
                        this_V.append(V)

                        #then, check policy stuff
                        _,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                            ESS_noprune,CWPDIS_obj_noprune) = softmax_policy_value_objective_term(
                            nat_params,R,V,B,
                            action_probs_te,beh_probs_te,actions_te,init_actions_te,observs_te,
                            init_observs_te,observs_mask_te,init_observs_mask_te,
                            rewards_te,seq_lens_te,gamma,
                            cached_beliefs=all_beliefs_te,
                            update_V = False,
                            gr_safety_thresh=gr_safety_thresh,
                            prune_num=0,
                            ESS_penalty=0)
                        this_te_policy_val.append(CWPDIS_obj_noprune)
                        this_te_ESS.append(ESS_noprune)

                    te_policy_val.append(this_te_policy_val)
                    te_ESS.append(this_te_ESS)
                    tracked_Vs.append(this_V)

                ### save
                save_dict = update_and_write_savedict(save_dict)           


            #refresh belief set & refresh V every so often...
            if lambd > 0 and tot_iter == 250: #only refresh for not 2 stage
                print("getting new belief set...",flush=True)
                pi,T,O,R = params

                V,B = init_B_and_V(params)

                for _ in range(10):
                    V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,eps=.001,
                        PBVI_temps=[.01,.01,.01])
                print("setup beliefs and V...",flush=True)

                tracked_Bs.append(B)

                          

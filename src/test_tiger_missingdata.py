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
import itertools
import pickle
import copy
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
from pbvi_cts import *
from action_hmm_cts import *
from envs_cts import *
from OPE_funcs import *
       

def update_and_write_savedict(save_dict):
    #### save out to a dict
    save_dict['objs'] = objs
    save_dict['RL_objs'] = RL_objs
    save_dict['HMM_objs'] = HMM_objs

    # save_dict['RL_grads'] = RL_grads
    # save_dict['HMM_grads'] = HMM_grads

    save_dict['grad_norms'] = grad_norms
    save_dict['RL_grad_norm'] = RL_grad_norm
    save_dict['HMM_grad_norm'] = HMM_grad_norm
                    
    save_dict['HMM_te_objs'] = HMM_te_objs
    save_dict['grad_norms_HMM_te'] = grad_norms_HMM_te    

    save_dict['est_te_policy_val'] = est_te_policy_val
    # save_dict['est_te_policy_val_np'] = est_te_policy_val_np

    save_dict['avg_te_returns_det'] = avg_te_returns_det
    # save_dict['quant_te_returns_det'] = quant_te_returns_det

    # save_dict['tr_ESS'] = tr_ESS
    # save_dict['tr_ESS_noprune'] = tr_ESS_noprune
    save_dict['tr_CWPDIS_obj'] = tr_CWPDIS_obj
    # save_dict['tr_CWPDIS_obj_noprune'] = tr_CWPDIS_obj_noprune

    # save_dict['te_ESS_noprune'] = te_ESS_noprune

    # save_dict['tracked_params'] = tracked_params         
    # save_dict['tracked_Vs'] = tracked_Vs            

    save_dict['params'] = params   

    try:
        print("saving!")
        sys.stdout.flush()    
        with open(OUT_PATH+RESULTS_FOLDER+model_string+'.p','wb') as f:
            pickle.dump(save_dict, f) 
    except:
        print("save failed!!")
        sys.stdout.flush()  

    return save_dict   


def simdata_random_policy(N,pis,Ts,Os,Rs,min_traj_len,max_traj_len,listen_prob,rng=None):
    """
    create a bunch of simulated trajectories for tiger, taking random actions
    """
    if rng is None:
        rng = np.random

    n_env = len(Os)
    n_dim_per_env = np.shape(Os[0][0])[0]
    n_dim = n_env*n_dim_per_env 

    n_A = Rs[0].shape[1]

    seq_lens = max_traj_len*np.ones(N,dtype="int") #traj_len is max length
    observs = -np.inf*np.ones((N,max_traj_len,n_dim),dtype='int')
    rewards = -np.inf*np.ones((N,max_traj_len))
    actions = -1*np.ones((N,max_traj_len),dtype='int')
    action_probs = -1.0*np.ones((N,max_traj_len))

    #initially, listen for min_traj_len steps before eventually maybe opening a door
    action_prob_init = np.zeros(n_A)
    action_prob_init[-1] = 1.0

    #after min_traj_len steps, listen with this prob, else open a door
    # listen_prob = .5
    action_prob = (1-listen_prob)/(n_A-1)*np.ones(n_A)
    action_prob[-1] = listen_prob
                
    for n in range(N):      
        t = 1
        prob = action_prob if t >= min_traj_len else action_prob_init
        actions[n,0] = draw_discrete(prob,rng=rng)
        action_probs[n,0] = prob[actions[n,0]]

        #loop over dims 
        last_states = []
        next_states = []
        this_observs = []
        for i,(pi,T,O) in enumerate(zip(pis,Ts,Os)):
            last_states.append(draw_discrete(pi,rng=rng))
            next_states.append(draw_discrete(T[:,last_states[i],actions[n,0]],rng=rng))
            o = rng.normal(0,1,n_dim_per_env)
            this_observs.append(O[0][:,next_states[i],actions[n,0]]+O[1][:,next_states[i],actions[n,0]]*o)

        observs[n,0,:] = this_observs        
        rewards[n,0] = Rs[0][last_states[0],actions[n,0]] #NOTE: s_0 now!
        last_states = next_states
        if tiger_env and actions[n,0]!=n_A-1: #if chose and didn't listen, end for tiger 
            seq_lens[n] = t
            continue
        
        for t in range(1,seq_lens[n]):
            prob = action_prob if t+1 >= min_traj_len else action_prob_init
            actions[n,t] = draw_discrete(prob,rng=rng)
            action_probs[n,t] = prob[actions[n,t]]

            next_states = []
            this_observs = []
            for i,(pi,T,O) in enumerate(zip(pis,Ts,Os)):
                next_states.append(draw_discrete(T[:,last_states[i],actions[n,t]],rng=rng))
                o = rng.normal(0,1,n_dim_per_env)
                this_observs.append(O[0][:,next_states[i],actions[n,t]]+O[1][:,next_states[i],actions[n,t]]*o)

            observs[n,t,:] = this_observs        
            rewards[n,t] = Rs[0][last_states[0],actions[n,t]] #NOTE: s_t now!

            last_states = next_states
            if tiger_env and actions[n,t]!=n_A-1: #if chose and didn't listen, end for tiger 
                seq_lens[n] = t+1
                break

    max_T = np.max(seq_lens)
    observs = observs[:,:max_T,:]
    rewards = rewards[:,:max_T]
    actions = actions[:,:max_T]
    action_probs = action_probs[:,:max_T]

    return observs,rewards,actions,action_probs,seq_lens


def params_init_random(n_A,n_S,O_dims,alpha_pi=25,alpha_T=1):
    """ 
    random initialization
    """
    T = np.zeros((n_S,n_S,n_A))   
    for s in range(n_S):
        for a in range(n_A):
            T[:,s,a] = np.random.dirichlet(alpha_T*np.ones(n_S))
       
    pi = np.random.dirichlet(alpha_pi*np.ones(n_S))
    
    O_means = np.random.normal(np.mean(O_means_wit),np.std(O_means_wit),(O_dims,n_S,n_A))
    
    #draw O_sds; moment match in an unconstrained space
    mean_reals_Osdswit = np.mean(inv_softplus(O_sds_wit))
    std_reals_Osdswit = np.std(inv_softplus(O_sds_wit))+.01
    
    O_sds = softplus(np.random.normal(mean_reals_Osdswit,std_reals_Osdswit,(O_dims,n_S,n_A)))
    
    O = (O_means,O_sds)
    
    #for now, assume R is normal with unknown means and known, small variances (eg .1)
    R = np.random.normal(np.mean(R_wit),np.std(R_wit),(n_S,n_A))
    
    return pi,T,O,R


def params_init_kmeans(n_S,n_A,n_dim,rewards_tr,actions_tr,observs_tr,seq_lens_tr,alpha_pi=25,alpha_T=25,seed=None):
    """ 
    random initialization
    """

    #roughly uniform
    pi = np.random.dirichlet(alpha_pi*np.ones(n_S))

    #transitions: sticky, usually stay in same state
    T = np.zeros((n_S,n_S,n_A))   
    for s in range(n_S):
        alpha_vec = np.ones(n_S)
        alpha_vec[s] = alpha_T
        T[:,s,:] = np.random.dirichlet(alpha_vec)[:,None]

    ##### trickiest: obs

    #first, get all observations 
    Tt = np.sum(seq_lens_tr)
    all_rewards = np.zeros(Tt)
    all_obs = np.zeros((Tt,n_dim))
    all_acts = np.zeros(Tt)

    ct = 0
    for i in range(N):
        all_rewards[ct:ct+seq_lens_tr[i]] = rewards_tr[i,:seq_lens_tr[i]]
        all_acts[ct:ct+seq_lens_tr[i]] = actions_tr[i,:seq_lens_tr[i]]        
        all_obs[ct:ct+seq_lens_tr[i],:] = observs_tr[i,:seq_lens_tr[i],:]
        ct += seq_lens_tr[i]   

    ### fit kmeans to obs
    #TODO: do we want to do a different kmeans for obs from each action, to better estimate p(o|s,a)??
    #       are there label switching issues we'd need to be aware of...??

    ### treat all actions together
    kmeans = KMeans(n_clusters=n_S,random_state=seed).fit(all_obs)

    O_means = np.tile(kmeans.cluster_centers_.T[:,:,None],(n_A)) #n_dim x S x A
    O_sds = np.zeros((n_dim,n_S,n_A))

    for s in range(n_S):
        inds = kmeans.labels_ == s
        this_obs = all_obs[inds,:]
        O_sds[:,s,:] = np.std(this_obs,0)[:,None]


    #random, in vicinity of truth. doesn't matter much - doesn't directly in opt / gradients
    #since have this random clustering from KMeans - may as well use here as well...
    R = np.zeros((n_S,n_A))
    for s in range(n_S):
        for a in range(n_A):
            inds = np.logical_and(kmeans.labels_ == s, all_acts == a)
            this_rew = all_rewards[inds]
            R[s,a] = np.mean(this_rew)
    
    return pi,T,(O_means,O_sds),R


def params_init_EM(init_type,O_var_move=False,EM_iters = 50,EM_tol = 1e-6):
     
    if init_type=='random':
        pi,T,O,R = params_init_random(n_A,n_S,n_dim,alpha_pi=25,alpha_T=25)
    if init_type=='kmeans':
        pi,T,O,R = params_init_kmeans(n_A,n_S,n_dim,all_dat,ids_tr,seed=None)


    params = (pi,T,O,R)
    nat_params = to_natural_params(params)            
    lls = []
        
    print("at the start!")
    sys.stdout.flush()   
    
    last_ll = -np.inf
    for i in range(EM_iters):   
        t = time()        
        
        ll,gam,E_Njka = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,
            observs_missing_mask=observs_mask_tr)
        pi,T,O,R = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,E_Njka,R_sd,
            observs_missing_mask=observs_mask_tr)

        params = (pi,T,O,R)
        nat_params = to_natural_params(params)    
        
        lls.append(ll)  
        print(i,ll,np.round(time()-t,2))   
        sys.stdout.flush()    
        
        if np.abs(ll-last_ll)<EM_tol and not O_var_move:
            break
        last_ll = ll
                
    return params,lls


def params_init(param_init):
    #### get initialization...
    #### options:
    ####     - random  
    ####     - kmeans, then random sticky transitions
    ####     - EM-type, starting at kmeans (nothing special for rewards)
    ####        -- EM_init_type: either init with random or kmeans

    #simpler inits
    if param_init == 'random':
        params = params_init_random(n_A,n_S,n_dim,alpha_pi=25,alpha_T=25)
    # if param_init == 'kmeans':
    #     params = params_init_kmeans(n_A,n_S,n_dim,all_dat,ids_tr,alpha_pi=25,alpha_T=25,seed=None)

    #more complex inits that involve learning a model
    if param_init == 'EM-random':
        params,train_info = params_init_EM(init_type='random',O_var_move=False,EM_iters = 150,EM_tol = 1e-7)  #was 25, 1e-5
    # if param_init == 'EM-kmeans':
    #     params,train_info = params_init_EM(init_type='kmeans',O_var_move=False,EM_iters = 25,EM_tol = 1e-5)

    return params


def init_B_and_V():
    V_min = np.min(true_Rs[0])/(1-gamma)
    # B = initialize_B(params,V_min,gamma,n_expandB_iters=min(int(n_S*2),50))
    b = np.linspace(.01,.99,30)
    B = np.array([b,1-b]).T

    n_B = B.shape[0]
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]
    return V,B


def get_param_inits(param_init,n_PBVI_iters=25):
    """
    helper func, to get a bunch of inits by different means and then 
    test how well they do, and choose the best to run with
    """
    # inits = np.array(['random','kmeans','EM-random','EM-kmeans','reward-sep','BP-sep'])

    restarts_per_init = {
    'random': 250, #250
    'kmeans': 25,
    'EM-random': 50, #50
    'EM-kmeans': 25,
    }
    n_restarts = restarts_per_init[param_init]

    lls_tr = []
    polvals_tr = []
    ESS_tr = []
    objs = []

    best_obj = np.inf #will select the best init based on PC objective: HMM_obj + lambda*RL_obj
    best_EM_obj = np.inf

    for restart in range(n_restarts):
        t = time()

        params = params_init(param_init)
        nat_params = to_natural_params(params)
        pi,T,O,R = params

        print("learning policy for restart %d" %restart)
        sys.stdout.flush()

        V,B = init_B_and_V()
        for ii in range(n_PBVI_iters):
            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,eps=.001,PBVI_temps=[.01,.01,.01],n_samps=100)

        #check value of init:
        #   - log-lik of HMM on test data 
        #   - val of learned policy

        ll = HMM_obj_fun(nat_params,observs_tr,actions_tr,observs_missing_mask=observs_mask_tr)
        lls_tr.append(-ll)

        """

        all_beliefs_tr = get_beliefs(params,seq_lens_tr,actions_tr,observs_tr,observs_missing_mask=observs_mask_tr)
        CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
            behav_action_probs=action_probs_tr,all_beh_probs=None,actions=actions_tr,
            init_actions=None,observs=observs_tr,init_observs=None,
            observs_missing_mask=observs_mask_tr,init_observs_missing_mask=None,
            rewards=rewards_tr,seq_lens=seq_lens_tr,gamma=gamma,alpha_temp=.01,
            PBVI_update_iters=0,belief_with_reward=False,
            PBVI_temps=[.01,.01,.01],update_V = False,
            cached_beliefs=all_beliefs_tr,gr_safety_thresh=0,prune_num=0)
        polvals_tr.append(-CWPDIS_obj)
        ESS_tr.append(ESS)

        ###
        ### based on current lambda, select the best overall objective
        ###

        if lambd == np.inf:
            obj = log_prior(nat_params) + 1e8*CWPDIS_obj
        else:
            obj = ll + lambd*CWPDIS_obj
        objs.append(obj)

        if obj < best_obj:
            best_obj = obj
            best_nat_params = nat_params
            best_params = params
            best_V = V
            best_B = B

            best_te_ll = -HMM_obj_fun(nat_params,observs_te,actions_te,observs_missing_mask=observs_mask_te)

            all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,observs_missing_mask=observs_mask_te)
            CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
                behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                init_actions=None,observs=observs_te,init_observs=None,
                observs_missing_mask=observs_mask_te,init_observs_missing_mask=None,
                rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                PBVI_update_iters=0,belief_with_reward=False,
                PBVI_temps=[.01,.01,.01],update_V = False,
                cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

            returns = []
            for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                #traj is just s,b,a,r,o
                traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd,V=V,steps=eval_n_steps,
                                  seed=ii,tiger_env=tiger_env,T_est=T,
                                  O_est=O,belief=pi,R_est=R,temp=None,
                                  belief_with_reward=belief_with_reward,
                                  obs_meas_probs=obs_meas_probs)
                                  
                returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
            HMM_avg_returns_det = np.mean(returns)
            HMM_quant_returns_det = np.percentile(returns,[0,1,5,25,50,75,95,99,100])

            save_dict['best_init_avgreturns'] = HMM_avg_returns_det
            save_dict['best_init_quantreturns'] = HMM_quant_returns_det
            save_dict['best_init_params'] = best_params
            save_dict['best_init_natparams'] = best_nat_params
            save_dict['best_restart_ind'] = restart
            save_dict['best_obj'] = best_obj
            save_dict['best_V_init'] = best_V
            save_dict['best_B_init'] = best_B
            save_dict['best_init_te_ESS'] = ESS
            save_dict['best_init_te_ll'] = best_te_ll
            save_dict['best_init_te_polval'] = -CWPDIS_obj

        """

        ###
        ### if this was an EM init, also cache the overall best EM-based ll objective (this will be redundant across lambds)
        ###
        if param_init=='EM-random':
            if ll < best_EM_obj:
                best_EM_obj = ll

                best_te_ll = -HMM_obj_fun(nat_params,observs_te,actions_te,observs_missing_mask=observs_mask_te)
                all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,observs_missing_mask=observs_mask_te)
                CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
                    behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                    init_actions=None,observs=observs_te,init_observs=None,
                    observs_missing_mask=observs_mask_te,init_observs_missing_mask=None,
                    rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                    PBVI_update_iters=0,belief_with_reward=False,
                    PBVI_temps=[.01,.01,.01],update_V = False,
                    cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

                returns = []
                for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                    #traj is just s,b,a,r,o
                    traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,
                        R_sd,V=V,steps=eval_n_steps,
                          seed=ii,tiger_env=tiger_env,T_est=T,
                          O_est=O,belief=pi,R_est=R,temp=None,
                          belief_with_reward=belief_with_reward,
                          obs_meas_probs=obs_meas_probs)

                    returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
                HMM_avg_returns_det = np.mean(returns)
                HMM_quant_returns_det = np.percentile(returns,[0,1,5,25,50,75,95,99,100])
        
                save_dict['best_EMinit_avgreturns'] = HMM_avg_returns_det
                save_dict['best_EMinit_quantreturns'] = HMM_quant_returns_det
                save_dict['best_EM_obj'] = best_EM_obj
                save_dict['best_EMinit_te_ll'] = best_te_ll
                save_dict['best_EMinit_te_polval'] = -CWPDIS_obj
                save_dict['best_EMinit_te_ESS'] = ESS
                save_dict['best_EMinit_params'] = params
                save_dict['best_EMinit_natparams'] = nat_params
                save_dict['best_EMinit_V'] = V

        print("took %.2f" %(time()-t))
        sys.stdout.flush()

    #init stuff in case we want to check them later
    save_dict['init_objs'] = objs
    save_dict['init_lls_tr'] = lls_tr
    save_dict['init_polvals_tr'] = polvals_tr
    save_dict['init_ESS_tr'] = ESS_tr

    # return best_params,best_V,best_B



def run_PBVI_on_truth(T_wit,O_wit,R_wit,pi_wit,T_true,O_true,R_true,
    pi_true,belief_with_reward,fold_num,obs_meas_probs,
    n_PBVI_iters=25,n_traj=1000,gamma=0.9,eval_n_steps=20,R_sd=.1):
    """
    run PBVI on witness parameters for use as ground truth
    """

    V,B = init_B_and_V()
    for ii in range(n_PBVI_iters):
        V = update_V_softmax(V,B,T_wit,O_wit,R_wit,gamma,max_iter=1,verbose=False,eps=.001,PBVI_temps=[.01,.01,.01],n_samps=100)


    returns = []
    for ii in range(fold_num*n_traj,(fold_num+1)*n_traj):    
        #traj is just s,b,a,r,o
        traj = run_softmax_policy(T_true,O_true,R_true,pi_true,R_sd,V=V,
                          steps=eval_n_steps,seed=ii,tiger_env=tiger_env,
                          T_est=T_wit,O_est=O_wit,R_est=R_wit,belief=pi_wit,temp=None,
                          belief_with_reward=belief_with_reward,
                          obs_meas_probs=obs_meas_probs) 
        returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
    avg_return_true = np.mean(returns)
    return_quantiles_true = np.percentile(returns,[0,1,5,25,50,75,95,99,100])
    print("---------------------")
    print("Avg return: %.4f" %(avg_return_true))
    print("quantiles:")    
    print(return_quantiles_true)
    print("---------------------")
    sys.stdout.flush()    
            
    V_true = V.copy()
    B_true = B.copy()

    return avg_return_true,return_quantiles_true,V_true,B_true
    




if __name__ == "__main__":    

    #####
    ##### from job number, figure out param settings
    #####
        
    job_id = int(sys.argv[1])
    
    cluster = sys.argv[2]
    assert cluster=='h' or cluster=='d' or cluster=='l'
    if cluster=='h': 
        OUT_PATH = '/n/scratchlfs/doshi-velez_lab/jfutoma/prediction_constrained_RL/experiments/tiger_miss/'
    if cluster=='d':
        OUT_PATH = '/hpchome/statdept/jdf38/prediction_constrained_RL/experiments/tiger_miss/'
    if cluster=='l':
        OUT_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger_miss/'

    RESULTS_FOLDER = 'results_5oct-EM/'
    LOGS_FOLDER = 'logs_5oct-EM/'
    if not os.path.exists(OUT_PATH+RESULTS_FOLDER):
       os.makedirs(OUT_PATH+RESULTS_FOLDER)
    if not os.path.exists(OUT_PATH+LOGS_FOLDER):
       os.makedirs(OUT_PATH+LOGS_FOLDER)

    # 7 x 10 x 5 = 350
    sig_goods = np.array([0.3]) 
    sig_others = np.array([0.3]) 
    good_dim_meas_probs = np.array([0.05,0.1,0.2,0.3,0.5,0.7,0.9])
    lambds = np.array([1])  #np.array([1e-1,1e0,1e1,1e2,np.inf]) 
    n_envs = np.array([2])    
    prune_nums = np.array([0]) #10
    inits = np.array(['EM-random']) #random
    ESS_penalties = np.array([0]) #0, 10
    seeds = np.arange(1,11)
    folds = np.arange(5)
    
    hyperparams_all = itertools.product(seeds,lambds,inits,sig_goods,
        sig_others,n_envs,prune_nums,ESS_penalties,good_dim_meas_probs,folds)
    ct = 0
    for hyperparams in hyperparams_all:
        if job_id == ct:
            (seed,lambd,param_init,sig_good,sig_other,n_env,prune_num,
                ESS_penalty,good_dim_meas_prob,fold) = hyperparams
        ct += 1
    
    N = 2500
    env_name = 'tiger-miss' 
    tiger_env = True
    model_string = 'EMonly-%s_nenv%d_good-dim-meas-prob%.2f_siggood%.1f_sigother%.1f_N%d_lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d' %(
                env_name,n_env,good_dim_meas_prob,sig_good,sig_other,N,lambd,param_init,prune_num,ESS_penalty,seed,fold)

    sys.stdout = open(OUT_PATH+LOGS_FOLDER+model_string+"_out.txt","w")
    sys.stderr = open(OUT_PATH+LOGS_FOLDER+model_string+"_err.txt","w")
    print("starting job_id %d" %job_id)
    print(model_string)
    sys.stdout.flush()

    ##### start to run!

    n_dim_per_env = 1 #only 1 dim for each tiger env
    n_dim = n_dim_per_env*n_env

    belief_with_reward = False

    np.set_printoptions(threshold=10000)                                                      
    np.random.seed(seed) #for different random restarts
    rng = np.random.RandomState(fold+1)
    
    save_dict = {}
    max_traj_len = 250 #conservative upper bound on traj_lens; in practice should be much shorter
    min_traj_len = 75
    final_listen_prob = 0.5

    #dim 1 only observed fraction of the time; dim 2 always observed
    obs_meas_probs = np.array([good_dim_meas_prob,1.0]) 
    
    R_sd_end = .01 

    #RL eval params
    gamma = 0.99
    eval_n_traj = 2500 #how many traj to test on & get average return of a policy
    eval_pbvi_iters = 25 #how many eval iters of PBVI to run for ground truth
    eval_n_steps = 100 #how long each eval trajectory should be (at most)
    
    n_S = 2 #by design

    O_dims = n_dim
    true_pis,true_Ts,true_Os,true_Rs = create_tiger_env(n_S,n_dim,sig_good,sig_other,R_listen=-.1,R_good=1,R_bad=-5)
    n_A = np.shape(true_Rs[0])[1]

    ### create witness parameters as best we can hope to learn, since way too many true states
    pi_wit,T_wit,O_wit,R_wit = create_tiger_witness(true_pis,true_Ts,true_Os,true_Rs,sig_good,sig_other)
    O_means_wit = O_wit[0]
    O_sds_wit = O_wit[1]

    nat_params_wit = to_natural_params((pi_wit,T_wit,O_wit,R_wit)) 
    params_wit = (pi_wit,T_wit,O_wit,R_wit)
    params_true = (true_pis,true_Ts,true_Os,true_Rs)
    save_dict['params_true'] = params_true
    save_dict['nat_params_wit'] = nat_params_wit        
    
    ### sample train/val/test data
    observs_tr,rewards_tr,actions_tr,action_probs_tr,seq_lens_tr = simdata_random_policy(
            N,true_pis,true_Ts,true_Os,true_Rs,min_traj_len,max_traj_len,final_listen_prob,rng)
    observs_mask_tr = np.random.uniform(0,1,observs_tr.shape) <= obs_meas_probs
    bad_inds = np.isinf(observs_tr)
    observs_mask_tr[bad_inds] = 0

    observs_te,rewards_te,actions_te,action_probs_te,seq_lens_te = simdata_random_policy(
            N,true_pis,true_Ts,true_Os,true_Rs,min_traj_len,max_traj_len,final_listen_prob,rng)
    observs_mask_te = np.random.uniform(0,1,observs_te.shape) <= obs_meas_probs
    bad_inds = np.isinf(observs_te)
    observs_mask_te[bad_inds] = 0


    ### check HMM objective at witness parameters (ground truth too large to check)
    HMM_obj_fun = MAP_objective
    run_E = forward_backward_Estep

    # print("ll at witness:")
    # ll_true = -HMM_obj_fun(nat_params_wit,observs_tr,actions_tr,observs_missing_mask=observs_mask_tr)
    # print(ll_true)
    # print("test ll at witness:")
    # ll_te_true = -HMM_obj_fun(nat_params_wit,observs_te,actions_te,observs_missing_mask=observs_mask_te)
    # print(ll_te_true)
    # sys.stdout.flush()
    # save_dict['ll_te_true'] = ll_te_true
    # save_dict['ll_true'] = ll_true

    #####
    ##### PBVI on witness (truth is way too expensive for larger dims...)
    #####
    
    # print("Testing PBVI on witness model params...")
    # avg_return_true,return_quantiles_true,V_true,B_true = run_PBVI_on_truth(
    #     T_wit,O_wit,R_wit,pi_wit,true_Ts,true_Os,true_Rs,true_pis,belief_with_reward,fold,obs_meas_probs,
    #     n_PBVI_iters=eval_pbvi_iters,n_traj=eval_n_traj,gamma=gamma,eval_n_steps=eval_n_steps,R_sd=.01)
    # save_dict['PBVI_true_results'] = (avg_return_true,return_quantiles_true,V_true,B_true)
        

    #####
    ##### Joint Learning!
    #####
     
    #learning setup params
    action_prob_finaltemp = action_prob_temp = .01
    PBVI_train_update_iters = 5
    PBVI_temps = [.01,.01,.01] 
    
    #reward stuff
    R_sd = R_sd_end = .01 
    V,B = init_B_and_V()

    ##### setup gradient functions
    ## explicitly split our objective into HMM term and RL term so we can 
    ## track each value and gradients separately
    
    RLobj_V_g = value_and_output_and_grad(softmax_policy_value_objective_term)
    Prior_obj_g = vg(log_prior)
    HMMobj_g = vg(HMM_obj_fun)

    ### learning params
    n_epochs = 5000
    batchsize = N 
    lr = 1e-3

    # params,V,B = get_param_inits(param_init)
    get_param_inits(param_init)

    # nat_params = to_natural_params(params)
    # pi,T,O,R = params

    try:
        print("saving!")
        sys.stdout.flush()    
        with open(OUT_PATH+RESULTS_FOLDER+model_string+'.p','wb') as f:
            pickle.dump(save_dict, f) 
    except:
        print("save failed!!")
        sys.stdout.flush()  
    
    """

    #do a forward backward pass to get gam which we'll need to update R
    _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,get_xi=False,
        observs_missing_mask=observs_mask_tr)
        
    flat_nat_params,unflatten = flatten(nat_params)
    
    #store progress
    objs = []
    RL_objs = []
    HMM_objs = []
    
    RL_grads = []
    HMM_grads = []
    
    grad_norms = []
    RL_grad_norm = []
    HMM_grad_norm = []
     
    HMM_te_objs = []
    grad_norms_HMM_te = []
    
    avg_te_returns_det = []
    quant_te_returns_det = []

    est_te_policy_val = []
    est_te_policy_val_np = []
    te_ESS_noprune = []

    # tr_ESS_noprune = []
    tr_CWPDIS_obj = []
    tr_CWPDIS_obj_noprune = []
    
    tracked_params = []
    tracked_Vs = []

    tot_iter = 0
    last_v = np.inf #last objective value
    last_g = np.zeros(len(flat_nat_params))
    step_sizes = lr*np.ones(len(flat_nat_params))
    last_steps = np.zeros(len(flat_nat_params))
    
    for epoch in range(n_epochs):
        print("starting epoch %d" %epoch)
        sys.stdout.flush()

        for n_iter in range(N//batchsize): 
            t = time()
            
            #####
            ##### HMM objective
            #####

            if np.isinf(lambd): #no HMM likelihood just the prior
                HMM_obj,HMM_grad = Prior_obj_g(nat_params)
            else:
                HMM_obj,HMM_grad = HMMobj_g(nat_params,observs_tr,
                    actions_tr,observs_missing_mask=observs_mask_tr)
            HMM_grad = flatten(HMM_grad)[0]
            

            #####
            ##### RL objective 
            #####

            RL_obj,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                ESS_noprune,CWPDIS_obj_noprune),RL_grad = RLobj_V_g(nat_params,R,R_sd,V,B,
                behav_action_probs=action_probs_tr,all_beh_probs=None,actions=actions_tr,
                init_actions=None,observs=observs_tr,init_observs=None,
                observs_missing_mask=observs_mask_tr,init_observs_missing_mask=None,
                rewards=rewards_tr,seq_lens=seq_lens_tr,gamma=gamma,alpha_temp=.01,
                PBVI_update_iters=0,belief_with_reward=False,
                PBVI_temps=[.01,.01,.01],update_V = True,
                cached_beliefs=None,gr_safety_thresh=0,prune_num=0)

            try:
                V = [V[0]._value,V[1]._value]
            except:
                pass

            if np.isinf(lambd): #rescale so RL objective stronger than prior
                RL_obj *= np.abs(HMM_obj)*1e8
                RL_grad = flatten(RL_grad)[0]*1e8
            else:
                RL_obj *= lambd
                RL_grad = flatten(RL_grad)[0]*lambd   
        
            g = RL_grad + HMM_grad
            v = RL_obj + HMM_obj

            g = np.clip(g,-1e4,1e4)

            #save stuff
            objs.append(v)
            grad_norms.append(np.sum(np.abs(g)))

            HMM_objs.append(HMM_obj)
            HMM_grad_norm.append(np.sum(np.abs(HMM_grad)))

            RL_objs.append(RL_obj)
            RL_grad_norm.append(np.sum(np.abs(RL_grad)))

            # tr_ESS.append(ESS._value) 
            # tr_ESS_noprune.append(ESS_noprune._value)
            # tr_CWPDIS.append((CWPDIS_nums._value, CWPDIS_denoms._value))
            tr_CWPDIS_obj_noprune.append(CWPDIS_obj_noprune._value)
            tr_CWPDIS_obj.append(CWPDIS_obj._value)


            #finally, apply gradient!
            flat_nat_params,last_g,step_sizes,last_steps = rprop(flat_nat_params,g,last_g,step_sizes,last_steps,v,last_v)
            last_v = v
                
            pi,T,O,R = to_params(unflatten(flat_nat_params))
            params = (pi,T,O,R)            
            nat_params = to_natural_params(params)
            
            #update R separately via incremental EM on this minibatch 
            _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,
                get_xi=False,observs_missing_mask=observs_mask_tr)
            R = M_step_just_reward(nat_params,observs_tr,actions_tr,rewards_tr,gam,R_sd)

            params = (pi,T,O,R)            
            nat_params = to_natural_params(params)
            flat_nat_params,unflatten = flatten(nat_params)


            #####
            ##### End of learning iteration, now do some checks every so often...
            #####

            tot_iter += 1
            if tot_iter%1==0:
                print("epoch %d, iter %d, RL obj %.4f, HMM obj %.4f, total obj %.4f grad L1-norm %.4f, took %.2f" 
                      %(epoch,tot_iter,RL_obj,HMM_obj,v,np.sum(np.abs(g)),time()-t))
                sys.stdout.flush()
            
            #every so often, check HMM on test set
            if tot_iter % 50 == 1:                              

                ##### check HMM performance on held-out test set
                HMM_obj,HMM_grad = HMMobj_g(nat_params,observs_te,actions_te,
                    observs_missing_mask=observs_mask_te)  
                print("HMM loglik on test data %.4f, grad norm %.4f" 
                      %(-HMM_obj,np.sum(np.abs(flatten(HMM_grad)[0]))))
                
                HMM_te_objs.append(-HMM_obj)
                grad_norms_HMM_te.append(np.sum(np.abs(flatten(HMM_grad)[0])))


                ## test the OPE on test set as well...
                all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                    observs_missing_mask=observs_mask_te)
                _,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                    ESS_noprune,CWPDIS_obj_noprune) = softmax_policy_value_objective_term(
                    nat_params,R,R_sd,V,B,
                    behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                    init_actions=None,observs=observs_te,init_observs=None,
                    observs_missing_mask=observs_mask_te,init_observs_missing_mask=None,
                    rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                    PBVI_update_iters=0,belief_with_reward=False,
                    PBVI_temps=[.01,.01,.01],update_V = False,
                    cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

                print('iter %d, est value of policy on test data: %.5f' %(tot_iter,CWPDIS_obj_noprune))
                est_te_policy_val.append(CWPDIS_obj)
                est_te_policy_val_np.append(CWPDIS_obj_noprune)
                # te_ESS.append(ESS)
                te_ESS_noprune.append(ESS_noprune)
                # te_CWPDIS.append((CWPDIS_nums,CWPDIS_denoms))

                tracked_params = params
                tracked_Vs = V


                print("testing deterministic policy via rollouts...")
                sys.stdout.flush()
                returns = []
                for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                    #traj is just s,b,a,r,o
                    traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd,V=V,steps=eval_n_steps,
                                      seed=ii,tiger_env=tiger_env,T_est=T,
                                      O_est=O,belief=pi,R_est=R,temp=None,
                                      belief_with_reward=belief_with_reward,
                                      obs_meas_probs=obs_meas_probs)
                    returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
                avg_te_returns_det.append(np.mean(returns))
                quant_te_returns_det.append(np.percentile(returns,[0,1,5,25,50,75,95,99,100]))
            
                print("Avg test return: %.4f" %avg_te_returns_det[-1])
                print("quantiles:")    
                print(quant_te_returns_det[-1])
                sys.stdout.flush()
                        
                ### save
                save_dict = update_and_write_savedict(save_dict)           
    
    """
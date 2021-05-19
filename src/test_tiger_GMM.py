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
from scipy import stats

#import matplotlib.pyplot as plt

from util import *
from pbvi_cts import *
from action_hmm_cts import *
from envs_cts import *   


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
    save_dict['est_te_policy_val_np'] = est_te_policy_val_np

    save_dict['avg_te_returns_det'] = avg_te_returns_det
    save_dict['quant_te_returns_det'] = quant_te_returns_det

    # save_dict['tr_ESS'] = tr_ESS
    # save_dict['tr_ESS_noprune'] = tr_ESS_noprune
    save_dict['tr_CWPDIS_obj'] = tr_CWPDIS_obj
    save_dict['tr_CWPDIS_obj_noprune'] = tr_CWPDIS_obj_noprune

    save_dict['te_ESS_noprune'] = te_ESS_noprune

    save_dict['tracked_params'] = tracked_params         
    save_dict['tracked_Vs'] = tracked_Vs            

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


def simdata_random_policy(N,pis,Ts,Os,Rs,min_traj_len,max_traj_len,listen_prob):
    """
    create a bunch of simulated trajectories for tiger, taking random actions
    """

    n_env = len(Os)
    n_dim_per_env = 1
    n_dim = n_env*n_dim_per_env 

    n_A = Rs[0].shape[1]

    seq_lens = max_traj_len*np.ones(N,dtype="int") #traj_len is max length
    observs = -np.inf*np.ones((N,max_traj_len,n_dim))
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
        actions[n,0] = draw_discrete(prob)
        action_probs[n,0] = prob[actions[n,0]]

        #loop over dims 
        last_states = []
        next_states = []
        this_observs = []
        for i,(pi,T,O) in enumerate(zip(pis,Ts,Os)):
            last_states.append(draw_discrete(pi))
            next_states.append(draw_discrete(T[:,last_states[i],actions[n,0]]))
            o = O(next_states[i],actions[n,0])
            this_observs.append(o)

        observs[n,0,:] = this_observs        
        rewards[n,0] = Rs[0][last_states[0],actions[n,0]] #NOTE: s_0 now!
        last_states = next_states
        if actions[n,0]!=n_A-1: #if chose and didn't listen, end for tiger 
            seq_lens[n] = t
            continue
        
        for t in range(1,seq_lens[n]):
            prob = action_prob if t+1 >= min_traj_len else action_prob_init
            actions[n,t] = draw_discrete(prob)
            action_probs[n,t] = prob[actions[n,t]]

            next_states = []
            this_observs = []
            for i,(pi,T,O) in enumerate(zip(pis,Ts,Os)):
                next_states.append(draw_discrete(T[:,last_states[i],actions[n,t]]))
                o = O(next_states[i],actions[n,t])
                this_observs.append(o)

            observs[n,t,:] = this_observs        
            rewards[n,t] = Rs[0][last_states[0],actions[n,t]] #NOTE: s_t now!

            last_states = next_states
            if actions[n,t]!=n_A-1: #if chose and didn't listen, end for tiger 
                seq_lens[n] = t+1
                break

    max_T = np.max(seq_lens)
    observs = observs[:,:max_T,:]
    rewards = rewards[:,:max_T]
    actions = actions[:,:max_T]
    action_probs = action_probs[:,:max_T]

    return observs,rewards,actions,action_probs,seq_lens


def params_init_random(n_A,n_S,O_dims,alpha_pi=25,alpha_T=25):
    """ 
    random initialization
    """

    #transitions: sticky, usually stay in same state
    T = np.zeros((n_S,n_S,n_A))   
    for s in range(n_S):
        alpha_vec = np.ones(n_S)
        alpha_vec[s] = alpha_T
        T[:,s,:] = np.random.dirichlet(alpha_vec)[:,None]
       
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
        
        ll,gam,E_Njka = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,init_observs=None)
        pi,T,O,R = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,E_Njka,R_sd,init_observs=None)

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
        params,train_info = params_init_EM(init_type='random',O_var_move=False,EM_iters = 25,EM_tol = 1e-5)
    # if param_init == 'EM-kmeans':
    #     params,train_info = params_init_EM(init_type='kmeans',O_var_move=False,EM_iters = 25,EM_tol = 1e-5)

    return params


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

        ll = HMM_obj_fun(nat_params,observs_tr,actions_tr,rewards_tr,R_sd)
        lls_tr.append(-ll)

        all_beliefs_tr = get_beliefs(params,seq_lens_tr,None,actions_tr,observs_tr)
        CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,action_probs_tr,actions_tr,observs_tr,
            None,rewards_tr,seq_lens_tr,gamma,alpha_temp=.01,PBVI_update_iters=0,belief_with_reward=False,
            PBVI_temps=[.01,.01,.01],update_V = False,cached_beliefs=all_beliefs_tr,prune_num=0)
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

            best_te_ll = -HMM_obj_fun(best_nat_params,observs_te,actions_te,rewards_te,R_sd)
            all_beliefs_te = get_beliefs(params,seq_lens_te,None,actions_te,observs_te)
            CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,action_probs_te,actions_te,observs_te,
                None,rewards_te,seq_lens_te,gamma,alpha_temp=.01,PBVI_update_iters=0,belief_with_reward=False,
                PBVI_temps=[.01,.01,.01],update_V = False,cached_beliefs=all_beliefs_te,prune_num=0)

            returns = []
            for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                #traj is just s,b,a,r,o
                traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd,V=V,steps=eval_n_steps,
                                  seed=ii,tiger_env=tiger_env,T_est=T,
                                  O_est=O,belief=pi,R_est=R,temp=None,
                                  belief_with_reward=belief_with_reward)
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

        ###
        ### if this was an EM init, also cache the overall best EM-based ll objective (this will be redundant across lambds)
        ###
        if param_init=='EM-random':
            if ll < best_EM_obj:
                best_EM_obj = ll

                best_te_ll = -HMM_obj_fun(nat_params,observs_te,actions_te,rewards_te,R_sd)
                all_beliefs_te = get_beliefs(params,seq_lens_te,None,actions_te,observs_te)
                CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,action_probs_te,actions_te,observs_te,
                    None,rewards_te,seq_lens_te,gamma,alpha_temp=.01,PBVI_update_iters=0,belief_with_reward=False,
                    PBVI_temps=[.01,.01,.01],update_V = False,cached_beliefs=all_beliefs_te,prune_num=0)

                returns = []
                for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                    #traj is just s,b,a,r,o
                    traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd,V=V,steps=eval_n_steps,
                                      seed=ii,tiger_env=tiger_env,T_est=T,
                                      O_est=O,belief=pi,R_est=R,temp=None,
                                      belief_with_reward=belief_with_reward)
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

    return best_params,best_V,best_B


def init_B_and_V():
    V_min = np.min(true_Rs[0])/(1-gamma)
    # B = initialize_B(params,V_min,gamma,n_expandB_iters=min(int(n_S*2),50))
    b = np.linspace(.01,.99,20)
    B = np.array([b,1-b]).T

    n_B = B.shape[0]
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]
    return V,B



def softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,behav_action_probs,actions,observs,init_observs,
    rewards,seq_lens,gamma,alpha_temp,PBVI_update_iters,belief_with_reward,PBVI_temps,
    update_V,cached_beliefs,clip=True,clip_bds=[1e-16,1e3],prune_num=0,
    eval_mixture=False,eval_mixture_prob=0.5,rescale_obj=None,ESS_penalty=0,V_penalty=0):
    """
    this is our actual objective term, that aims to estimate how 
    good our current softmax policy is.
    
    we do this via importance sampling, comparing the current policy
    to a behavior policy, using the CWPDIS estimator.
    
    for on-policy settings where we have a simulator, this will be data from 
    a recent, stale version of our learned policy. 
    
    in off-policy, it's a batch (or a subset) of some retrospective data,
    and it's up to us to estimate behavior action probabilities beforehand...
    """
    ###NOTE: in order to avoid taking grads wrt R, we pass it in as a different arg 
    pi,T,O,_ = to_params(nat_params)
    O_means,O_sds = O # D x S x A
    EPS = 1e-16
    EPS_ACTPROB = .001

    log_T = np.log(T+EPS)
    log_pi = np.log(pi+EPS)
   
    if update_V:
        V = [np.array(V[0],copy=True),np.array(V[1],copy=True)]
        for _ in range(PBVI_update_iters):
            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,PBVI_temps=PBVI_temps)

    max_T = int(np.max(seq_lens))
    n_traj = np.shape(behav_action_probs)[0]

    CWPDIS_nums = []
    CWPDIS_denoms = []

    CWPDIS_nums_noprune = []
    CWPDIS_denoms_noprune = []
             
    rhos = np.ones(n_traj)
    old_rhos = np.zeros(0,"float") #store imp weights at end of all trajectories
    masked_actions = np.copy(actions)

    saved_action_probs = []
    saved_rhos = []
    ESS = [] 
    ESS_noprune = []
    saved_beh_action_probs = []

    #init beliefs
    if cached_beliefs is not None:
        beliefs = cached_beliefs[:,0,:]
    else:
        if init_observs is None:
            log_b = np.tile(log_pi[:,None],(1,N)) #S x N   
        else:
            init_actions = np.zeros(n_traj,"int")
            log_obs = np.sum(stat.norm.logpdf(init_observs[:,None,:],
                np.transpose(O_means[:,:,init_actions],[2,1,0]),
                np.transpose(O_sds[:,:,init_actions],[2,1,0])), 2).T #S x N
            log_b = log_pi[:,None] + log_obs #S x N 
        beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'


    for t in range(max_T):
        this_actions = actions[:,t] 
        mask = this_actions!=-1 #mask over all trajs, even those that ended
        rho_mask = masked_actions[:,t]!=-1 #variable size mask
        
        b_alphas = np.dot(beliefs[rho_mask,:],V[0].T)/alpha_temp 
        exp_balpha = np.exp(b_alphas - np.max(b_alphas,1)[:,None])
        alpha_probs = exp_balpha / np.sum(exp_balpha,1)[:,None]
        all_action_probs = np.dot(alpha_probs,V[1])           
        action_probs = all_action_probs[np.arange(np.sum(rho_mask)),this_actions[mask]]

        #TODO: add mask here for guardrails...to do this will need to precompute nn's for every transition in dataset...

        saved_action_probs.append(action_probs)
        saved_beh_action_probs.append(behav_action_probs[mask,t])

        action_probs = np.where(action_probs<EPS_ACTPROB,EPS_ACTPROB,action_probs) #fix 0 probs
        
        #combine learned policy action probs with behavior for evaluation
        if eval_mixture:
            #weight eval_mixture_prob on learned policy, (1-eval_mixture_prob) on behavior
            action_probs = eval_mixture_prob*action_probs + (1-eval_mixture_prob)*behav_action_probs[mask,t]

        old_rhos = np.concatenate([old_rhos,rhos[np.logical_not(rho_mask)]]) #cache old rhos; need for denom
        rhos = rhos[rho_mask]*action_probs/behav_action_probs[mask,t]

        if clip:
            rhos = np.clip(rhos,clip_bds[0],clip_bds[1])

        ESS_noprune.append((np.sum(rhos)+np.sum(old_rhos))**2 / (np.sum(rhos**2)+np.sum(old_rhos**2)))
        CWPDIS_nums_noprune.append(np.sum(rhos*rewards[mask,t])) #implicitly uses old_rhos, but those rewards are all 0...
        CWPDIS_denoms_noprune.append(np.sum(rhos)+np.sum(old_rhos))

        if prune_num > 0: #prune top K rhos at each time...
            all_rhos = np.concatenate([rhos,old_rhos])
            thresh = np.sort(all_rhos)[::-1][prune_num-1]

            pruned_rhos = np.where(rhos>=thresh,EPS,rhos)
            pruned_old_rhos = np.where(old_rhos>=thresh,EPS,old_rhos)

            ESS.append((np.sum(pruned_rhos)+np.sum(pruned_old_rhos))**2 / (np.sum(pruned_rhos**2)+np.sum(pruned_old_rhos**2)))
            saved_rhos.append(np.concatenate([pruned_rhos,pruned_old_rhos]))
            #cache sstats for CWPDIS
            CWPDIS_nums.append(np.sum(pruned_rhos*rewards[mask,t])) #implicitly uses old_rhos, but those rewards are all 0...
            CWPDIS_denoms.append(np.sum(pruned_rhos)+np.sum(pruned_old_rhos))
        else: #just use them as-is...
            ESS.append((np.sum(rhos)+np.sum(old_rhos))**2 / (np.sum(rhos**2)+np.sum(old_rhos**2)))
            saved_rhos.append(np.concatenate([rhos,old_rhos]))
            #cache sstats for CWPDIS
            CWPDIS_nums.append(np.sum(rhos*rewards[mask,t])) #implicitly uses old_rhos, but those rewards are all 0...
            CWPDIS_denoms.append(np.sum(rhos)+np.sum(old_rhos))

        masked_actions = masked_actions[rho_mask,:]

        if cached_beliefs is not None:
            beliefs = cached_beliefs[mask,t+1,:]
        else:

            #update all beliefs
            log_obs = np.sum(stat.norm.logpdf(
                    observs[mask,t,:].T[:,None,:], #D x 1 x N
                    O_means[:,:,this_actions[mask]], #D x S x N
                    O_sds[:,:,this_actions[mask]]) 
                    ,0) #S' x N

            #T: S' x S x A 
            lb = np.log(beliefs[rho_mask,:]+EPS) # N x S
            log_T_b = log_T[:,:,this_actions[mask]] + lb.T[None,:,:] # S' x S x N

            #assumes we filter without rewards
            log_b = log_obs + logsumexp(log_T_b,1) #S' x N
            beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'

    CWPDIS_obj = np.sum(np.power(gamma,np.arange(max_T))*
                        np.array(CWPDIS_nums)/(np.array(CWPDIS_denoms)+EPS))

    CWPDIS_obj_noprune = np.sum(np.power(gamma,np.arange(max_T))*
                        np.array(CWPDIS_nums_noprune)/(np.array(CWPDIS_denoms_noprune)+EPS))

    #NOTE: ESS_noprune, not ESS!!!
    # RL_obj = CWPDIS_obj
    RL_obj = CWPDIS_obj - ESS_penalty*np.sum(1/np.sqrt(np.array(ESS_noprune)))

    if update_V and V_penalty > 0:
        RL_obj = RL_obj - V_penalty*np.sum(np.power(V[0],2))

    return -RL_obj,(V,CWPDIS_obj,np.array(ESS),np.array(CWPDIS_nums),np.array(CWPDIS_denoms),np.array(ESS_noprune),CWPDIS_obj_noprune)


def run_PBVI_on_truth(T_wit,O_wit,R_wit,pi_wit,T_true,O_true,R_true,pi_true,belief_with_reward,fold_num,
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
                          belief_with_reward=belief_with_reward) 
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
    
    ###EXPERIMENT 1: vary number of dimensions.
    
    job_id = int(sys.argv[1])
    
    cluster = sys.argv[2]
    assert cluster=='h' or cluster=='d' or cluster=='l'
    if cluster=='h': 
        OUT_PATH = '/n/scratchlfs/doshi-velez_lab/jfutoma/prediction_constrained_RL/experiments/tiger_gmm/'
    if cluster=='d':
        OUT_PATH = '/hpchome/statdept/jdf38/prediction_constrained_RL/experiments/tiger_gmm/'
    if cluster=='l':
        OUT_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger_gmm/'

    RESULTS_FOLDER = 'results_21may/'
    LOGS_FOLDER = 'logs_21may/'
    if not os.path.exists(OUT_PATH+RESULTS_FOLDER):
       os.makedirs(OUT_PATH+RESULTS_FOLDER)
    if not os.path.exists(OUT_PATH+LOGS_FOLDER):
       os.makedirs(OUT_PATH+LOGS_FOLDER)

    # 2 2 4 2 2 2 5 5 = 3200
    sigma_0s = np.array([0.1,0.3]) 
    sigma_1s = np.array([0.5,1.0]) 
    lambds = np.array([1e0,1e1,1e2,np.inf]) #-1
    prune_nums = np.array([0,10])
    inits = np.array(['random','EM-random'])
    ESS_penalties = np.array([0,25])
    seeds = np.arange(1,6)
    folds = np.arange(5)
    
    hyperparams_all = itertools.product(seeds,lambds,inits,sigma_0s,sigma_1s,prune_nums,ESS_penalties,folds)
    ct = 0
    for hyperparams in hyperparams_all:
        if job_id == ct:
            seed,lambd,param_init,sigma_0,sigma_1,prune_num,ESS_penalty,fold = hyperparams
        ct += 1
    
    N = 1000
    n_env = 1
    lr = 1e-3
    env_name = 'tigergmm' 
    model_string = '%s_sig0-%.1f_sig1-%.1f_N%d_lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d' %(
            env_name,sigma_0,sigma_1,N,lambd,param_init,prune_num,ESS_penalty,seed,fold)

    sys.stdout = open(OUT_PATH+LOGS_FOLDER+model_string+"_out.txt","w")
    sys.stderr = open(OUT_PATH+LOGS_FOLDER+model_string+"_err.txt","w")
    print("starting job_id %d" %job_id)
    print(model_string)

    ##### start to run!

    # #just for local debugging
    # N = 2000
    sigma_0 = .1 #sd of left Gaussian
    sigma_1 = 1 #sd of right Gaussian
    lambd = 0
    lr = 1
    seed = 12345
    O_var_move_scale = 1
    n_env = 1

    tiger_env = 'gmm'
    n_dim_per_env = 1 #only 1 dim for each tiger env
    n_dim = n_dim_per_env*n_env

    belief_with_reward = False

    np.set_printoptions(threshold=10000)                                                      
    np.random.seed(seed)
    
    save_dict = {}
    max_traj_len = 250 #conservative upper bound on traj_lens; in practice should be much shorter
    min_traj_len = 5
    final_listen_prob = 0.9

    
    R_sd_end = .01 #anneal over time...

    #RL eval params
    gamma = 0.9
    eval_n_traj = 1000 #how many traj to test on & get average return of a policy
    eval_pbvi_iters = 25 #how many eval iters of PBVI to run
    eval_n_steps = 20 #how long each eval trajectory should be (at most)

    n_S = 2 #by design
    O_dims = n_dim

    z_prob = .5 # prob of drawing from rightmost Gaussian (vs the one at 0)

    #NOTE: ONLY VALID FOR Z_PROB=.5!
    #get the overall observations to look like 2 gaussians
    pz0 = stats.norm.cdf(0,loc=0,scale=sigma_0) / (stats.norm.cdf(0,loc=0,scale=sigma_0) + stats.norm.cdf(0,loc=1,scale=sigma_1))
    pz1 = (1-stats.norm.cdf(0,loc=0,scale=sigma_0)) / (1-stats.norm.cdf(0,loc=0,scale=sigma_0) + 1-stats.norm.cdf(0,loc=1,scale=sigma_1))
    s_prob = pz0 / (pz0+pz1) #prob of being in state 1 (vs state 0)

    true_pis,true_Ts,true_Os,true_Rs = create_tiger_gmm_env(n_env,n_S,sigma_0=sigma_0,sigma_1=sigma_1,
            R_listen=-.1,R_good=1,R_bad=-5,z_prob=z_prob,pi=np.array([1-s_prob,s_prob]))
    n_A = np.shape(true_Rs[0])[1]

    pi_wit,T_wit,O_wit,R_wit = create_tiger_GMM_witness(true_pis,true_Ts,true_Os,true_Rs)
    O_means_wit = O_wit[0]
    O_sds_wit = O_wit[1]


    nat_params_wit = to_natural_params((pi_wit,T_wit,O_wit,R_wit)) 
    params_wit = (pi_wit,T_wit,O_wit,R_wit)
    params_true = (true_pis,true_Ts,true_Os,true_Rs)
    # save_dict['params_true'] = params_true
    save_dict['nat_params_wit'] = nat_params_wit       

    
    N = 50000
    ### sample train/val/test data
    observs_tr,rewards_tr,actions_tr,action_probs_tr,seq_lens_tr = simdata_random_policy(
            N,true_pis,true_Ts,true_Os,true_Rs,min_traj_len,max_traj_len,final_listen_prob)
    # observs_te,rewards_te,actions_te,action_probs_te,seq_lens_te = simdata_random_policy(
            # N,true_pis,true_Ts,true_Os,true_Rs,min_traj_len,max_traj_len,final_listen_prob)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", font_scale=1)

    plt.close('all')
    plt.figure(figsize=(10,10))
    obs = observs_tr[np.logical_not(np.isinf(observs_tr))]
    obs_0 = obs[obs<0]
    obs_1 = obs[obs>0]
    bins = np.arange(-2.5,4.6,.1)

    # plt.hist(obs,bins)

    plt.hist(obs_0,bins,alpha=.6)
    plt.hist(obs_1,bins,alpha=.6)
    plt.savefig('/Users/josephfutoma/Dropbox/research/talks/harvard/tiger-gmm-viz.pdf')


    ### check HMM objective at witness parameters (ground truth too large to check)
    HMM_obj_fun = MAP_objective_reward if belief_with_reward else MAP_objective
    run_E = forward_backward_Estep_rewards if belief_with_reward else forward_backward_Estep

    print("ll at witness:")
    ll_true = -HMM_obj_fun(nat_params_wit,observs_tr,actions_tr,rewards_tr,R_sd_end)
    print(ll_true)
    print("test ll at witness:")
    ll_te_true = -HMM_obj_fun(nat_params_wit,observs_te,actions_te,rewards_te,R_sd_end)
    print(ll_te_true)
    sys.stdout.flush()
    save_dict['ll_te_true'] = ll_te_true
    save_dict['ll_true'] = ll_true

    #####
    ##### PBVI on witness (truth is way too expensive for larger dims...)
    #####
    
    print("Testing PBVI on witness model params...")
    avg_return_true,return_quantiles_true,V_true,B_true = run_PBVI_on_truth(
        T_wit,O_wit,R_wit,pi_wit,true_Ts,true_Os,true_Rs,true_pis,belief_with_reward,fold,
        n_PBVI_iters=eval_pbvi_iters,n_traj=eval_n_traj,gamma=0.9,eval_n_steps=eval_n_steps,R_sd=.01)
    save_dict['PBVI_true_results'] = (avg_return_true,return_quantiles_true,V_true,B_true)
        

    #####
    ##### Joint Learning!
    #####

    #learning setup params
    action_prob_finaltemp = action_prob_temp = .01
    PBVI_train_update_iters = 10
    PBVI_temps = [.01,.01,.01]  
    
    #reward stuff
    R_sd = R_sd_end = .01 #anneal over time, but doesn't matter much unless rewards used in beliefs
    V,B = init_B_and_V()


    ##### setup gradient functions
    ## explicitly split our objective into HMM term and RL term so we can 
    ## track each value and gradients separately
    
    RLobj_V_g = value_and_output_and_grad(softmax_policy_value_objective_term)
    Prior_obj_g = vg(log_prior)
    HMMobj_g = vg(HMM_obj_fun)

    ### learning params
    n_epochs = 20000
    batchsize = N 

    params,V,B = get_param_inits(param_init)
    nat_params = to_natural_params(params)
    pi,T,O,R = params


    #do a forward backward pass to get gam which we'll need to update R
    _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,get_xi=False)        
    
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

            this_observ = observs_tr
            this_act = actions_tr
            this_rew = rewards_tr

            #####
            ##### HMM objective
            #####

            if np.isinf(lambd): #no HMM likelihood just the prior
                HMM_obj,HMM_grad = Prior_obj_g(nat_params)
            else:
                HMM_obj,HMM_grad = HMMobj_g(nat_params,this_observ,this_act,this_rew,R_sd)
            HMM_grad = flatten(HMM_grad)[0]
            
            #####
            ##### RL objective 
            #####

            this_beh_action_probs = action_probs_tr
            this_beh_actions = actions_tr
            this_beh_observs = observs_tr
            this_beh_rewards = rewards_tr
            this_beh_seq_lens = seq_lens_tr

            RL_obj,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,ESS_noprune,CWPDIS_obj_noprune),RL_grad = RLobj_V_g(nat_params,
                R,R_sd,V,B,this_beh_action_probs,this_beh_actions,
                this_beh_observs,None,this_beh_rewards,this_beh_seq_lens,gamma,
                action_prob_temp,PBVI_train_update_iters,belief_with_reward,PBVI_temps,
                update_V=True,cached_beliefs=None,rescale_obj=None,clip=True,clip_bds=[1e-16,1e3],prune_num=prune_num,
                ESS_penalty=ESS_penalty) 

            V = [V[0]._value,V[1]._value]

            if np.isinf(lambd): #rescale so RL objective stronger than prior
                RL_obj *= np.abs(HMM_obj)*1e5
                RL_grad = flatten(RL_grad)[0]*1e5
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
            _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,get_xi=False,init_observs=None)
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
                tracked_params.append(params)

                ##### check HMM performance on held-out test set
                HMM_obj,HMM_grad = HMMobj_g(nat_params,observs_te,actions_te,rewards_te,R_sd)  
                print("HMM loglik on test data %.4f, grad norm %.4f" 
                      %(-HMM_obj,np.sum(np.abs(flatten(HMM_grad)[0]))))
                
                HMM_te_objs.append(-HMM_obj)
                grad_norms_HMM_te.append(np.sum(np.abs(flatten(HMM_grad)[0])))


                ## test the OPE on test set as well...
                all_beliefs_te = get_beliefs(params,seq_lens_te,None,actions_te,observs_te)

                _,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,ESS_noprune,CWPDIS_obj_noprune) = softmax_policy_value_objective_term(nat_params,
                    R,R_sd,V,B,action_probs_te,actions_te,observs_te,None,
                    rewards_te,seq_lens_te,gamma,alpha_temp=action_prob_temp,PBVI_update_iters=0,belief_with_reward=False,PBVI_temps=PBVI_temps,
                    update_V = False,cached_beliefs=all_beliefs_te,prune_num=prune_num)
                print('iter %d, est value of policy on test data: %.5f' %(tot_iter,CWPDIS_obj_noprune))
                est_te_policy_val.append(CWPDIS_obj)
                est_te_policy_val_np.append(CWPDIS_obj_noprune)
                # te_ESS.append(ESS)
                te_ESS_noprune.append(ESS_noprune)
                # te_CWPDIS.append((CWPDIS_nums,CWPDIS_denoms))

                tracked_params.append(params)
                tracked_Vs.append(V)


                print("testing deterministic policy via rollouts...")
                sys.stdout.flush()
                returns = []
                for ii in range(fold*eval_n_traj,(fold+1)*eval_n_traj):    
                    #traj is just s,b,a,r,o
                    traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd,V=V,steps=eval_n_steps,
                                      seed=ii,tiger_env=tiger_env,T_est=T,
                                      O_est=O,belief=pi,R_est=R,temp=None,
                                      belief_with_reward=belief_with_reward)
                    returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
                avg_te_returns_det.append(np.mean(returns))
                quant_te_returns_det.append(np.percentile(returns,[0,1,5,25,50,75,95,99,100]))
            
                print("Avg test return: %.4f" %avg_te_returns_det[-1])
                print("quantiles:")    
                print(quant_te_returns_det[-1])
                sys.stdout.flush()
                        
                ### save
                save_dict = update_and_write_savedict(save_dict)           


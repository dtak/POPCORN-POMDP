
import os
import sys
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

from util import *
from action_hmm_cts import *
from pbvi_cts import *


def softmax_policy_value_objective_term(nat_params,R,V,B,
    behav_action_probs,all_beh_probs,actions,
    init_actions,observs,init_observs,
    observs_missing_mask,init_observs_missing_mask,
    rewards,seq_lens,gamma,
    gr_safety_thresh=0,
    cached_beliefs=None,update_V=False,
    alpha_temp=0.01,PBVI_temps=[0.01,0.01,0.01],
    PBVI_update_iters=1,
    clip=True,clip_bds=[1e-16,1e3],
    prune_num=0,ESS_penalty=0,
    eval_mixture=False,eval_mixture_prob=0.5,
    rescale_obj=None,V_penalty=0,
    belief_with_reward=False,R_sd=0.01):
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
    EPS_ACTPROB = 1e-3

    log_T = np.log(T+EPS)
    log_pi = np.log(pi+EPS)
    log_behav_action_probs = np.log(behav_action_probs+EPS)
   
    if update_V:
        V = [np.array(V[0],copy=True),np.array(V[1],copy=True)]
        for _ in range(PBVI_update_iters):
            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,PBVI_temps=PBVI_temps)

    max_T = int(np.max(seq_lens))
    N = n_traj = np.shape(behav_action_probs)[0]

    #check and see if we're able to operate in log space or not 
    # (for now, can't do that unless rewards are nonnegative...)
    all_rews = rewards[np.logical_not(np.isinf(rewards))]
    operate_in_logspace = np.all(all_rews>=0)

    if operate_in_logspace:
        logCWPDIS_nums = []
        logCWPDIS_nums_noprune = []
    else:
        CWPDIS_nums = []
        CWPDIS_nums_noprune = []
    logCWPDIS_denoms_noprune = []
    logCWPDIS_denoms = []
             
    logrhos = np.ones(n_traj)
    old_logrhos = np.zeros(0,"float") #store imp weights at end of all trajectories
    
    ESS = [] 
    ESS_noprune = []

    #init beliefs
    if cached_beliefs is not None:
        beliefs = cached_beliefs[:,0,:]
    else:

        ### get initial beliefs from prior pi & initial obs (observed before any actions taken)
        if init_actions is None:
            init_actions = np.zeros(N,"int")

        if init_observs is None:
            log_b = np.tile(log_pi[:,None],(1,N)) #S x N                        
        else:
            if init_observs_missing_mask is None:
                log_obs = np.sum(stat.norm.logpdf(init_observs[:,None,:], #N x 1 x D
                    np.transpose(O_means[:,:,init_actions],[2,1,0]), #N x S x D
                    np.transpose(O_sds[:,:,init_actions],[2,1,0])), 2).T   #at end: S x N
            else:
                log_obs = np.sum( init_observs_missing_mask[:,None,:] * 
                    stat.norm.logpdf(init_observs[:,None,:], #N x 1 x D
                    np.transpose(O_means[:,:,init_actions],[2,1,0]), #N x S x D
                    np.transpose(O_sds[:,:,init_actions],[2,1,0])), 2).T #S x N
            log_b = log_pi[:,None] + log_obs #S x N          

        beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'

    masked_observs = np.copy(observs)
    masked_actions = np.copy(actions)
    if observs_missing_mask is not None:
        masked_observs_missing_mask = np.copy(observs_missing_mask)

    for t in range(max_T):
        this_actions = actions[:,t] 
        mask = this_actions!=-1 #mask over all trajs, even those that ended
        rho_mask = masked_actions[:,t]!=-1 #variable size mask
        
        b_alphas = np.dot(beliefs[rho_mask,:],V[0].T)/alpha_temp 
        exp_balpha = np.exp(b_alphas - np.max(b_alphas,1)[:,None])
        alpha_probs = exp_balpha / np.sum(exp_balpha,1)[:,None]
        all_action_probs = np.dot(alpha_probs,V[1])        

        #filter out and renormalize using mask for guardrails...based just off of was beh_prob below a certain thresh
        if gr_safety_thresh > 0:
            gr_mask = all_beh_probs[mask,t,:] >= gr_safety_thresh

            #TODO: unmask the action that was actually taken, to allow for 
            # the ability to place probability mass on the actual behavior action,
            # since there are some cases where the est. beh probs assign very low 
            # prob to the action actually taken (kNN may not be great...)

            #hard reset to mask on actions taken, on top of grs
            gr_mask[np.arange(np.sum(rho_mask)),this_actions[mask]] = True

            all_action_probs = all_action_probs * gr_mask
            all_action_probs += 1e-4
            all_action_probs = all_action_probs / np.sum(all_action_probs,1)[:,None]

        action_probs = all_action_probs[np.arange(np.sum(rho_mask)),this_actions[mask]]
        action_probs = np.where(action_probs<EPS_ACTPROB,EPS_ACTPROB,action_probs) #fix very small probs
        log_action_probs = np.log(action_probs)

        old_logrhos = np.concatenate([old_logrhos,logrhos[np.logical_not(rho_mask)]]) #cache old rhos; need for denom
        logrhos = logrhos[rho_mask] + log_action_probs - log_behav_action_probs[mask,t]

        if clip:
            logrhos = np.clip(logrhos,np.log(clip_bds[0]),np.log(clip_bds[1]))

        #cache metrics of interest 
        ESS_noprune.append( np.exp( 2*logsumexp(np.concatenate([logrhos,old_logrhos])) - 
            logsumexp(2*np.concatenate([logrhos,old_logrhos])) ))
        if operate_in_logspace: #can't operate in log-domain, negative rewards :(
            logCWPDIS_nums_noprune.append(logsumexp(logrhos+np.log(rewards[mask,t]+EPS))) #implicitly uses old_rhos, but those rewards are all 0...
        else:
            CWPDIS_nums_noprune.append(np.sum(rewards[mask,t]*np.exp(logrhos)))
        logCWPDIS_denoms_noprune.append(logsumexp(np.concatenate([logrhos,old_logrhos])))

        if prune_num > 0: #prune top K rhos at each time...
            all_logrhos = np.concatenate([logrhos,old_logrhos])
            thresh = np.sort(all_logrhos)[::-1][prune_num-1]

            pruned_logrhos = np.where(logrhos>=thresh,np.log(EPS),logrhos)
            pruned_old_logrhos = np.where(old_logrhos>=thresh,np.log(EPS),old_logrhos)

            ESS.append(np.exp( 2*logsumexp(np.concatenate([pruned_logrhos,pruned_old_logrhos])) - 
            logsumexp(2*np.concatenate([pruned_logrhos,pruned_old_logrhos])) ))

            if operate_in_logspace: #can't operate in log-domain, negative rewards :(
                logCWPDIS_nums.append(logsumexp(pruned_logrhos+np.log(rewards[mask,t]+EPS)))
            else:
                CWPDIS_nums.append(np.sum(rewards[mask,t]*np.exp(pruned_logrhos)))
            
            logCWPDIS_denoms.append(logsumexp(np.concatenate([pruned_logrhos,pruned_old_logrhos])))
        else: #just use them as-is...
            ESS.append(ESS_noprune[-1])
            if operate_in_logspace: #can't operate in log-domain, negative rewards :(
                logCWPDIS_nums.append(logCWPDIS_nums_noprune[-1]) 
            else:
                CWPDIS_nums.append(CWPDIS_nums_noprune[-1])
            logCWPDIS_denoms.append(logCWPDIS_denoms_noprune[-1])


        if cached_beliefs is not None:
            beliefs = cached_beliefs[mask,t+1,:]
        else:

            if observs_missing_mask is None:
                log_obs = np.sum(stat.norm.logpdf(masked_observs[rho_mask,None,t,:],
                    np.transpose(O_means[:,:,masked_actions[rho_mask,t]],[2,1,0]),
                    np.transpose(O_sds[:,:,masked_actions[rho_mask,t]],[2,1,0])),2)
            else:
                log_obs = np.sum(masked_observs_missing_mask[rho_mask,None,t,:] *
                    stat.norm.logpdf(masked_observs[rho_mask,None,t,:],
                    np.transpose(O_means[:,:,masked_actions[rho_mask,t]],[2,1,0]),
                    np.transpose(O_sds[:,:,masked_actions[rho_mask,t]],[2,1,0])),2)

            #T: S' x S x A 
            lb = np.log(beliefs[rho_mask,:]+EPS) # N x S
            log_T_b = log_T[:,:,this_actions[mask]] + lb.T[None,:,:] # S' x S x N

            #assumes we filter without rewards
            log_b = log_obs.T + logsumexp(log_T_b,1) #S' x N
            beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'

            masked_observs = masked_observs[rho_mask,:,:]
            if observs_missing_mask is not None:
                masked_observs_missing_mask = masked_observs_missing_mask[rho_mask,:,:]
        masked_actions = masked_actions[rho_mask,:]

    if operate_in_logspace:
        CWPDIS_obj = np.exp(logsumexp( np.arange(max_T)*np.log(gamma) +
            np.array(logCWPDIS_nums) - np.array(logCWPDIS_denoms) ))
        CWPDIS_obj_noprune = np.exp(logsumexp( np.arange(max_T)*np.log(gamma) +
            np.array(logCWPDIS_nums_noprune) - np.array(logCWPDIS_denoms_noprune) ))
    else:
        CWPDIS_obj = np.sum(np.power(gamma,np.arange(max_T))*
                            np.array(CWPDIS_nums)/np.exp(np.array(logCWPDIS_denoms)))
        CWPDIS_obj_noprune = np.sum(np.power(gamma,np.arange(max_T))*
                            np.array(CWPDIS_nums_noprune)/np.exp(np.array(logCWPDIS_denoms_noprune)))

    #NOTE: ESS_noprune, not ESS!!! Penalize wrt ESS where we don't prune
    RL_obj = CWPDIS_obj - ESS_penalty*np.sum(1/np.sqrt(np.array(ESS_noprune)))

    #light regularization on alpha vectors themselves...
    if update_V and V_penalty > 0:
        RL_obj = RL_obj - V_penalty*np.sum(np.power(V[0],2))

    if rescale_obj is not None:
        RL_obj = RL_obj*rescale_obj

    if operate_in_logspace:
        return (-RL_obj,(V,CWPDIS_obj,np.array(ESS),np.array(logCWPDIS_nums),
            np.array(logCWPDIS_denoms),np.array(ESS_noprune),CWPDIS_obj_noprune))
    else:
        return (-RL_obj,(V,CWPDIS_obj,np.array(ESS),np.array(CWPDIS_nums),
            np.exp(np.array(logCWPDIS_denoms)),np.array(ESS_noprune),CWPDIS_obj_noprune))



def get_beliefs(params,seq_lens,actions,observs,
    init_observs=None,init_actions=None,
    observs_missing_mask=None,
    init_observs_missing_mask=None):
    """ helper func to get all beliefs """

    pi,T,O,R = params
    O_means,O_sds = O # D x S x A
    EPS = 1e-16
    log_T = np.log(T+EPS)
    log_pi = np.log(pi+EPS)

    N = n_traj = len(seq_lens)
    max_T = int(np.max(seq_lens))

    n_S = len(pi)

    all_beliefs = -1*np.ones((n_traj,max_T+1,n_S))

    ### get initial beliefs from prior pi & initial obs (observed before any actions taken)
    if init_actions is None:
        init_actions = np.zeros(N,"int")

    if init_observs is None:
        log_b = np.tile(log_pi[:,None],(1,N)) #S x N                        
    else:
        if init_observs_missing_mask is None:
            log_obs = np.sum(stat.norm.logpdf(init_observs[:,None,:], #N x 1 x D
                np.transpose(O_means[:,:,init_actions],[2,1,0]), #N x S x D
                np.transpose(O_sds[:,:,init_actions],[2,1,0])), 2).T   #at end: S x N
        else:
            log_obs = np.sum( init_observs_missing_mask[:,None,:] * 
                stat.norm.logpdf(init_observs[:,None,:], #N x 1 x D
                np.transpose(O_means[:,:,init_actions],[2,1,0]), #N x S x D
                np.transpose(O_sds[:,:,init_actions],[2,1,0])), 2).T #S x N
        log_b = log_pi[:,None] + log_obs #S x N          

    beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'
    all_beliefs[:,0,:] = beliefs
 
    masked_observs = np.copy(observs)
    masked_actions = np.copy(actions)
    if observs_missing_mask is not None:
        masked_observs_missing_mask = np.copy(observs_missing_mask)

    for t in range(max_T):
        this_actions = actions[:,t] 
        mask = this_actions!=-1 #mask over all trajs, even those that ended
        rho_mask = masked_actions[:,t]!=-1 #variable size mask

        if observs_missing_mask is None:
            log_obs = np.sum(stat.norm.logpdf(masked_observs[rho_mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[rho_mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[rho_mask,t]],[2,1,0])),2)
        else:
            log_obs = np.sum(masked_observs_missing_mask[rho_mask,None,t,:] *
                stat.norm.logpdf(masked_observs[rho_mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[rho_mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[rho_mask,t]],[2,1,0])),2)

        #T: S' x S x A 
        lb = np.log(beliefs[rho_mask,:]+EPS) # N x S
        log_T_b = log_T[:,:,this_actions[mask]] + lb.T[None,:,:] # S' x S x N

        #assumes we filter without rewards
        log_b = log_obs.T + logsumexp(log_T_b,1) #S' x N
        beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'

        all_beliefs[mask,t+1,:] = beliefs

        masked_observs = masked_observs[rho_mask,:,:]
        masked_actions = masked_actions[rho_mask,:]
        if observs_missing_mask is not None:
            masked_observs_missing_mask = masked_observs_missing_mask[rho_mask,:,:]

    return all_beliefs


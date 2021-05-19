#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utils for input-output hidden Markov models (IO-HMMs).

We have an HMM that also takes into account actions, ie 
transition and observation matrices have an extra dim for 
which action was taken.

@author: josephfutoma
"""

import sys

import autograd
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad as vg
from autograd.scipy.misc import logsumexp
from autograd.misc.flatten import flatten_func
import autograd.scipy.stats as stat

from util import (softplus,inv_softplus,to_proba_vec,
                  from_proba_vec,to_proba_3darr,from_proba_3darr)

EPS = 1e-16 

def to_natural_params(params):
    pi,T,O,R = params
    (O_means,O_sds) = O

    pi_reals = from_proba_vec(pi)
    T_reals = from_proba_3darr(T)
    O_sds_reals = inv_softplus(O_sds)
    
    return (pi_reals,T_reals,(O_means,O_sds_reals),R)

def to_params(nat_params):
    pi_reals,T_reals,O_reals,R = nat_params
    (O_means,O_sds_reals) = O_reals

    pi = to_proba_vec(pi_reals)
    T = to_proba_3darr(T_reals)
    O_sds = softplus(O_sds_reals)+EPS

    return (pi,T,(O_means,O_sds),R)

#####
##### HMM objective function terms, NOT using reward
#####

def log_prior(nat_params,alpha_pi=10,alpha_T=.1,O_mean_var=100,O_logsd_var=1):
    """
    prior term for regularization
    """  
    pi,T,O,R = to_params(nat_params)
    O_means,O_sds = O
    
    log_pi = np.log(pi+1e-16)
    log_T = np.log(T+1e-16)
    
    obj = 0
    
    obj += (alpha_pi-1)*np.sum(log_pi)
    obj += (alpha_T-1)*np.sum(log_T)
    obj += -1/(2*O_mean_var)*np.sum(np.power(O_means,2))    
    # obj += -1/(2*O_logsd_var)*np.sum(np.power(O_log_sds,2))    

    return -obj

def HMM_marginal_likelihood(nat_params,observs,actions,init_observs=None,
    init_actions=None,observs_missing_mask=None,init_observs_missing_mask=None):

    pi,T,O,R = to_params(nat_params)
    O_means,O_sds = O
    
    log_pi = np.log(pi+1e-16)
    log_T = np.log(T+1e-16)
    
    N = np.shape(observs)[0]
    max_T = observs.shape[1]
   
    obj = 0

    if init_actions is None:
        init_actions = np.zeros(N,"int")

    #option for whether or not we have an initial obs before taking a0.
    #either: we're in s0, take a0, see o1, r1, move to s1,...
    #   OR see o0 from s0, then take a0, see o1, r1, move to s1,...
    #EDIT: also allow init_actions. ie an action a_-1 before o0 & s0,
    # so that our initial obs p(o0 | s0, a_-1) can use the same emissions
    # as other obs; otherwise either need to have separate emissions for 
    # this initial state (= extra params...), or assume a_-1 = eg 0/null action

    if init_observs is None:
        log_alpha = np.tile(log_pi[:,None],(1,N)) #S x N                        
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
        log_alpha = log_pi[:,None] + log_obs #S x N                          

    ### TODO: option to return back log_alpha as well so we have beliefs to reuse
    # for the OPE downstream, rather than computing twice.....

    log_Z = logsumexp(log_alpha,0)
    log_alpha = log_alpha - log_Z 
    obj = obj + np.sum(log_Z)
    
    #TODO why do we even have these masks...??? there must be a cleaner way 
    # that doesn't force us to keep copying data??

    masked_observs = np.copy(observs)
    masked_actions = np.copy(actions)
    if observs_missing_mask is not None:
        masked_observs_missing_mask = np.copy(observs_missing_mask)
    
    for t in range(max_T):    
        mask = np.logical_not(np.isinf(masked_observs[:,t,0])) #variable size mask; only get obs actually go this long

        log_Talpha = logsumexp(log_T[:,:,masked_actions[mask,t]] 
            + log_alpha[:,mask],1) #S x N

        if observs_missing_mask is None:
            log_obs = np.sum(stat.norm.logpdf(masked_observs[mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[mask,t]],[2,1,0])),2)
        else:
            log_obs = np.sum(masked_observs_missing_mask[mask,None,t,:] *
                stat.norm.logpdf(masked_observs[mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[mask,t]],[2,1,0])),2)

        log_alpha = log_obs.T + log_Talpha             
        log_Z = logsumexp(log_alpha,0)
        log_alpha -= log_Z
        obj = obj + np.sum(log_Z)

        masked_observs = masked_observs[mask,:,:]
        masked_actions = masked_actions[mask,:] 
        if observs_missing_mask is not None:
            masked_observs_missing_mask = masked_observs_missing_mask[mask,:,:]

    return -obj

def MAP_objective(nat_params,observs,actions,
    init_observs=None,init_actions=None,
    observs_missing_mask=None,init_observs_missing_mask=None):
    
    #count up number of individual obs in likelihood, to get scale factor
    obs_ct = 0 
    if init_observs is not None:
        if init_observs_missing_mask is None:
            N = observs.shape[0]
            n_obs = observs.shape[2]
            obs_ct += n_obs*N
        else:
            obs_ct += np.sum(init_observs_missing_mask)
    if observs_missing_mask is None:
        obs_ct += np.sum(np.logical_not(np.isinf(observs)))
    else:
        obs_ct += np.sum(observs_missing_mask)
    scale = 1/obs_ct

    return scale*( log_prior(nat_params)+HMM_marginal_likelihood(nat_params,
        observs,actions,init_observs,init_actions,
        observs_missing_mask,init_observs_missing_mask) )

#####
##### E and M step functions to run EM, NOT using rewards
#####

def forward_backward_Estep(nat_params,observs,actions,rewards,R_sd=0.01,
    get_xi=True,init_observs=None,init_actions=None,
    observs_missing_mask=None,init_observs_missing_mask=None):
    """
    E step by using forward-backward algorithm
    Return all sufficient statistics we need for M step,
        AND return the marginal likelihood for tracking!
    """
    pi,T,O,R = to_params(nat_params)
    O_means,O_sds = O
       
    log_pi = np.log(pi+1e-16)
    log_T = np.log(T+1e-16)
    
    max_T = observs.shape[1]
    n_S = log_pi.shape[0]
    n_A = R.shape[1]
    N = observs.shape[0]
    n_obs = observs.shape[2]
    
    if init_actions is None:
        init_actions = np.zeros(N,"int")

    ### forward

    obj = 0 #HMM marginal likelihood objective
    log_alpha = np.zeros((max_T+1,n_S,N)) #cache for forward pass
    log_obs_all = [] #cache for backward pass

    #option for whether or not we have an initial obs before taking a0.
    #either: we're in s0, take a0, see o1, r1, move to s1...
    #   OR see o0 from s0, then take a0, see o1, r1, move to s1

    if init_observs is None:
        l_alpha = np.tile(log_pi[:,None],(1,N)) #S x N                        
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
        l_alpha = log_pi[:,None] + log_obs #S x N                          
    
    #don't cache initial log_obs at first state, as it's only used in forward pass
    #and as part of overall likelihood
      
    log_Z = logsumexp(l_alpha,0)
    l_alpha -= log_Z
    obj = obj + np.sum(log_Z)
    log_alpha[0,:,:] = l_alpha
        
    masked_observs = np.copy(observs)
    masked_actions = np.copy(actions)
    if observs_missing_mask is not None:
        masked_observs_missing_mask = np.copy(observs_missing_mask)
    
    for t in range(max_T):    
        mask = np.logical_not(np.isinf(masked_observs[:,t,0])) #only get obs actually go this long
        mask_orig = np.logical_not(np.isinf(observs[:,t,0])) #mask with same size as orig input

        log_Talpha = logsumexp(log_T[:,:,masked_actions[mask,t]] 
            + log_alpha[t,:,mask_orig].T,1) #S x N
        
        if observs_missing_mask is None:
            log_obs = np.sum(stat.norm.logpdf(masked_observs[mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[mask,t]],[2,1,0])),2)
        else:
            log_obs = np.sum(masked_observs_missing_mask[mask,None,t,:] *
                stat.norm.logpdf(masked_observs[mask,None,t,:],
                np.transpose(O_means[:,:,masked_actions[mask,t]],[2,1,0]),
                np.transpose(O_sds[:,:,masked_actions[mask,t]],[2,1,0])),2)


        l_alpha = log_obs.T + log_Talpha 
        
        log_obs_all.append(log_obs)
        
        log_Z = logsumexp(l_alpha,0)
        l_alpha -= log_Z
        log_alpha[t+1,:,mask_orig] = l_alpha.T
        obj += np.sum(log_Z)
        
        masked_observs = masked_observs[mask,:]
        masked_actions = masked_actions[mask,:]
        if observs_missing_mask is not None:
            masked_observs_missing_mask = masked_observs_missing_mask[mask,:,:]

    ### backward

    log_beta = np.zeros((max_T+1,n_S,N)) #cache for forward pass
    for t in range(max_T-1,-1,-1):
        mask = np.logical_not(np.isinf(observs[:,t,0])) #only get obs actually go this long
        log_obs_beta = log_obs_all[t] + log_beta[t+1,:,mask] #N x S
        log_beta[t,:,mask] = logsumexp(log_T[:,:,actions[mask,t]] 
            + np.transpose(log_obs_beta[None,:,:],[2,0,1]),0).T

    #posterior smoothed prob of state | all obs
    log_gamma = log_alpha+log_beta
    log_gamma -= logsumexp(log_gamma,1)[:,None,:]
    gamma = np.exp(log_gamma)
    
    #count up number of individual obs in likelihood, to get scale factor
    obs_ct = 0 
    if init_observs is not None:
        if init_observs_missing_mask is None:
            obs_ct += n_obs*N
        else:
            obs_ct += np.sum(init_observs_missing_mask)
    if observs_missing_mask is None:
        obs_ct += np.sum(np.logical_not(np.isinf(observs)))
    else:
        obs_ct += np.sum(observs_missing_mask)
    scale = 1/obs_ct
    MAP_obj = scale*(log_prior(nat_params)-obj) 
    #EDIT: minus sign, so it's minus MAP, in align w MAP_objective

    #don't save xi just accumulate sstats
    if get_xi:    
        #posterior smoothed transition probs

        E_Njka = np.zeros((n_S,n_S,n_A)) #S,S',A
        for t in range(max_T):
            xi = np.zeros((1,n_S,n_S,N)) #T,S->S',N

            mask = np.logical_not(np.isinf(observs[:,t,0])) #only get obs actually go this long
            log_obs_beta = log_obs_all[t] + log_beta[t+1,:,mask] #N x S
            log_alpha_obs_beta = log_alpha[t,:,mask][:,:,None] + log_obs_beta[:,None,:] #N x S x S'              
            log_xi = log_T[:,:,actions[mask,t]] + np.transpose(log_alpha_obs_beta,[2,1,0]) #S' x S x N
            log_xi -= logsumexp(log_xi,(0,1))  

            xi[0,:,:,mask] = np.transpose(np.exp(log_xi),[2,1,0])    # N x S x S  

            for a in range(n_A):
                ind = actions[:,t]==a
                E_Njka[:,:,a] += np.sum(xi[0,:,:,ind],0)     

        return MAP_obj,gamma,E_Njka  
    else:
        return MAP_obj,gamma

def M_step_just_reward(nat_params,observs,actions,rewards,gam,
    R_sd=0.01,R_mean_var=25):
    """
    use sufficient statistics from E step to update HMM params
    
    note that we'll use MAP estimation, and same hyperparams as in log_prior 
    term in gradient-based objective
    """
    pi,T,O,R = to_params(nat_params)
    O_means,O_sds = O
   
    max_T = observs.shape[1]
    n_S = R.shape[0]
    n_A = T.shape[2]

    #combine suff stats from E step with data
    
    ### update R
    E_Nka_rew = EPS*np.ones((n_S,n_A))
    E_Rka = np.zeros((n_S,n_A))
    for t in range(max_T):
        for a in range(n_A):
            ind = actions[:,t]==a 
            E_Nka_rew[:,a] += np.sum(gam[t,:,ind],0) #NOTE: fixed indexing bug here, t not t+1!!!!!
            E_Rka[:,a] += np.einsum('ab,a->b',gam[t,:,ind],rewards[ind,t]) #NOTE: fixed indexing bug here, t not t+1!!!!!
    
    R_hat = E_Rka/E_Nka_rew * (E_Nka_rew*R_mean_var/(E_Nka_rew*R_mean_var+R_sd**2))

    return R_hat

def M_step(nat_params,observs,actions,rewards,gamma,E_Njka,R_sd=.01,
           alpha_pi=25,alpha_T=.1,O_mean_var=100,R_mean_var=25,
           init_observs=None,init_actions=None,
           observs_missing_mask=None,init_observs_missing_mask=None):
    """
    use sufficient statistics from E step to update HMM params
    
    note that we'll use MAP estimation, and same hyperparams as in log_prior 
    term in gradient-based objective
    """
    pi,T,O,R = to_params(nat_params)
    O_means,O_sds = O
    
    O_dim = O_means.shape[0]
    max_T = observs.shape[1]
    n_S = pi.shape[0]
    N = observs.shape[0]
    n_A = T.shape[2]

    #combine suff stats from E step with data
    
    ### update pi
    E_Nk0 = np.sum(gamma[0,:,:],1)
    pi_hat = (E_Nk0 + alpha_pi - 1)/(N + n_S*alpha_pi - n_S)
    pi_hat[pi_hat<0] = EPS
    pi_hat /= np.sum(pi_hat) #just in case

    ### update T
    # E_Njka = np.zeros((n_S,n_S,n_A)) #S,S',A
    # for t in range(max_T):
        # for a in range(n_A):
            # ind = actions[:,t]==a
            # E_Njka[:,:,a] += np.sum(xi[t,:,:,ind],0)
    T_hat = (E_Njka + alpha_T - 1)/(np.sum(E_Njka,1)[:,None,:] + n_S*alpha_T - n_S)
    T_hat = np.transpose(T_hat,[1,0,2]) #S',S,A; in line with T
    T_hat[T_hat<0] = EPS
    T_hat /= np.sum(T_hat,0) #just in case
    
    ### update O,R
    if observs_missing_mask is None:
        E_Nka_obs = EPS*np.ones((n_S,n_A))
    else:
        E_Nka_obs = EPS*np.ones((O_dim,n_S,n_A))

    E_xka = np.zeros((O_dim,n_S,n_A))
    E_x2ka = np.zeros((O_dim,n_S,n_A))

    E_Rka = np.zeros((n_S,n_A))
    E_Nka_rew = EPS*np.ones((n_S,n_A))    

    for t in range(max_T):
        for a in range(n_A):
            ind = actions[:,t]==a
            ### gamma: max_T+1 x S x N
            this_gam = gamma[t+1,:,ind] #N x S

            if observs_missing_mask is None:
                E_Nka_obs[:,a] += np.sum(this_gam,0)     
                this_obs = observs[ind,t,:]
            # NOTE: careful here!!
            else:
                ### TODO: CHECK THIS
                this_mask = observs_missing_mask[ind,t,:] #N x O
                this_obs = this_mask * observs[ind,t,:]
                for o in range(O_dim):
                    E_Nka_obs[o,:,a] += np.sum(this_mask[:,o] * this_gam.T,1)

            # N x S , N x O -> O x S
            E_xka[:,:,a] += np.einsum('ab,ac->cb',this_gam,this_obs)
            E_x2ka[:,:,a] += np.einsum('ab,ac->cb',this_gam,this_obs**2)

            #rewards are separate!!
            E_Nka_rew[:,a] += np.sum(gamma[t,:,ind],0)       
            E_Rka[:,a] += np.einsum('ab,a->b',gamma[t,:,ind],rewards[ind,t]) #NOTE: t and not t+1 in gamma!!!!!


    ### add in effect of initial obs, if there are any...

    if init_actions is None:
        init_actions = np.zeros(N,"int")

    if init_observs is not None:
        for a in range(n_A):
            ind = init_actions==a
            this_gam = gamma[0,:,ind]

            if init_observs_missing_mask is None:
                this_obs = init_observs[ind,:]
                E_Nka_obs[:,a] += np.sum(this_gam,0)  
            else:
                this_mask = init_observs_missing_mask[ind,:] #N x O
                this_obs = this_mask * init_observs[ind,:]
                for o in range(O_dim):
                    E_Nka_obs[o,:,a] += np.sum(this_mask[:,o] * this_gam.T,1)

            E_xka[:,:,a] += np.einsum('ab,ac->cb',this_gam,this_obs)
            E_x2ka[:,:,a] += np.einsum('ab,ac->cb',this_gam,this_obs**2)


    # old way, still works fine. hooray broadcasting!
    O_means_hat = E_xka/E_Nka_obs #* (E_Nka_obs*O_mean_var/(E_Nka_obs*O_mean_var+O_sds**2))
    O_sds_hat = np.sqrt(np.abs(E_x2ka/E_Nka_obs-O_means_hat**2))+1e-6

    R_hat = E_Rka/E_Nka_rew #* (E_Nka_rew*R_mean_var/(E_Nka_rew*R_mean_var+R_sd**2))

    return pi_hat,T_hat,(O_means_hat,O_sds_hat),R_hat

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:30:44 2018

Implementation of PBVI

@author: josephfutoma
"""

import sys
import importlib

import autograd
import autograd.numpy as np
import autograd.scipy.stats as stat
from autograd.scipy.misc import logsumexp
# import matplotlib.pyplot as plt

from util import draw_discrete_gumbeltrick,draw_discrete

def initialize_B(params,V_min,gamma,n_expandB_iters=10):
    pi,T,O,R = params
    n_S = len(pi)
    
    high = .99
    B_init = (1-high)/(n_S-1)*np.ones((n_S+1,n_S))
    B_init[0,:] = 1/n_S
    for i in range(n_S):
        B_init[i+1,i] = high      
      
    ### expand using the true dynamics to get a decent sized (~250) set of beliefs
    n_B = np.shape(B_init)[0]
    # V_min = np.min(R_true)/(1-gamma) #seems reasonable to assume we know min possible reward
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]
    for i in range(n_expandB_iters):
        _,B_init = expand_B(V,B_init,T,O,R,V_min,gamma)
    
    return B_init 


##########
########## functions to run a policy in a given environment
########## 

def sim_step(s,a,T,O,R,rng=None):
    """
    given state and action, return a reward and new observation
    """
    O_dims = np.shape(O)[1]
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A    
    
    new_s = draw_discrete(T[:,s,a],rng)
    r = R[s,a]    
    
    if rng is None:
        o = np.random.normal(0,1,O_dims)
    else:   
        o = rng.normal(0,1,O_dims)
    o = O_means[:,new_s,a] + O_sds[:,new_s,a]*o
    
    return new_s,o,r



def sim_step_bydim(s,a,T,O,R,rng=None,tiger_env=None):
    """
    given state and action, return a reward and new observation
    """
    n_env = len(O)
    try:
        n_dim_per_env = np.shape(O[0][0])[0]
    except:
        n_dim_per_env = 1
    n_dim = n_env*n_dim_per_env 

    r = R[0][s[0],a]    


    if tiger_env == 'gmm':
        new_s = []
        new_obs = []
        for d in range(n_env):
            new_s.append(draw_discrete(T[d][:,s[d],a],1,rng))
            if rng is None:
                o = O[d](new_s[-1],a,n=1)
            else:   
                o = O[d](new_s[-1],a,n=1,rng=rng)
            new_obs.append(o)

    else:
        new_s = []
        new_obs = []
        for d in range(n_env):
            new_s.append(draw_discrete(T[d][:,s[d],a],1,rng))
            if rng is None:
                o = np.random.normal(0,1,n_dim_per_env)
            else:   
                o = rng.normal(0,1,n_dim_per_env)
            o = O[d][0][:,new_s[-1],a] + O[d][1][:,new_s[-1],a]*o
            new_obs.append(o)
    
    return new_s,np.array(new_obs).flatten(),r


def update_belief_reward(belief,obs,action,reward,log_T,O,R,R_sd): 
    """
    update & return new belief from old belief, new obs & reward, and action
    
    NOTE: In settings where true model not known, 
    these T and O should be estimated and *not* the truth.
    """
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A       

    log_obs = np.sum(stat.norm.logpdf(obs,O_means[:,:,action].T,O_sds[:,:,action].T),1) #S'
    log_rew = stat.norm.logpdf(reward,R[:,action],R_sd) #S
    
    #T: S' x S 
    lb = np.log(belief+1e-16) # S
    log_T_b_r = log_T[:,:,action] + lb[None,:] + log_rew[None,:]# S' x S

    log_b = log_obs + logsumexp(log_T_b_r,1)
    return np.exp(log_b - logsumexp(log_b))


def update_belief(belief,obs,action,log_T,O,obs_mask=None): 
    """
    update & return new belief from old belief, new obs & reward, and action
    
    NOTE: In settings where true model not known, 
    these T and O should be estimated and *not* the truth.
    """
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A       

    if obs_mask is None:
        log_obs = np.sum(stat.norm.logpdf(
            obs,O_means[:,:,action].T,O_sds[:,:,action].T),1) #S'
    else:
        log_obs = np.sum(obs_mask*stat.norm.logpdf(
            obs,O_means[:,:,action].T,O_sds[:,:,action].T),1) #S'

    #T: S' x S 
    lb = np.log(belief+1e-16) # S
    log_T_b = log_T[:,:,action] + lb[None,:]# S' x S

    log_b = log_obs + logsumexp(log_T_b,1)
    return np.exp(log_b - logsumexp(log_b))

###
### TODO: convert this function so it evals a bunch of trajs & vectorize so fast
###

def run_softmax_policy(T,O,R,pi,R_sd,T_est,O_est,R_est,belief,
    obs_meas_probs = None,
    V=None,steps=1000,
    seed=8675309,tiger_env=True,temp=None,belief_with_reward=False):
    """
    runs a policy for a period of time, and records trajectories.
    policy is parameterized by a value function composed of alpha vectors.
    
    inputs:
        V: if None, use a random policy where we just select random actions.
            Otherwise should be value function as represented in PBVI func
        
    NOTE: this func is highly specific to tiger env / stacked envs...

    outputs:
        full trajectories 
    """
    rng = np.random.RandomState(seed)

    n_A = np.shape(R[0])[1]
    n_S = np.shape(R[0])[0]

    n_env = len(O)
    try:
        n_dim_per_env = np.shape(O[0][0])[0]
    except:
        n_dim_per_env = 1
    n_dim = n_env*n_dim_per_env 
    
    #default values
    if V is None:
        action_probs = 1/n_A*np.ones(n_A) 

    if obs_meas_probs is None:
        obs_meas_probs = np.ones(n_dim)


    # if belief is None:
    #     belief = pi  
    # if T_est is None:
    #     T_est = T     
    # if O_est is None:
    #     O_est = O
    # if R_est is None:
    #     R_est = R
    
    if temp is not None:
        assert(temp>0)

    log_T_est = np.log(T_est+1e-16)
    
    states = []
    beliefs = []
    actions = []
    rewards = []
    observations = []
        
    #initial state
    state = []
    for d in range(n_env):
        state.append(draw_discrete(pi[d],1,rng))
    states.append(state)    
    beliefs.append(belief)
        
    #loop & sim trajectory
    for t in range(steps):
        if V is None: #random actions 
            action = draw_discrete(action_probs,1,rng)
        else: #use our learned value function from planning to greedily select optimal action            
            if temp is None: #deterministic policy
                b_alpha = np.dot(V[0],belief)
                alpha = np.argmax(b_alpha)
                action = np.argmax(V[1][alpha])
            else: #stochastic policy defined by softmax over alphas w/ temperature 
                b_alpha = np.dot(V[0],belief)/temp    
                exp_alpha = np.exp(b_alpha-np.max(b_alpha))
                alpha_probs = exp_alpha/np.sum(exp_alpha)
                
                action_probs = np.sum(alpha_probs[:,None] * V[1],0)
                action = draw_discrete(action_probs,1,rng)
        
        state,obs,reward = sim_step_bydim(state,action,T,O,R,rng,tiger_env)
        #randomly mask out part of observations according to fixed probs
        obs_mask = rng.uniform(0,1,n_dim) <= obs_meas_probs

        # if belief_with_reward:
            # belief = update_belief_reward(belief,obs,action,reward,log_T_est,O_est,R_est,R_sd)
        # else:
        belief = update_belief(belief,obs,action,log_T_est,O_est,obs_mask)
               
        states.append(state)
        beliefs.append(belief)
        actions.append(action)
        rewards.append(reward)
        observations.append(obs)
    
        if tiger_env and action!=n_A-1: #if chose and didn't listen, end for tiger scenarios
            break
    
    return ( np.array(states),np.array(beliefs),np.array(actions),
            np.array(rewards),np.array(observations) )    
    


##########
########## functions to run PBVI
########## 

##### https://stackoverflow.com/questions/18365073/why-is-numpys-einsum-faster-than-numpys-built-in-functions
##### Note: np.einsum should be faster than any numpy function unless it's np.dot...???

def update_V_softmax(V,B,T,O,R,gamma,eps=None,PBVI_temps=None,
                     max_iter=100,verbose=False,n_samps=100,seed=False):
    """
    inputs:
        V (list):
            V[0]: n_B x n_S array of alpha-vector values for each belief
            V[1]: n_B array, denoting which action generated each alpha-vector                
        B: n_B x n_S array of belief states to be updated
        
    optional inputs:
        
        
    outputs:
        V (same as input), updated
    """
    if PBVI_temps is None:
        temp1=.01
        temp2=.01
        temp3=.01
    else:
        temp1 = PBVI_temps[0]
        temp2 = PBVI_temps[1]
        temp3 = PBVI_temps[2]
        
    if seed: #testing
        np.random.seed(711)
    
    n_B = np.shape(B)[0]
    n_V = np.shape(B)[0]
    n_A = np.shape(R)[1]
    n_S = np.shape(R)[0]
    O_dims = np.shape(O)[1]
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A
    if eps is None:
        eps = 0.01*n_S    

    #### no reason to resample O each iteration; so sample obs beforehand and cache
    O_samps = np.random.normal(0,1,(n_samps,O_dims,n_S,n_A)) #K x D x S x A
    O_samps = O_means + O_sds*O_samps
    
    #precompute and cache b^ao for sampled observations...
    O_logprob = np.sum(stat.norm.logpdf(O_samps[:,:,:,:,None], #K x D x S x A x 1
                                        np.transpose(O_means,[0,2,1])[:,None,:,:],
                                        np.transpose(O_sds,[0,2,1])[:,None,:,:],),1)
    #K: # samples drawn
    log_B = np.log(B+1e-16) # B x S
    log_T = np.log(T+1e-16) #S' x S x A
    log_TB =  logsumexp(log_B[None,:,:,None] + log_T[:,None,:,:],2)# S' x S

    log_bao = np.transpose(O_logprob[:,:,None,:,:] + log_TB[None,:,:,:,None],[2,0,3,1,4])
    b_ao =  np.exp(log_bao - logsumexp(log_bao,4)[:,:,:,:,None]) #B x K x A x S' x S
    
    for ct in range(max_iter):     
        
        old_V = np.array(V[0],copy=True)
                
        alpha_bao = np.einsum('ab,cdefb->acdef',V[0],b_ao)/temp1 #V x B x K x A x S'
        
        #softmax
        exp_alpha_bao = np.exp(alpha_bao - np.max(alpha_bao,0)) #V x B x K x A x S'
        alpha_bao_probs = exp_alpha_bao/np.sum(exp_alpha_bao,0) #V x B x K x A x S'
        
        #soft mean
        prob_meta_obs = np.mean(alpha_bao_probs,axis=2) #V x B x A x S'
        
        alpha_aO_alpha2 = np.einsum('ab,bcd,efdb->efdac',V[0],T,prob_meta_obs) #V' x B x A x V x S     
        B_alpha_aO_alpha2 = np.einsum('ab,cadeb->cade',B,alpha_aO_alpha2)/temp2 #V' x B x A x V
        
        #softmax
        exp_aB = np.exp(B_alpha_aO_alpha2 - np.max(B_alpha_aO_alpha2,3)[:,:,:,None]) #V' x B x A x V
        aB_probs = exp_aB/np.sum(exp_aB,3)[:,:,:,None] #V' x B x A x V
        
        #soft mean
        avg_B_alpha_aO_alpha2 = np.sum(alpha_aO_alpha2 * aB_probs[:,:,:,:,None], axis=3) #V' x B x A x S
        
        alpha_ab = R.T + gamma*np.einsum('abcd->bcd',avg_B_alpha_aO_alpha2) #B x A x S 
        alpha_ab_B = np.einsum('ab,acb->ac',B,alpha_ab)/temp3 #B x A        
        
        #softmax
        exp_alpha_ab_B = np.exp(alpha_ab_B - np.max(alpha_ab_B,1)[:,None]) #B x A
        alpha_ab_B_probs = exp_alpha_ab_B/np.sum(exp_alpha_ab_B,1)[:,None]  #B x A
        
        #soft mean
        avg_alpha_abB = np.sum(alpha_ab * alpha_ab_B_probs[:,:,None], 1) #B x S
        
        V[0] = avg_alpha_abB #B x S; alpha-vecs
        V[1] = alpha_ab_B_probs #B x A; action probs for each alpha-vec
            
        diff = np.sum(np.abs(V[0]-old_V))
        
        #check for convergence
        if diff < eps:
            return V 
        
    if verbose:
        print("didn't converge during update :(" %np.sum(np.abs(V[0]-old_V)))        
    return V


def expand_B(V,B,T,O,R,V_min,gamma,eps=None,n_samps=25):
    """
    inputs:
        V (list):
            V[0]: n_B x n_S array of alpha-vector values for each belief
            V[1]: n_B array, denoting which action generated each alpha-vector                        
        B: n_B x n_S array of belief states to be expanded to other reachable beliefs
        
    optional inputs:
        eps: tolerance, new beliefs less than eps away from old beliefs will not be added
    
    outputs:
        V, with placeholder values added for new beliefs
        B, updated with new beliefs    
    """    
    n_S = np.shape(R)[0]
    n_A = np.shape(R)[1]
    n_B = np.shape(B)[0]
    if eps is None:
        eps = 0.01*n_S

    if n_B > 250:
        print("already have over 250 belief points. careful! not expanding more...")
    
    O_dims = np.shape(O)[1]
    O_means = O[0]; O_sds = O[1]
    
    #### no reason to resample O each iteration; so sample obs beforehand and cache
    O_samps = np.random.normal(0,1,(n_samps,O_dims,n_S,n_A))
    O_samps = O_means + O_sds*O_samps
    
    #precompute and cache b^ao for sampled observations...
    O_logprob = np.sum(stat.norm.logpdf(O_samps[:,:,:,:,None],
                                        np.transpose(O_means,[0,2,1])[:,None,:,:],
                                        np.transpose(O_sds,[0,2,1])[:,None,:,:],),1)
        
    log_B = np.log(B+1e-16) # B x S
    log_T = np.log(T+1e-16) #S' x S x A
    log_TB =  logsumexp(log_B[None,:,:,None] + log_T[:,None,:,:],2)# S' x S

    log_bao = np.transpose(O_logprob[:,:,None,:,:] + log_TB[None,:,:,:,None],[2,0,3,1,4])
    b_ao =  np.exp(log_bao - logsumexp(log_bao,4)[:,:,:,:,None]) #B x K x A x S' x S

    #innermost bit is size B x B' x k x A x S' x S 
    b_ao_alldiffs = np.transpose(np.einsum('abcdef->abcde',np.abs(b_ao[:,None,:] - B[None,:,None,None,None,:])),[0,2,3,4,1])
    b_ao_diffs = np.min(b_ao_alldiffs,4) #B x k x A x S': how far this belief expansion is from existing B
        
    #get all beliefs that were max distance away
    tmp_max = np.max(b_ao_diffs,axis=(1,2,3))
    inds = tmp_max>eps #which expansions from original beliefs were decently far away
    inds2 = np.isclose(b_ao_diffs[inds,:,:,:],tmp_max[inds,None,None,None],atol=1e-6,rtol=1e-6) #get everything that was close 
    new_B = b_ao[inds][inds2,:] #relaxed inds2 a bit to let more in to check...
    
    if np.shape(new_B)[0]>0:
    
        ### should do some sort of check in new_B for redundancy...it's ok
        # if we accept several new beliefs if they're all close to the max
        # but in different directions...not great if they're all still close by 
        new_nB = np.shape(new_B)[0]    
        
        #B x B pairwise differences between vectors
#        diffs = np.reshape(np.sum(np.abs(np.tile(new_B,(new_nB,1))-np.repeat(new_B,new_nB,axis=0)),1),(new_nB,new_nB))
#        
#        diffs2 = np.zeros((new_nB,new_nB))
#        for i in range(new_nB):
#            for ii in range(new_nB):
#                diffs2[i,ii] = np.sum(np.abs(new_B[i,:]-new_B[ii,:]))     
#             
#        diff_inds = diffs>eps
    

        #TODO: more efficient way?
        keep_inds = [0]
        for i in range(1,new_nB):
            diffs = np.sum(np.abs(new_B[keep_inds,:] - new_B[i,:]),1)         
            if np.all(diffs>eps):
                keep_inds.append(i)                    
        new_B = new_B[keep_inds,:]
    
    B = np.vstack([B,new_B])    
    #add values to existing value function
    V[0] = np.vstack([V[0],V_min*np.ones(np.shape(new_B))])
    n_B = np.shape(B)[0]
    V[1] = -1*np.ones(n_B)
        
    return V,B



def pbvi(T,O,R,gamma,B=None,V=None,max_iter=100,eps=.1,verbose=False,n_B_steps=3,
         max_V_iters=100):
    """
    main PBVI function.
    
    inputs:
        B: n_B x n_S array with initial belief set. if none, use pre-chosen one
        
    optional inputs:
        V: in case we're doing a warm start from a previous iteration
        max_iter:
        eps:
        
    outputs:
        V:
        B:
    """
    n_S = np.shape(R)[0]
    
    if B is None:
        high = .99
        B = (1-high)/(n_S-1)*np.ones((n_S+1,n_S))
        B[0,:] = 1/n_S
        for i in range(n_S):
            B[i+1,i] = high  

    n_B = np.shape(B)[0]
    if verbose:
        print("starting with %d beliefs..." %n_B)
        sys.stdout.flush()
    
    V_min = np.min(R)/(1-gamma)
    if V is None:
        #init value function: V[0] is array of alpha vectors, V[1] is corresponding action
        V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]
    
    V_maxdiff = np.inf #difference between successive value functions (alpha vectors)
    n_iter = 0
    
    while V_maxdiff > eps: #loop until V has converged or hit max_iter 
        if verbose:
            print("iter %d" %n_iter)
            sys.stdout.flush()
    
        #####
        ##### improvement step
        #####    
        if verbose:
            print("updating beliefs...")
            sys.stdout.flush()                 
        V = update_V_softmax(V,B,T,O,R,gamma,max_iter=max_V_iters)
     
        ##TODO: what is the stopping criteria here??
        #want to end after an update step and not an expand step...
        #after expand step we have a bunch of extra beliefs without a value
        
        if n_iter == max_iter:
            if verbose:
                print("updated in last iter, quitting...")
                sys.stdout.flush()           
            break
        
        #####
        ##### expansion step where we add new belief points to set
        #####   
        if verbose:
            print("expanding beliefs...")
            sys.stdout.flush()
            
        for i in range(n_B_steps):
            V,B = expand_B(V,B,T,O,R,V_min,gamma) 
        
        n_B = np.shape(B)[0]
        if verbose:
            print("there are now %d beliefs in B..." %n_B)  
            sys.stdout.flush()
        
        n_iter += 1    
            
    return V,B 


################ OLD FUNCS NOT USED ANYMORE (since everything is softmax now)

# def update_V(V,B,T,O,R,gamma,eps=None,max_iter=50,verbose=True,n_samps=100,
#              seed=False):
#     """
#     inputs:
#         V (list):
#             V[0]: n_B x n_S array of alpha-vector values for each belief
#             V[1]: n_B array, denoting which action generated each alpha-vector                
#         B: n_B x n_S array of belief states to be updated
        
#     optional inputs:
        
        
#     outputs:
#         V (same as input), updated
#     """
#     if seed: #testing
#         np.random.seed(711)
    
#     n_B = np.shape(B)[0]
#     n_V = np.shape(B)[0]
#     n_A = np.shape(R)[1]
#     n_S = np.shape(R)[0]
#     O_dims = np.shape(O)[1]
#     O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A
#     if eps is None:
#         eps = 0.01*n_S
    
#     #### no reason to resample O each iteration; so sample obs beforehand and cache
#     O_samps = np.random.normal(0,1,(n_samps,O_dims,n_S,n_A))
#     O_samps = O_means + O_sds*O_samps
    
#     #precompute and cache b^ao for sampled observations...
#     O_logprob = np.sum(stat.norm.logpdf(O_samps[:,:,:,:,None],
#                                         np.transpose(O_means,[0,2,1])[:,None,:,:],
#                                         np.transpose(O_sds,[0,2,1])[:,None,:,:],),1)

#     log_B = np.log(B+1e-16) # B x S
#     log_T = np.log(T+1e-16) #S' x S x A
#     log_TB =  logsumexp(log_B[None,:,:,None] + log_T[:,None,:,:],2)# S' x S

#     log_bao = np.transpose(O_logprob[:,:,None,:,:] + log_TB[None,:,:,:,None],[2,0,3,1,4])
#     b_ao =  np.exp(log_bao - logsumexp(log_bao,4)[:,:,:,:,None]) #B x K x A x S' x S

# #    O_prob = np.exp(O_logprob)
# #    b_ao = np.einsum('abcd,ef,bfc->eacbd',O_prob,B,T) #B x K x A x S' x S
# #    b_ao /= np.einsum('abcde->abcd',b_ao)[:,:,:,:,None]       
    
    
#     ### precompute funky indexing needed...
#     v_ = []
#     b_ = []
#     a_ = []
    
#     for v in range(n_V):
#         for b in range(n_B):
#             for a in range(n_A):
#                 v_.append(v)
#                 b_.append(b)
#                 a_.append(a) 
    
#     v_ = np.reshape(v_,[n_V,n_B,n_A])
#     b_ = np.reshape(b_,[n_V,n_B,n_A])
#     a_ = np.reshape(a_,[n_V,n_B,n_A])    
    
#     for ct in range(max_iter):     
        
#         old_V = np.array(V[0],copy=True)
                
#         alpha_bao = np.einsum('ab,cdefb->acdef',V[0],b_ao) #V x B x K x A x S'
#         argmax_alpha_bao = np.argmax(alpha_bao,0) #B x K x A x S'
            
#         prob_meta_obs = np.array([np.mean(argmax_alpha_bao==i,axis=1) for i in range(n_V)]) #V x B x A x S'
        
#         alpha_aO_alpha2 = np.einsum('ab,bcd,efdb->efdac',V[0],T,prob_meta_obs) #V' x B x A x V x S     
                
#         B_alpha_aO_alpha2 = np.einsum('ab,cadeb->cade',B,alpha_aO_alpha2) #V' x B x A x V
#         argmax_aB = np.argmax(B_alpha_aO_alpha2,axis=3) #V' x B x A
        
#         #tricky indexing             
#         selected_B_alpha_aO_alpha2 = alpha_aO_alpha2[v_,b_,a_,argmax_aB,:] #V' x B x A x S
           
#         alpha_ab = R.T + gamma*np.einsum('abcd->bcd',selected_B_alpha_aO_alpha2) #B x A x S 
#         alpha_ab_B = np.einsum('ab,acb->ac',B,alpha_ab) #B x A
        
#         argmax_alpha_abB = np.argmax(alpha_ab_B,axis=1) #B
#         selected_alpha_abB = alpha_ab[np.arange(n_B),argmax_alpha_abB,:] #B x S; again tricky indexing
        
#         V[0] = selected_alpha_abB
#         V[1] = argmax_alpha_abB                      
        
#         diff = np.sum(np.abs(V[0]-old_V))
            
#         #check for convergence
#         if diff < eps:
#             return V     
        
# #    if verbose:
# #        print("didn't converge during update :(" %np.sum(np.abs(V[0]-old_V)))        
#     return V 


# def run_policy(T,O,R,pi,R_sd,T_est=None,O_est=None,R_est=None,belief=None,V=None,steps=1000,
#                seed=8675309,tiger_env=False,temp=None):
#     """
#     runs a policy for a period of time, and records trajectories.
#     policy is parameterized by a value function composed of alpha vectors.
    
#     inputs:
#         V: if None, use a random policy where we just select random actions.
#             Otherwise should be value function as represented in PBVI func
        
#     outputs:
#         full trajectories 
#     """
#     rng = np.random.RandomState(seed)

#     n_A = np.shape(R)[1]
#     n_S = np.shape(R)[0]
    
#     #default values
#     if V is None:
#         action_probs = 1/n_A*np.ones(n_A) 
#     if belief is None:
#         belief = pi  
#     # if T_est is None:
#     #     T_est = T     
#     # if O_est is None:
#     #     O_est = O
#     # if R_est is None:
#     #     R_est = R
    
#     if temp is not None:
#         assert(temp>0)

    
#     log_T_est = np.log(T_est)
#     n_dim = len(O)

#     states = []
#     beliefs = []
#     actions = []
#     rewards = []
#     observations = []
        
#     #initial state
#     state = []
#     for s in range(n_S):
#         state.append(draw_discrete(pi[s],rng))
#     states.append(state)
        
#     #loop & sim trajectory
#     for t in range(steps):
#         if V is None: #random actions 
#             action = draw_discrete(action_probs,rng)
#         else: #use our learned value function from planning to greedily select optimal action            
#             if temp is None: #deterministic policy
#                 b_alpha = np.dot(V[0],belief)
#                 action = V[1][np.argmax(b_alpha)]
#             else: #stochastic policy defined by softmax over alphas w/ temperature 
#                 b_alpha = np.dot(V[0],belief)/temp               
#                 alpha_logprobs = b_alpha-logsumexp(b_alpha)
#                 alpha = draw_discrete(np.exp(alpha_logprobs))  
#                 action = V[1][alpha]
    
#         state,obs,reward = sim_step_bydim(state,action,T,O,R,rng)
#         belief = update_belief(belief,obs,action,log_T_est,O_est)
                    
#         states.append(state)
#         beliefs.append(belief)
#         actions.append(action)
#         rewards.append(reward)
#         observations.append(obs)
    
#         if tiger_env and action!=n_A-1: #if chose and didn't listen, end for tiger scenarios
#             break
#     return ( np.array(states),np.array(beliefs),np.array(actions),
#             np.array(rewards),np.array(observations) )
 










if __name__ == "__main__":    

    np.set_printoptions(threshold=1000)
    np.random.seed(111)

# from envs_cts import create_tiger_plus_environment,create_tiger_plus_witness


    n_dim=4
    sig_good = .2
    
    n_doors = 2
    n_dims_in_door = n_dim-1
    states_per_dim = 2
    (O_means,O_sds),T_true,R_true,pi_true = create_tiger_plus_environment(
            n_doors,n_dims_in_door,states_per_dim,
            R_good=1,R_bad=-5,R_listen=-.1,sig_good=sig_good,sig_bad=.1)
    O_true = (O_means,O_sds)   
    n_S_true = n_doors*states_per_dim**n_dims_in_door
    n_A = n_doors+1
    O_dims = n_dims_in_door+1 #first dim is across doors; rest within each door
    n_S = n_doors
        
    
    (O_means_wit,O_sds_wit),T_wit,R_wit,pi_wit = create_tiger_plus_witness(n_doors,n_dims_in_door,states_per_dim,
            R_good=1,R_bad=-5,R_listen=-.1,sig_good=sig_good,sig_bad=.1)    
        
    O = (O_means_wit,O_sds_wit)
    T = T_wit
    R = R_wit
    pi = pi_wit

    params = pi,T,O,R
    gamma = 0.9
    

    B = initialize_B(params,n_expandB_iters=50)
    n_B = np.shape(B)[0]
    print("starting joint learning with %d belief pts" %n_B)
    sys.stdout.flush()    
    V_min = np.min(R_true)/(1-gamma)
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]     


    V_soft = update_V_softmax(V,B,T,O,R,gamma,temp1=1,temp2=1,temp3=.1,eps=None,max_iter=500,
                         verbose=True,n_samps=100,seed=True)
    V_soft = V_soft.copy()
    
    #NOTE: appears temp3 is most important to be low to learn good policy;
    #others can be a bit higher to allow a little more gradient info thru

#    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]     
#    V = update_V(V,B,T,O,R,gamma,eps=None,max_iter=500,
#                         verbose=True,n_samps=500,seed=True)    


    plt.figure(figsize=(15,15)); 
    plt.subplot(1,3,1); 
    plt.imshow(V_soft_1[1]); 
    plt.colorbar(); 
    plt.subplot(1,3,2); 
    plt.imshow(V_soft[1]); 
    plt.colorbar();
    plt.subplot(1,3,3); 
    plt.imshow(B); 
    plt.colorbar();    

#    plt.figure(figsize=(15,15)); 
#    plt.subplot(1,2,1); 
#    plt.imshow(V_soft[0]); 
#    plt.colorbar(); 
#    plt.subplot(1,2,2); 
#    plt.imshow(V[0]); 
#    plt.colorbar();
    

    #now run our softmax policy & see how it does 
    returns = []
    for ii in range(1000):    
        #traj is just s,b,a,r,o
        traj = run_softmax_policy(T,O,R,pi,.1,T_est=None,O_est=None,R_est=None,
                                 belief=pi,V=V_soft,steps=25,
                                 seed=ii,tiger_env=True,temp=None)
        returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
    print("avg return: %.4f" %(np.mean(returns)))
    sys.stdout.flush()
    q = np.percentile(returns,[1,5,25,50,75,95,99])
    print(q)


    returns = []
    for ii in range(2500):    
        #traj is just s,b,a,r,o
        traj = run_policy(T,O,R,pi,.1,T_est=None,O_est=None,R_est=None,
                                 belief=pi,V=V,steps=50,
                                 seed=ii,tiger_env=True,temp=.1)
        returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
    print("avg return: %.4f" %(np.mean(returns)))
    sys.stdout.flush()
    q = np.percentile(returns,[1,5,25,50,75,95,99])
    print(q)




    #chainworld setup
#
#    n_S = 10
#    n_A = 3
#    O_dims = 2
#    O,T,R,pi = create_chainworld_env(
#            n_S,SIGNAL_T=.9,sig_good=.5,sig_other=.1)
#    (O_means,O_sds) = O
#    gamma = 0.9
    

    #init to equal prob & high prob of each state
#    high = .99
#    B = (1-high)/(n_S-1)*np.ones((n_S+1,n_S))
#    B[0,:] = 1/n_S
#    for i in range(n_S):
#        B[i+1,i] = high

    B = 1/n_S*np.ones((1,n_S))
    
    V = None

    n_iter = 20
    n_traj = 2000
    avg_returns = []
    return_quantiles = []
    
    #TODO: show error bars on this 
#    V_diffs = []
    n_Bs = [np.shape(B)[0]]
    
    settings_dict = {}
    V_all = []
    
    for i in range(n_iter):
        
        #TODO should really be at end of loop and run once before loop for random policy
        print("-------")
        print("PBVI has run for %d iters" %i)
        sys.stdout.flush()
        returns = []
        for ii in range(n_traj):    
            #traj is just s,b,a,r,o
            traj = run_softmax_policy(T,O,R,pi,.1,T_est=None,O_est=None,R_est=None,belief=None,V=None,steps=50,
               seed=ii,tiger_env=True,temp=.1)
            returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
        print("avg return: %.4f" %(np.mean(returns)))
        sys.stdout.flush()
        avg_returns.append(np.mean(returns))
        q = np.percentile(returns,[1,5,25,50,75,95,99])
        return_quantiles.append(q)
        print(q)
    
        #run PBVI and learn an approximately optimal value function
        V,B = pbvi(T,O,R,gamma,B=B,V=V,max_iter=1,verbose=True,n_B_steps=3) 
        
        n_Bs.append(np.shape(B)[0])
#        V_diffs.append(diff)
        V_all.append(V.copy())
        
                    
    plt.plot(avg_returns[1:]); plt.show()
    
    plt.figure(figsize=(15,50))
    plt.subplot(1,2,1)
    plt.imshow(B)
    plt.subplot(1,2,2)
    plt.imshow(V[0]); plt.show()
    
    plt.scatter(np.arange(len(V[1])),V[1]) 
    plt.grid(alpha=.3) 
    plt.yticks(np.arange(n_A))
    plt.xticks(np.arange(0,len(V[1]),5))
    plt.show()
   
    plt.figure(figsize=(15,15))
    plt.imshow(B.T)


    #check beliefs where it takes each action

    ind0 = np.where(V[1]==0)[0]
    plt.imshow(B[ind0,:])
    for i in ind0:
        print(str(np.round(B[i,:],3)) + "  " + str(V[1][i]))


    ind1 = np.where(V[1]==1)[0]
    plt.imshow(B[ind1,:])
    for i in ind1:
        print(str(np.round(B[i,:],3)) + "  " + str(V[1][i]))


    
    ind2 = np.where(V[1]==2)[0]
    plt.imshow(B[ind2,:])
    for i in ind2:
        print(str(np.round(B[i,:],3)) + "  " + str(V[1][i]))


    #see what happens if we just keep updating V for a while and not expand B...

    
    tst_returns = []
    V_diffs = []
    for i in range(1000000):
        V,diff_all = update_V(V,B,T,O,R,gamma,eps=.1,max_iter=10,verbose=False,n_samps=50)
        V_diffs.extend(diff_all)
        print(diff_all)
#        print("-------")
#        print("PBVI has run for %d iters" %i)
#        sys.stdout.flush()
        rewards = 0
        for ii in range(10000):    
            traj = run_policy(T,O,R,pi,V=V,steps=25,seed=ii,tiger_env=True)
            rewards += np.sum(traj[3]*gamma**np.arange(len(traj[3])))
        print("reward: %.2f" %rewards)
        sys.stdout.flush()
        tst_returns.append(rewards)





    #compare current V with V after a single iter
    inds = []  
    rewards0 = []
    rewards1 = []

    for ii in range(1000):        
        s0,b0,a0,r0,o0 = run_policy(T,O,R,pi,V=V,steps=25,seed=ii,tiger_env=True)
        s1,b1,a1,r1,o1 = run_policy(T,O,R,pi,V=V_now,steps=25,seed=ii,tiger_env=True)
        rewards0.append(np.sum(r0*gamma**np.arange(len(r0))))
        rewards1.append(np.sum(r1*gamma**np.arange(len(r1))))
        if np.sum(r0)>np.sum(r1):
            inds.append(ii)
    rewards0 = np.array(rewards0)
    rewards1 = np.array(rewards1)









    #plot saved stuff from V update over time

    for b in range(np.shape(B)[0]):
        V_traces = [[] for _ in range(n_S)]
        acts = []
        for i in range(len(V_all[0])):
            acts.append(V_all[1][i][b])              
            for s in range(n_S):
                V_traces[s].append(V_all[0][i][b,s])
        
        plt.figure(figsize=(8,8))
        plt.subplot(2,1,1)
        for s in range(n_S):
            plt.plot(V_traces[s],label=s)
        plt.legend()
        plt.title(str(b)+" "+str(np.round(B[b,:],3)))
        plt.subplot(2,1,2)
        plt.plot(acts)
        plt.show()
        input()
                


            
            





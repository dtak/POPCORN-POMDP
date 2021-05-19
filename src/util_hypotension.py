#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10 Mar 2019

@author: josephfutoma
"""

import autograd.numpy as np
from autograd.misc.flatten import flatten_func,flatten
from autograd.scipy.misc import logsumexp
import autograd.scipy.stats as stat

from sklearn.cluster import KMeans

from util import *



def get_padded_beh_probs_from_IDs(IDs,all_beh_probs):

    N = len(IDs)

    #first get the max sequence length so know how long to pad
    seq_lens = -np.inf*np.ones(N,dtype="int") #traj_len is max length
    for i,ID in enumerate(IDs):
        dat = all_beh_probs[ID]
        seq_lens[i] = np.shape(dat)[0] #one more obs than action or rewards 

    max_traj_len = int(np.max(seq_lens))

    n_A = np.shape(dat)[1]

    this_beh_probs = -np.inf*np.ones((N,max_traj_len,n_A))

    for i,ID in enumerate(IDs):
        dat = all_beh_probs[ID]
        seq_len = int(seq_lens[i])

        this_beh_probs[i,:seq_len,:] = dat

    return this_beh_probs


def get_padded_databatch_from_IDs(IDs,all_dat,obs_var_names,n_A):
    """
    use to create a test set that's pre-padded, and also
    call during training to create minibatches for opt
    """
    obs_varind_names = np.array([v+'_ind' for v in obs_var_names])
    behprob_var_names = np.array(['action'+str(a)+'_behprob' for a in range(n_A)])
    n_dim = len(obs_var_names) #dim of obs space

    N = len(IDs)

    #first get seqlens & max sequence length so know how long to pad
    seq_lens = np.zeros(N,dtype="int") #num_total_states, excluding initial
    for i,ID in enumerate(IDs):
        dat = all_dat[ID]
        seq_lens[i] = np.shape(dat)[0]-1 #one more obs than action (ignoring init_action) or rewards 

    max_traj_len = int(np.max(seq_lens))

    init_observs = -np.inf*np.ones((N,n_dim),dtype="float32")
    observs = -np.inf*np.ones((N,max_traj_len,n_dim),dtype="float32")
    init_observs_mask = np.zeros((N,n_dim),dtype="bool")
    observs_mask = np.zeros((N,max_traj_len,n_dim),dtype="bool")

    rewards = -np.inf*np.ones((N,max_traj_len),dtype="float32")

    init_actions = -1*np.ones(N,dtype='int32')
    actions = -1*np.ones((N,max_traj_len),dtype='int32')
    action_probs = -1.0*np.ones((N,max_traj_len),dtype="float32")   
    beh_probs = -np.inf*np.ones((N,max_traj_len,n_A))

    ### NOTE: it's some 33x slower to use the names & slice in pandas,
    # so convert df's to np & then slice direct on arrays
    dat_colnames = np.array(dat.columns)
    obsvar_inds = np.in1d(dat_colnames,obs_var_names)
    obsvarind_inds = np.in1d(dat_colnames,obs_varind_names)
    reward_ind = dat_colnames=='reward'
    action_ind = dat_colnames=='action'
    taken_action_ind = dat_colnames=='taken_action_behprob'
    behprob_vars_inds = np.in1d(dat_colnames,behprob_var_names)

    for i,ID in enumerate(IDs):
        dat = np.array(all_dat[ID])
        seq_len = seq_lens[i]

        init_observs[i,:] = dat[0,obsvar_inds]
        observs[i,:seq_len,:] = dat[1:,obsvar_inds]
        init_observs_mask[i,:] = dat[0,obsvarind_inds]
        observs_mask[i,:seq_len,:] = dat[1:,obsvarind_inds]

        rewards[i,:seq_len] = dat[:-1,reward_ind].flatten()

        init_actions[i] = dat[0,action_ind]
        actions[i,:seq_len] = dat[1:,action_ind].flatten()
        action_probs[i,:seq_len] = dat[1:,taken_action_ind].flatten()
        beh_probs[i,:seq_len,:] = dat[1:,behprob_vars_inds]

    return (init_observs,observs,init_observs_mask,observs_mask,
        rewards,init_actions,actions,action_probs,beh_probs,seq_lens)


def random_MLP_param_inits(scale,layer_sizes,rs=np.random):
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def leaky_RELU(x,a=.01):
    return np.maximum(x,a*x)

def l2_norm(MLP_params):
    flattened, _ = flatten(MLP_params)
    return np.dot(flattened,flattened)

def MLP_predict_action(MLP_params,inputs):
    for W,b in MLP_params:
        outputs = np.dot(inputs,W) + b
        inputs = leaky_RELU(outputs)
    return np.exp(outputs - logsumexp(outputs,axis=1,keepdims=True))


def get_obs_dims_rew_corr(all_dat,IDs,n_A):
    """
    helper function, to assess which dims of observation space are most correlated
    with the observed rewards. we'll use this to figure out which dims to
    up/downweight the observation variances during initial optimization.
    """

    N = len(IDs)
    #first get the max sequence length so know how long to pad
    seq_lens = -np.inf*np.ones(N,dtype="int") #traj_len is max length
    for i,ID in enumerate(IDs):
        dat = all_dat[ID]
        seq_lens[i] = int(np.shape(dat)[0]-1) #one more obs than action or rewards

    n_dim = int(np.shape(dat)[1]-3) #dim of obs space; all columns in dat except last 3 (act, act_prob, rew)

    Tt = int(np.sum(seq_lens))
    all_rewards = np.zeros(Tt,dtype="float32")
    all_obs = np.zeros((Tt,n_dim),dtype="float32")
    all_acts = np.zeros(Tt,dtype="float32")

    ct = 0
    for i,ID in enumerate(IDs):
        dat = np.array(all_dat[ID])

        seq_len = int(seq_lens[i])

        all_rewards[ct:ct+seq_len] = dat[:-1,-1]
        all_acts[ct:ct+seq_len] = dat[:-1,-3]
        all_obs[ct:ct+seq_len,:] =  dat[1:,:-3]
        ct += seq_len       
        
    # corr between dim and rewards, ignoring actions
    # corrs = []
    # for d in range(n_dim):
    #     corrs.append(np.corrcoef(all_obs[:,d],all_rewards)[0,1])

    # corr between dim and rewards, split out by action...

    all_obs_act = []
    all_rewards_act = []
    obs_rew_corrs = []
    for a in range(n_A):
        inds = all_acts==a
        all_obs_act.append(all_obs[inds,:])
        all_rewards_act.append(all_rewards[inds])
        
        this_corrs = []
        for d in range(n_dim):    
            this_corrs.append(np.cov(all_obs[inds,d],all_rewards[inds])[0,1]/
                (np.std(all_obs[inds,d])*np.std(all_rewards[inds])+1e-8))
        obs_rew_corrs.append(this_corrs)
   
    obs_rew_corrs = np.array(obs_rew_corrs) #A x D
    obsdim_maxcorrs = np.max(np.abs(obs_rew_corrs),0) 

    good_dims = obsdim_maxcorrs > .2
    bad_dims = obsdim_maxcorrs <= .2

    #regression of obs on rewards...
    # beta_hat = np.linalg.solve(np.dot(all_obs.T,all_obs),np.dot(all_obs.T,all_rewards))

    # beta_hat_abs = np.abs(beta_hat)
    # good_dims = beta_hat_abs > .1
    # bad_dims = beta_hat_abs <= .1

    return good_dims,bad_dims



def viz_actions_rewards():
    all_actions = []
    all_rewards = []
    all_maps = []

    for dat in all_dat.values():
        all_actions.append(np.array(dat.action)[1:-1])
        all_rewards.append(np.array(dat.reward)[1:-1])
        all_maps.append(np.array(dat.map)[1:-1])

    all_actions = np.concatenate(all_actions)
    all_rewards = np.concatenate(all_rewards)
    all_maps = np.concatenate(all_maps)

    means_sds = pickle.load(open('/Users/josephfutoma/Dropbox/research/mimic_data/hypotension_management/model_data/state_means_sds_ctsinds_1hr_bpleq65.p','rb'))
    maps_mean = means_sds[0][31]
    maps_sd = means_sds[1][31]
    all_maps = all_maps*maps_sd + maps_mean

    all_actions = all_actions[all_maps<55]
    all_rewards = all_rewards[all_maps<55]

    plt.scatter(all_actions+np.random.normal(0,.15,len(all_actions)),all_rewards,alpha=.05)
    plt.title('all transitions; vaso4/fluid3 (V0F0,V0F1,V0F2,V1F0,...)')
    plt.xlabel('action index')
    plt.ylabel('reward')
    # plt.savefig('/Users/josephfutoma/Dropbox/research/mimic_data/hypotension_management/data_cleaning/figs/vaso4_fluid3_rewards_actions_all.pdf')
    

    plt.figure(figsize=(20,20))
    for action in range(12):
        plt.subplot(4,3,action+1)
        plt.hist(all_rewards[all_actions==action], 1000,normed=True, cumulative=True, label='CDF',
         histtype='step', alpha=0.8)
        plt.grid(alpha=.2)
        plt.xlabel('reward')

    plt.savefig('/Users/josephfutoma/Dropbox/research/mimic_data/hypotension_management/data_cleaning/figs/vaso4_fluid3_rewards_actions_BPl65_CDFs.pdf')
    plt.show()

    plt.close('all')



def get_policy_value_given_actionprobs(action_probs,behav_action_probs,actions,
    rewards,seq_lens,gamma,plot=False,EPS=1e-2):
    """

    CWPDIS, but for baseline fixed policies (precomputed)

    """

    max_T = int(np.max(seq_lens))
    n_traj = np.shape(behav_action_probs)[0]

    CWPDIS_nums = []
    CWPDIS_denoms = []
             
    rhos = np.ones(n_traj)
    old_rhos = np.zeros(0,"float") #store imp weights at end of all trajectories
    masked_actions = np.copy(actions)

    saved_rhos = []

    for t in range(max_T):
        this_actions = actions[:,t] 
        mask = this_actions!=-1 #mask over all trajs, even those that ended
        rho_mask = masked_actions[:,t]!=-1 #variable size mask

        this_action_probs = action_probs[mask,t]

        this_action_probs = np.where(this_action_probs<EPS,EPS,this_action_probs) #fix 0 probs
        
        old_rhos = np.concatenate([old_rhos,rhos[np.logical_not(rho_mask)]])
        rhos = rhos[rho_mask]*this_action_probs/behav_action_probs[mask,t]

        saved_rhos.append(rhos)

        masked_actions = masked_actions[rho_mask,:]
        
        #cache sstats for CWPDIS


        CWPDIS_nums.append(np.sum(rhos*rewards[mask,t]))
        CWPDIS_denoms.append(np.sum(rhos)+np.sum(old_rhos))
        
    CWPDIS_obj = np.sum(np.power(gamma,np.arange(max_T))*
                        np.array(CWPDIS_nums)/np.array(CWPDIS_denoms))  

    # plt.subplot(3,1,1)
    # plt.plot(np.array(CWPDIS_nums)); plt.title('CWPDIS nums')
    # plt.subplot(3,1,2)
    # plt.plot(np.array(CWPDIS_denoms)); plt.title('CWPDIS denoms')
    # plt.subplot(3,1,3)
    # plt.plot(np.array(CWPDIS_nums)/np.array(CWPDIS_denoms)); plt.title('CWPDIS terms')

    # # plt.suptitle('random policy equal prob')
    # # plt.suptitle('no act policy')
    # plt.suptitle('random policy prop to obs prob')

    # plt.show()

    return CWPDIS_obj



def get_baseline_policy_values():
    ###TODO: add in bootstrapping on these to get intervals for lower??

    returns = []
    gam_times = gamma**np.arange(int(np.max(seq_lens_tr)))
    for i in range(N):
        l = int(seq_lens_tr[i])
        returns.append(np.sum(rewards_tr[i,:l]*gam_times[:l]))
    returns = np.array(returns)
    avg_returns = np.mean(returns)
    print("train set, empirical avg returns: %.4f" %avg_returns)

    #empiric avg reward on test set is...
    returns = []
    gam_times = gamma**np.arange(int(np.max(seq_lens_te)))
    for i in range(Nte):
        l = int(seq_lens_te[i])
        returns.append(np.sum(rewards_te[i,:l]*gam_times[:l]))
    returns = np.array(returns)
    avg_returns = np.mean(returns)
    print("test set, empirical avg returns: %.4f" %avg_returns)


    # RANDOM POLICY, EQUAL PROB ACTIONS
    action_probs = 1/n_A*np.ones(np.shape(action_probs_te))
    rand_pol_value = get_policy_value_given_actionprobs(action_probs,action_probs_te,actions_te,
        rewards_te,seq_lens_te,gamma)
    print("random policy, equal prob all actions: %.4f" %rand_pol_value)


    # NO ACTION POLICY
    action_probs = np.zeros(np.shape(action_probs_te))
    action_probs[actions_te==0] = 1
    noact_pol_value = get_policy_value_given_actionprobs(action_probs,action_probs_te,actions_te,
        rewards_te,seq_lens_te,gamma,EPS=1e-2)
    print("no action policy: %.4f" %noact_pol_value)


    # RANDOM POLICY, PROB WRT OBS
    obs_act_probs = np.zeros(n_A)
    for a in range(n_A):
        obs_act_probs[a] += np.sum(actions_tr==a)
    obs_act_probs /= np.sum(obs_act_probs)

    action_probs = -1*np.ones(np.shape(action_probs_te))
    for a in range(n_A):
        action_probs[actions_te==a] = obs_act_probs[a]

    obsrand_pol_value = get_policy_value_given_actionprobs(action_probs,action_probs_te,actions_te,
        rewards_te,seq_lens_te,gamma)
    print("random policy, prob prop to obs actions: %.4f" %obsrand_pol_value)




#####
##### funcs for on-policy evaluation
#####

def run_policy(T,O,R,pi,T_est,O_est,R_est,pi_est,gamma,N=1000,steps=100,seed=8675309,MLP_params=None,V=None,alpha_temp=.1):
    """
    runs a policy for a period of time, and records trajectories.
    policy is parameterized by a value function composed of alpha vectors.
    
    inputs:
        V: if None, use a random policy where we just select random actions.
            Otherwise should be value function as represented in PBVI func
        
    outputs:
        full trajectories 
    """

    assert(V is not None or MLP_params is not None)

    rng = np.random.RandomState(seed)
    n_A = np.shape(R)[1]
    n_S = np.shape(R)[0]
    log_T_est = np.log(T_est+1e-16)
    O_dims = np.shape(O)[1]
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A    

    all_returns = np.zeros(N)
    beliefs = np.tile(pi_est,(N,1))

    #initial states
    states = vect_draw_discrete(beliefs,rng)

    #loop & sim trajectories. 
    #TODO: extend to setting with different length trajs...
    for t in range(steps):

        if V is not None:
            b_alphas = np.dot(beliefs,V[0].T)/alpha_temp 
            exp_balpha = np.exp(b_alphas - np.max(b_alphas,1)[:,None])
            alpha_probs = exp_balpha / np.sum(exp_balpha,1)[:,None]
            action_probs = np.dot(alpha_probs,V[1])   

        if MLP_params is not None:
            action_probs = MLP_predict_action(MLP_params,beliefs)

        actions = vect_draw_discrete(action_probs,rng)
        rewards = R[states,actions]
        states = vect_draw_discrete(T[:,states,actions].T,rng)

        eps = rng.normal(0,1,(N,O_dims))
        observs = O_means[:,states,actions].T + O_sds[:,states,actions].T*eps

        all_returns += rewards*gamma**t 

        ### update all beliefs
        log_obs = np.sum(stat.norm.logpdf(
                observs.T[:,None,:], #D x 1 x N
                O_means[:,:,actions], #D x S x N
                O_sds[:,:,actions]) 
                ,0) #S' x N

        #T: S' x S x A 
        lb = np.log(beliefs+EPS) # N x S
        log_T_b = log_T_est[:,:,actions] + lb.T[None,:,:] # S' x S x N

        #assumes we filter without rewards
        log_b = log_obs + logsumexp(log_T_b,1) #S' x N
        beliefs = np.exp(log_b - logsumexp(log_b,0)).T #N x S'
              
    return np.mean(all_returns)










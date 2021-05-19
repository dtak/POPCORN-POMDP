#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:59:39 2018

@author: josephfutoma
"""

import sys

import autograd
import autograd.numpy as np
import autograd.scipy.stats as stat

from util import draw_discrete
np.set_printoptions(threshold=1000)

##########
########## Tiger setup
##########

def create_single_tiger_env(n_S,sigma,R_listen=-.1,R_good=1,R_bad=-5):
    """
    create just a single 1d tiger env.
    combine a bunch together for full multi-dim env.
    """

    n_A = n_S+1

    #setup transition model: Pr(s_t+1 | s_t, a_t)
    #just same state throughout, never change
    #same if listen; new state at random if not and then traj automatically ends
    T = np.zeros((n_S,n_S,n_A))
    for s in range(n_S):
        T[s,s,:] = 1.0   
    
    #setup obs model: Pr(o_t+1 | s_t+1, a_t)
    O_means = np.zeros((1,n_S,n_A))
    O_sds = sigma*np.ones((1,n_S,n_A))

    for s in range(n_S):
        O_means[0,s,:] = s
                
    #setup reward model: R(s_t,a_t)
    R = np.zeros((n_S,n_A))

    #listen
    R[:,n_A-1] = R_listen
    for s in range(n_S):
        for a in range(n_A-1):
            if s==a:
                R[s,a] = R_good
            else:
                R[s,a] = R_bad

                
    pi = 1/n_S*np.ones(n_S)
            
    return pi,T,(O_means,O_sds),R


def create_tiger_env(n_S,n_dim,sig_good,sig_other,R_listen=-.1,R_good=1,R_bad=-5):
    """ 
    build out full env from a tiger env in each dim
    """
    true_pis = []
    true_Ts = []
    true_Os = []
    true_Rs = []

    for d in range(n_dim):
        if d==0: #relevant signal dim
             pi,T,O,R = create_single_tiger_env(n_S,sig_good,
                R_listen=R_listen,R_good=R_good,R_bad=R_bad)
        else: #irrelevant dims
             pi,T,O,R = create_single_tiger_env(n_S,sig_other,
                R_listen=R_listen,R_good=R_good,R_bad=R_bad)
        true_pis.append(pi)
        true_Ts.append(T)
        true_Os.append(O)
        true_Rs.append(R)

    return true_pis,true_Ts,true_Os,true_Rs


def create_tiger_witness(pis,Ts,Os,Rs,sig_good,sig_other):
    """
    just pull off the exact marginals in first dim
    for everything else, do the best you can hope for 
    given misspecified model
    """
    pi = pis[0]
    T = Ts[0]
    R = Rs[0]

    n_S,n_A = np.shape(R)
    d = len(pis)

    ### only thing we need to construct is witness obs model

    O_means = np.zeros((d,n_S,n_A))
    O_sds = np.zeros((d,n_S,n_A))

    #first dimension is easy: we want to model this exactly!
    for s in range(n_S):
        O_means[0,s,:] = s
        O_sds[0,s,:] = sig_good
    
    #for every other dimension, we don't really care. so just fit a single univariate
    #normal for each dim, to what is actually a mixture over the possible different normals we might get
    #for each dimension...and for simplicity, we've assumed these excess dims all behave the same.
    
    #TODO: calculate this analytically rather than just sample...
    n_smps = 1000000
    
    smps = []
    for s in range(n_S):
        smps.extend(np.random.normal(s,sig_other,n_smps))
   
    O_means[1:,:,:] = np.mean(smps)
    O_sds[1:,:,:] = np.std(smps)

    return pi,T,(O_means,O_sds),R

##########
########## Tiger env with GMM in obs model
##########


def create_single_tiger_GMM_env(n_S=2,sigma_0=.5,sigma_1=.5,R_listen=-.1,R_good=1,R_bad=-5,z_prob=.5,pi=None):
    #rather than outputting O, instead output a function that takes in s,a and returns samples;
    #since this is building the true env that's fine, don't actually need O_means or O_sds anyways

    n_A = n_S+1

    #setup transition model: Pr(s_t+1 | s_t, a_t)
    #just same state throughout, never change
    #same if listen; new state at random if not and then traj automatically ends
    T = np.zeros((n_S,n_S,n_A))
    for s in range(n_S):
        T[s,s,:] = 1.0   
    

    #new obs model: sample from a GMM, 0.5*N(0,s0) + 0.5*N(1,s1); then threshold depending on state to <=0, >0
    def sample_O(s,a,n=1,rng=np.random):

        good_smps = []
        n_smps = len(good_smps)
        z_smps = []

        while n_smps < n:
            z = rng.uniform(0,1,n-n_smps) < z_prob 
            smps = (1-z)*rng.normal(0,sigma_0,n-n_smps) + z*rng.normal(1,sigma_1,n-n_smps)
            
            #rejection sampling
            if s==0:
                inds = smps<=0
            if s==1: 
                inds = smps>0

            good_smps.extend(smps[inds])
            z_smps.extend(z[inds])
            n_smps = len(good_smps)

        assert(len(good_smps)==n)

        return np.array(good_smps) #,np.array(z_smps)


    #setup reward model: R(s_t,a_t)
    R = np.zeros((n_S,n_A))

    #listen
    R[:,n_A-1] = R_listen
    for s in range(n_S):
        for a in range(n_A-1):
            if s==a:
                R[s,a] = R_good
            else:
                R[s,a] = R_bad

    if pi is None:          
        pi = 1/n_S*np.ones(n_S)
      
    return pi,T,sample_O,R



def create_tiger_gmm_env(n_env,n_S,sigma_0=.1,sigma_1=.1,
    R_listen=-.1,R_good=1,R_bad=-5,z_prob=.5,pi=None):
    """
    stack a bunch of gridworld envs together
    """

    true_pis = []
    true_Ts = []
    true_Os = []
    true_Rs = []

    for d in range(n_env):
        if d==0: #relevant signal dim
             pi,T,O,R = create_single_tiger_GMM_env(n_S=n_S,sigma_0=sigma_0,sigma_1=sigma_1,
                R_listen=R_listen,R_good=R_good,R_bad=R_bad,z_prob=z_prob,pi=pi)
        else: #irrelevant dims
             pi,T,O,R = create_single_tiger_GMM_env(n_S=n_S,sigma_0=sigma_0,sigma_1=sigma_1,
                R_listen=R_listen,R_good=R_good,R_bad=R_bad,z_prob=z_prob,pi=pi)
        true_pis.append(pi)
        true_Ts.append(T)
        true_Os.append(O)
        true_Rs.append(R)

    return true_pis,true_Ts,true_Os,true_Rs


def create_tiger_GMM_witness(pis,Ts,sample_Os,Rs):
    n_env = len(pis)

    pi = pis[0]
    T = Ts[0] 
    R = Rs[0]

    n_S,n_A = np.shape(R)


    O_means = np.zeros((n_env,n_S,n_A))
    O_sds = np.zeros((n_env,n_S,n_A))

    n_smps = 1000000
    
    for s in range(n_S):

        smps = sample_Os[0](s,0,n_smps)
        
        O_means[0,s,:] = np.mean(smps)
        O_sds[0,s,:] = np.std(smps)

    return pi,T,(O_means,O_sds),R


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 10/6/19

Test out the sepsis simulator...

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

from util import *
from pbvi_cts import *
from action_hmm_cts import *
from OPE_funcs import *

from sepsis_simulator.sepsisSimDiabetes.DataGenerator import DataGenerator

from sepsis_simulator.sepsisSimDiabetes.MDP import MDP
from sepsis_simulator.sepsisSimDiabetes.State import State
from sepsis_simulator.sepsisSimDiabetes.Action import Action


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

    # save_dict['est_te_policy_val'] = est_te_policy_val
    save_dict['est_te_policy_val_np'] = est_te_policy_val_np
    save_dict['te_ESS_noprune'] = te_ESS_noprune

    save_dict['avg_te_returns_det'] = avg_te_returns_det
    save_dict['quant_te_returns_det'] = quant_te_returns_det

    # save_dict['tr_ESS'] = tr_ESS
    # save_dict['tr_ESS_noprune'] = tr_ESS_noprune
    # save_dict['tr_CWPDIS_obj'] = tr_CWPDIS_obj
    save_dict['tr_CWPDIS_obj_noprune'] = tr_CWPDIS_obj_noprune


    # save_dict['tracked_params'] = tracked_params         
    # save_dict['tracked_Vs'] = tracked_Vs            
    # save_dict['tracked_Bs'] = tracked_Bs   

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

    #more complex inits that involve learning a model
    if param_init == 'EM-random':
        params,train_info = params_init_EM(init_type='random',O_var_move=False,EM_iters = 75,EM_tol = 1e-7)  #was 25, 1e-5

    return params


def params_init_random(n_A,n_S,n_dim,alpha_pi=25,alpha_T=25):
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


def params_init_EM(init_type,O_var_move=False,EM_iters = 75,EM_tol = 1e-7):
    
    if init_type=='random':
        pi,T,O,R = params_init_random(n_A,n_S,n_dim,alpha_pi=25,alpha_T=25)

    params = (pi,T,O,R)
    nat_params = to_natural_params(params)            
    lls = []
        
    print("at the start!")
    sys.stdout.flush()   
    
    last_ll = -np.inf
    for i in range(EM_iters):   
        t = time()        
        
        ll,gam,E_Njka = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,
            init_observs=init_observs_tr,init_observs_missing_mask=init_observs_mask_tr,
            observs_missing_mask=observs_mask_tr)
        pi,T,O,R = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,E_Njka,R_sd,
            init_observs=init_observs_tr,init_observs_missing_mask=init_observs_mask_tr,
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


def init_B_and_V(params):
    # V_min = np.min(rewards_tr)/(1-gamma)
    V_min = -1/(1-gamma)
    B = initialize_B(params,V_min,gamma,n_expandB_iters=min(int(n_S*4),50))

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
    'random': 250, #250, #250
    'kmeans': 25,
    'EM-random': 50, #25, #50
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

        V,B = init_B_and_V(params)
        for ii in range(n_PBVI_iters):
            V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,
                eps=.001,PBVI_temps=[.01,.01,.01],n_samps=100)

        #check value of init:
        #   - log-lik of HMM on test data 
        #   - val of learned policy

        ll = HMM_obj_fun(nat_params,observs_tr,actions_tr,observs_missing_mask=observs_mask_tr,
                init_observs=init_observs_tr,init_observs_missing_mask=init_observs_mask_tr)
        lls_tr.append(-ll)

        all_beliefs_tr = get_beliefs(params,seq_lens_tr,actions_tr,observs_tr,
            observs_missing_mask=observs_mask_tr,
            init_observs=init_observs_tr,init_observs_missing_mask=init_observs_mask_tr)
        CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
            behav_action_probs=action_probs_tr,all_beh_probs=None,actions=actions_tr,
            init_actions=None,observs=observs_tr,init_observs=init_observs_tr,
            observs_missing_mask=observs_mask_tr,init_observs_missing_mask=init_observs_mask_tr,
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

            best_te_ll = -HMM_obj_fun(nat_params,observs_te,actions_te,observs_missing_mask=observs_mask_te,
                init_observs=init_observs_te,init_observs_missing_mask=init_observs_mask_te)

            all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                observs_missing_mask=observs_mask_te,init_observs=init_observs_te,
                init_observs_missing_mask=init_observs_mask_te)
            CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
                behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                init_actions=None,observs=observs_te,init_observs=init_observs_te,
                observs_missing_mask=observs_mask_te,init_observs_missing_mask=init_observs_mask_te,
                rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                PBVI_update_iters=0,belief_with_reward=False,
                PBVI_temps=[.01,.01,.01],update_V = False,
                cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

            # Cache and reset numpy state bc internal simulator requires 
            # setting global np seed and not local rng's...dumb...
            np_state = np.random.get_state()

            np.random.seed((fold+1)*10) #fixed seed when doing eval for this fold...
            dg = DataGenerator()
            sim_rewards = dg.simulate_PBVI_policy(N, max_steps,
                V, params, PBVI_temp = None, 
                policy_idx_type='full', p_diabetes=0.2,
                output_state_idx_type='full', obs_sigmas=obs_sig, 
                meas_probs=meas_prob)
            np.random.set_state(np_state)

            gam_t = np.power(gamma,np.arange(max_steps))
            returns = np.sum(sim_rewards*gam_t,1)

            avg_returns_det = np.mean(returns)
            quant_returns_det = np.percentile(returns,[0,1,5,25,50,75,95,99,100])

            save_dict['best_init_avgreturns'] = avg_returns_det
            save_dict['best_init_quantreturns'] = quant_returns_det
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

                best_te_ll = -HMM_obj_fun(nat_params,observs_te,actions_te,observs_missing_mask=observs_mask_te,
                    init_observs=init_observs_te,init_observs_missing_mask=init_observs_mask_te)
                
                all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                    observs_missing_mask=observs_mask_te,init_observs=init_observs_te,
                    init_observs_missing_mask=init_observs_mask_te)
                CWPDIS_obj,(_,_,ESS,_,_,_,_) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
                    behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                    init_actions=None,observs=observs_te,init_observs=init_observs_te,
                    observs_missing_mask=observs_mask_te,init_observs_missing_mask=init_observs_mask_te,
                    rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                    PBVI_update_iters=0,belief_with_reward=False,
                    PBVI_temps=[.01,.01,.01],update_V = False,
                    cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

                # Cache and reset numpy state bc internal simulator requires 
                # setting global np seed and not local rng's...dumb...
                np_state = np.random.get_state()

                np.random.seed((fold+1)*10) #fixed seed when doing eval for this fold...
                dg = DataGenerator()
                sim_rewards = dg.simulate_PBVI_policy(N, max_steps,
                    V, params, PBVI_temp = None, 
                    policy_idx_type='full', p_diabetes=0.2,
                    output_state_idx_type='full', obs_sigmas=obs_sig, 
                    meas_probs=meas_prob)
                np.random.set_state(np_state)

                gam_t = np.power(gamma,np.arange(max_steps))
                returns = np.sum(sim_rewards*gam_t,1)

                avg_returns_det = np.mean(returns)
                quant_returns_det = np.percentile(returns,[0,1,5,25,50,75,95,99,100])

                save_dict['best_EMinit_avgreturns'] = avg_returns_det
                save_dict['best_EMinit_quantreturns'] = quant_returns_det
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




if __name__ == "__main__":    

    #####
    ##### from job number, figure out param settings
    #####
        
    job_id = int(sys.argv[1])
    
    cluster = sys.argv[2]
    assert cluster=='h' or cluster=='d' or cluster=='l'
    if cluster=='h': 
        DATA_PATH = '/n/scratchlfs/doshi-velez_lab/jfutoma/prediction_constrained_RL/data/sepsisSimData/sepsis_simulator_policies.p'
        OUT_PATH = '/n/scratchlfs/doshi-velez_lab/jfutoma/prediction_constrained_RL/experiments/sepsis_sim/'
    if cluster=='d':
        OUT_PATH = '/hpchome/statdept/jdf38/prediction_constrained_RL/experiments/sepsis_sim/'
    if cluster=='l':
        OUT_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/sepsis_sim/'
        DATA_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/data/sepsisSimData/sepsis_simulator_policies.p'

    RESULTS_FOLDER = 'results_6oct/'
    LOGS_FOLDER = 'logs_6oct/'
    if not os.path.exists(OUT_PATH+RESULTS_FOLDER):
       os.makedirs(OUT_PATH+RESULTS_FOLDER)
    if not os.path.exists(OUT_PATH+LOGS_FOLDER):
       os.makedirs(OUT_PATH+LOGS_FOLDER)

    # 4 x 7 x 2 x 2 x 15 x 5 = 8400

    obs_sigs = np.array([0.3]) 
    meas_probs = np.array([0.5,0.75,0.9,1.0])
    num_states = np.array([5])
    lambds = np.power(10,np.array([-2.0,-1.5,-1.0,-0.5,0,0.5,np.inf]))
    prune_nums = np.array([0]) #10
    inits = np.array(['random','EM-random']) #random
    ESS_penalties = np.array([0,10]) #0, 10
    seeds = np.arange(1,16)
    folds = np.arange(5)
    
    hyperparams_all = itertools.product(seeds,lambds,inits,obs_sigs,
        prune_nums,ESS_penalties,meas_probs,num_states,folds)
    ct = 0
    for hyperparams in hyperparams_all:
        if job_id == ct:
            (seed,lambd,param_init,obs_sig,prune_num,
                ESS_penalty,meas_prob,n_S,fold) = hyperparams
        ct += 1

    N = 5000
    env_name = 'sepsis-sim' 
    model_string = '%s_nS%d_meas-prob%.2f_obssig%.1f_N%d_lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d' %(
                env_name,n_S,meas_prob,obs_sig,N,lambd,param_init,prune_num,ESS_penalty,seed,fold)

    sys.stdout = open(OUT_PATH+LOGS_FOLDER+model_string+"_out.txt","w")
    sys.stderr = open(OUT_PATH+LOGS_FOLDER+model_string+"_err.txt","w")
    print("starting job_id %d" %job_id)
    print(model_string)
    sys.stdout.flush()

    ##### start to run!

    # DATA_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/data/sepsisSimData/sepsis_simulator_policies.p'


    np.set_printoptions(threshold=10000)                                                      
    
    save_dict = {}

    #RL eval params
    gamma = 0.99
    eval_n_traj = N #how many traj to test on & get average return of a policy
    eval_pbvi_iters = 25 #how many eval iters of PBVI to run
    max_steps = 20 #how long each trajectory should be (at most)
    
    O_dims = n_dim = 5
    n_A = 8
    obs_meas_probs = meas_prob*np.ones(n_dim)

    saved_policies = pickle.load(open(DATA_PATH, "rb"))
    optimal_policy = saved_policies['optimal_policy']
    eps_greedy_policy = saved_policies['eps_greedy_policy_0.14']


    #generate train & test data under eps greedy

    np.random.seed(fold+1) #set seed for data gen

    dg = DataGenerator()

    ### first sim data under optimal policy to get range of what is best
    (_, _, _, rewards_opt,_, _, _, _, _, _) = dg.simulate(N, max_steps,
        policy=optimal_policy, policy_idx_type='full', p_diabetes=0.2,
        output_state_idx_type='full',obs_sigmas=obs_sig)
    rewards_opt[np.isinf(rewards_opt)] = 0
    gam_t = np.power(gamma,np.arange(max_steps))
    opt_returns = np.sum(rewards_opt*gam_t,1)
    avg_opt_returns = np.mean(opt_returns)

    print('optimal policy value: %.3f' %avg_opt_returns)
    save_dict['optimal_pol_value'] = avg_opt_returns

    ### now sim train and test

    ### TODO: BUG!!! looks like we didn't actually do anything with missing data...

    np.random.seed(fold+1) #set seed for data gen

    (_, actions_tr, seq_lens_tr, rewards_tr, 
        _, init_observs_tr, observs_tr, init_observs_mask_tr, 
        observs_mask_tr, action_probs_tr) = dg.simulate(N, max_steps,
            policy=eps_greedy_policy, policy_idx_type='full', p_diabetes=0.2,
            output_state_idx_type='full',obs_sigmas=obs_sig)

    eps_greedy_rew = np.copy(rewards_tr)
    eps_greedy_rew[np.isinf(eps_greedy_rew)] = 0
    gam_t = np.power(gamma,np.arange(max_steps))
    eps_returns = np.sum(eps_greedy_rew*gam_t,1)
    avg_eps_returns = np.mean(eps_returns)

    print('eps greedy policy value: %.3f' %avg_eps_returns)
    save_dict['epsgreed_pol_value'] = avg_eps_returns

    (_, actions_te, seq_lens_te, rewards_te, 
        _, init_observs_te, observs_te, init_observs_mask_te, 
        observs_mask_te, action_probs_te) = dg.simulate(N, max_steps,
            policy=eps_greedy_policy, policy_idx_type='full', p_diabetes=0.2,
            output_state_idx_type='full',obs_sigmas=obs_sig)


    #####
    ##### Joint Learning!
    #####

    HMM_obj_fun = MAP_objective
    run_E = forward_backward_Estep
     
    np.random.seed(seed*100) #for different random restarts

    #learning setup params
    action_prob_finaltemp = action_prob_temp = .01
    PBVI_train_update_iters = 5
    PBVI_temps = [.01,.01,.01] 
    
    #reward stuff
    R_sd = R_sd_end = .01 

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

    params,V,B = get_param_inits(param_init)
    nat_params = to_natural_params(params)
    pi,T,O,R = params
    
    #do a forward backward pass to get gam which we'll need to update R
    _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,get_xi=False,
        observs_missing_mask=observs_mask_tr,init_observs=init_observs_tr,
        init_observs_missing_mask=init_observs_mask_tr)
        
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
                    actions_tr,observs_missing_mask=observs_mask_tr,
                    init_observs=init_observs_tr,
                    init_observs_missing_mask=init_observs_mask_tr)
            HMM_grad = flatten(HMM_grad)[0]
            

            #####
            ##### RL objective 
            #####

            RL_obj,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                ESS_noprune,CWPDIS_obj_noprune),RL_grad = RLobj_V_g(nat_params,R,R_sd,V,B,
                behav_action_probs=action_probs_tr,all_beh_probs=None,actions=actions_tr,
                init_actions=None,observs=observs_tr,init_observs=init_observs_tr,
                observs_missing_mask=observs_mask_tr,init_observs_missing_mask=init_observs_mask_tr,
                rewards=rewards_tr,seq_lens=seq_lens_tr,gamma=gamma,alpha_temp=.01,
                PBVI_update_iters=0,belief_with_reward=False,
                PBVI_temps=[.01,.01,.01],update_V = True,
                ESS_penalty=ESS_penalty,V_penalty=1e-5,
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
            # tr_CWPDIS_obj.append(CWPDIS_obj._value)


            #finally, apply gradient!
            flat_nat_params,last_g,step_sizes,last_steps = rprop(flat_nat_params,g,last_g,step_sizes,last_steps,v,last_v)
            last_v = v
                
            pi,T,O,R = to_params(unflatten(flat_nat_params))
            params = (pi,T,O,R)            
            nat_params = to_natural_params(params)
            
            #update R separately via incremental EM on this minibatch 
            _,gam = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd,
                get_xi=False,observs_missing_mask=observs_mask_tr,
                init_observs=init_observs_tr,init_observs_missing_mask=init_observs_mask_tr)
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
            if tot_iter % 50 == 1 or tot_iter==n_epochs:                              

                ##### check HMM performance on held-out test set
                HMM_obj,HMM_grad = HMMobj_g(nat_params,observs_te,actions_te,
                    observs_missing_mask=observs_mask_te,
                    init_observs=init_observs_te,
                    init_observs_missing_mask=init_observs_mask_te) 
                print("HMM loglik on test data %.4f, grad norm %.4f" 
                      %(-HMM_obj,np.sum(np.abs(flatten(HMM_grad)[0]))))
                
                HMM_te_objs.append(-HMM_obj)
                grad_norms_HMM_te.append(np.sum(np.abs(flatten(HMM_grad)[0])))


                ## test the OPE on test set as well...
                all_beliefs_te = get_beliefs(params,seq_lens_te,actions_te,observs_te,
                    observs_missing_mask=observs_mask_te,init_observs=init_observs_te,
                    init_observs_missing_mask=init_observs_mask_te)

                _,(V,CWPDIS_obj,ESS,CWPDIS_nums,CWPDIS_denoms,
                    ESS_noprune,CWPDIS_obj_noprune) = softmax_policy_value_objective_term(nat_params,R,R_sd,V,B,
                    behav_action_probs=action_probs_te,all_beh_probs=None,actions=actions_te,
                    init_actions=None,observs=observs_te,init_observs=init_observs_te,
                    observs_missing_mask=observs_mask_te,init_observs_missing_mask=init_observs_mask_te,
                    rewards=rewards_te,seq_lens=seq_lens_te,gamma=gamma,alpha_temp=.01,
                    PBVI_update_iters=0,belief_with_reward=False,
                    PBVI_temps=[.01,.01,.01],update_V = False,
                    cached_beliefs=all_beliefs_te,gr_safety_thresh=0,prune_num=0)

                print('iter %d, est value of policy on test data: %.5f' %(tot_iter,CWPDIS_obj_noprune))
                # est_te_policy_val.append(CWPDIS_obj)
                est_te_policy_val_np.append(CWPDIS_obj_noprune)
                # te_ESS.append(ESS)
                te_ESS_noprune.append(ESS_noprune)
                # te_CWPDIS.append((CWPDIS_nums,CWPDIS_denoms))

                # tracked_params = params
                # tracked_Vs = V


                print("testing deterministic policy via rollouts...")
                sys.stdout.flush()

                # Cache and reset numpy state bc internal simulator requires 
                # setting global np seed and not local rng's...dumb...
                np_state = np.random.get_state()

                np.random.seed((fold+1)*10) #fixed seed when doing eval for this fold...
                dg = DataGenerator()
                sim_rewards = dg.simulate_PBVI_policy(N, max_steps,
                    V, params, PBVI_temp = None, 
                    policy_idx_type='full', p_diabetes=0.2,
                    output_state_idx_type='full', obs_sigmas=obs_sig, 
                    meas_probs=meas_prob)
                np.random.set_state(np_state)

                gam_t = np.power(gamma,np.arange(max_steps))
                returns = np.sum(sim_rewards*gam_t,1)

                avg_te_returns_det.append(np.mean(returns))
                quant_te_returns_det.append(np.percentile(returns,[0,1,5,25,50,75,95,99,100]))

                print("Avg test return: %.4f" %avg_te_returns_det[-1])
                print("quantiles:")    
                print(quant_te_returns_det[-1])
                sys.stdout.flush()
                        
                ### save
                save_dict = update_and_write_savedict(save_dict) 


            #refresh belief set & refresh V every so often...
            if tot_iter == 250:    # % 5000 == 500:    
                print("getting new belief set...")
                pi,T,O,R = params

                V,B = init_B_and_V(params)

                for _ in range(25):
                    V = update_V_softmax(V,B,T,O,R,gamma,max_iter=1,verbose=False,eps=.001,
                        PBVI_temps=[.01,.01,.01])
                print("setup beliefs and V...")

                # tracked_Bs.append(B)

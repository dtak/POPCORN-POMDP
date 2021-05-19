#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 17:55:33 2018

@author: josephfutoma
"""

import os
import sys
import itertools
import pickle

# import seaborn as sns
import autograd
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad as vg
from autograd import make_vjp
from autograd.scipy.misc import logsumexp
from autograd.misc.flatten import flatten_func,flatten
import autograd.scipy.stats as stat

from scipy import stats
import matplotlib.pyplot as plt

# from pbvi_cts import update_belief,run_policy,update_V,expand_B,pbvi,sim_step
# from action_hmm_cts import (HMM_marginal_likelihood,to_natural_params,M_step_just_reward,
#     to_params,log_prior_reward,HMM_marginal_likelihood_reward,MAP_objective_reward,
#     log_prior,MAP_objective,M_step,forward_backward_Estep,forward_backward_Estep_rewards)
# from envs_cts import (create_chainworld_env,create_tiger_plus_environment,
#                       create_tiger_plus_witness,create_chainworld_witness)




def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



date='21may'
RESULTS_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger_gmm/results_'+date+'/'
FIG_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/PC-POMDP_experiment_figs/tiger_gmm/figs_'+date+'/'

if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
    os.makedirs(FIG_PATH+'obj_grad_traces/')
    os.makedirs(FIG_PATH+'param_traces/')
    os.makedirs(FIG_PATH+'results/')
    os.makedirs(FIG_PATH+'learned_param_viz/')



sigma_0s = np.array([0.1,0.3]) 
sigma_1s = np.array([0.5,1.0]) 
lambds = np.array([1e0,1e1,1e2,np.inf]) #-1
prune_nums = np.array([0,10])
inits = np.array(['random','EM-random'])
ESS_penalties = np.array([0,25])
seeds = np.arange(1,6)
folds = np.arange(5)

N = 1000
n_env = 1
lr = 1e-3
env_name = 'tigergmm' 



all_te_lls = {}
all_te_polvals = {}
all_objs = {}

all_wit_polvals = {}
all_wit_lls = {}

all_EM_polvals = {}
all_EM_lls = {}

for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:
        for prune_num in prune_nums:
            for param_init in inits:
                for ESS_penalty in ESS_penalties:
                    for lambd in lambds:

                        for fold in folds:

                            lls = []
                            polvals = []
                            objs = []
                            wit_lls = []
                            wit_polvals = []
                            EM_lls = []
                            EM_polvals = []

                            for seed in seeds:
                                model_string = RESULTS_PATH+'%s_sig0-%.1f_sig1-%.1f_N%d_lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d.p' %(
                                        env_name,sigma_0,sigma_1,N,lambd,param_init,prune_num,ESS_penalty,seed,fold)

                                try:
                                    save_dict = pickle.load(open(model_string,"rb"))       

                                    wit_lls.append(save_dict['ll_te_true'])
                                    wit_polvals.append(save_dict['PBVI_true_results'][0])

                                    lls.append(save_dict['HMM_te_objs'])
                                    polvals.append(save_dict['avg_te_returns_det'])
                                    objs.append(save_dict['objs'])

                                    if param_init == 'EM-random': #pull out EM results too
                                        EM_lls.append(save_dict['best_EMinit_te_ll'])
                                        EM_polvals.append(save_dict['best_EMinit_avgreturns'])

                                except:
                                    pass                    
                            
                            settings = (sigma_0,sigma_1,prune_num,param_init,ESS_penalty,lambd,fold)
                            all_te_lls[settings] = lls
                            all_te_polvals[settings] = polvals
                            all_wit_polvals[settings] = wit_polvals
                            all_wit_lls[settings] = wit_lls
                            all_objs[settings] = objs
                            if param_init == 'EM-random': #pull out EM results too
                                all_EM_polvals[settings] = EM_polvals
                                all_EM_lls[settings] = EM_lls

pickle.dump([all_te_lls,all_te_polvals,all_wit_polvals,all_wit_lls,all_EM_polvals,all_EM_lls,all_objs],
    open(RESULTS_PATH+'../'+date+'all_aggregate_results.p','wb'))

[all_te_lls,all_te_polvals,all_wit_polvals,all_wit_lls,
    all_EM_polvals,all_EM_lls,all_objs] = pickle.load(open(RESULTS_PATH+'../'+date+'all_aggregate_results.p','rb'))




te_lls = {}
te_polvals = {}

wit_polvals = {}
wit_lls = {}

EM_polvals = {}
EM_lls = {}

for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:
        for prune_num in prune_nums:
            for param_init in inits:
                for ESS_penalty in ESS_penalties:
                    for lambd in lambds:

                        setting = (sigma_0,sigma_1,prune_num,param_init,ESS_penalty,lambd)
                        te_lls[setting] = []
                        te_polvals[setting] = []
                        wit_polvals[setting] = []
                        wit_lls[setting] = []
                        if param_init == 'EM-random':
                            EM_polvals[setting] = []
                            EM_lls[setting] = []

                        for fold in folds:
                            run = (sigma_0,sigma_1,prune_num,param_init,ESS_penalty,lambd,fold)

                            if len(all_te_polvals[run]) > 0:

                                # a bit optimistic...
                                # best_vals = np.array([np.max(x) for x in all_te_polvals[run]])
                                # best_val_inds = np.array([np.argmax(x) for x in all_te_polvals[run]])
                                # best_seed = np.where(best_vals==np.max(best_vals))[0][0]




                                te_lls[setting].append(all_te_lls[run][best_seed][best_val_inds[best_seed]])
                                te_polvals[setting].append(all_te_polvals[run][best_seed][best_val_inds[best_seed]])

                                wit_lls[setting].append(all_wit_lls[run][best_seed])
                                wit_polvals[setting].append(all_wit_polvals[run][best_seed])

                                if param_init == 'EM-random':
                                    EM_lls[setting].append(all_EM_lls[run][best_seed])
                                    EM_polvals[setting].append(all_EM_polvals[run][best_seed])

# add in extra EM results for random init for easier plotting
for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:
        for prune_num in prune_nums:
            for param_init in inits:
                for ESS_penalty in ESS_penalties:
                    for lambd in lambds:
                        setting = (sigma_0,sigma_1,prune_num,'random',ESS_penalty,lambd)
                        setting2 = (sigma_0,sigma_1,prune_num,'EM-random',ESS_penalty,lambd)
                        EM_lls[setting] = EM_lls[setting2]
                        EM_polvals[setting] = EM_polvals[setting2]




for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:
        for prune_num in prune_nums:
            for param_init in inits:
                for ESS_penalty in ESS_penalties:


                    sigma_0 = 0.1
                    sigma_1 = 1.0
                    prune_num = 0
                    param_init = 'EM-random'
                    ESS_penalty = 0


                    this_wit_vals_means = []; this_wit_vals_ses = []
                    this_EM_vals_means = []; this_EM_vals_ses = []
                    this_lam_vals_means = []; this_lam_vals_ses = []
                    this_linf_vals_means = []; this_linf_vals_ses = []

                    this_wit_lls_means = []; this_wit_lls_ses = []
                    this_EM_lls_means = []; this_EM_lls_ses = []
                    this_lam_lls_means = []; this_lam_lls_ses = []
                    this_linf_lls_means = []; this_linf_lls_ses = []

                    v = wit_polvals[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_wit_vals_means.append(np.mean(v))
                    this_wit_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = wit_lls[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_wit_lls_means.append(np.mean(v))
                    this_wit_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                    v = EM_polvals[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_EM_vals_means.append(np.mean(v))
                    this_EM_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = EM_lls[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_EM_lls_means.append(np.mean(v))
                    this_EM_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                    v = te_polvals[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_linf_vals_means.append(np.mean(v))
                    this_linf_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = te_lls[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,np.inf)]
                    this_linf_lls_means.append(np.mean(v))
                    this_linf_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                    for lambd in lambds[:-1]: #[1.]
                        ll_means = []; ll_ses = []
                        val_means = []; val_ses = []
                        v = te_polvals[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,lambd)]
                        val_means.append(np.mean(v))
                        val_ses.append(np.std(v)/np.sqrt(len(v)))
                        v = te_lls[(sigma_0,sigma_1,prune_num,param_init,ESS_penalty,lambd)]
                        ll_means.append(np.mean(v))
                        ll_ses.append(np.std(v)/np.sqrt(len(v)))
                        this_lam_vals_means.append(val_means); this_lam_vals_ses.append(val_ses)
                        this_lam_lls_means.append(ll_means); this_lam_lls_ses.append(ll_ses)


                    #EDIT 10/3/19: get and write out the results for the final results figure we use:

                    import pandas as pd
                    pd.set_option("display.max_columns",101)

                    dat = pd.DataFrame()

                    dat['oracle_policy_value_mean'] = this_wit_vals_means
                    dat['oracle_policy_value_stderrs'] = this_wit_vals_ses
                    dat['EM_policy_value_mean'] = this_EM_vals_means
                    dat['EM_policy_value_stderrs'] = this_EM_vals_ses
                    dat['RLonly_policy_value_mean'] = this_linf_vals_means
                    dat['RLonly_policy_value_stderrs'] = this_linf_vals_ses
                    for i,lambd in enumerate(lambds[:-1]):
                        dat['lambd'+str(lambd)+'_policy_value_mean'] = this_lam_vals_means[i]
                        dat['lambd'+str(lambd)+'_policy_value_stderrs'] = this_lam_vals_ses[i]

                    dat['oracle_policy_lls_mean'] = this_wit_lls_means
                    dat['oracle_policy_lls_stderrs'] = this_wit_lls_ses
                    dat['EM_policy_lls_mean'] = this_EM_lls_means
                    dat['EM_policy_lls_stderrs'] = this_EM_lls_ses
                    dat['RLonly_policy_lls_mean'] = this_linf_lls_means
                    dat['RLonly_policy_lls_stderrs'] = this_linf_lls_ses
                    for i,lambd in enumerate(lambds[:-1]):
                        dat['lambd'+str(lambd)+'_policy_lls_mean'] = this_lam_lls_means[i]
                        dat['lambd'+str(lambd)+'_policy_lls_stderrs'] = this_lam_lls_ses[i]

                    out_path = RESULTS_PATH+'mean_stderr_results_%s_sig0-%.1f_sig1-%.1f_init-%s_prune%d_ESSpenalty%.3f.csv' %(
                                        env_name,sigma_0,sigma_1,param_init,prune_num,ESS_penalty)

                    dat.to_csv(out_path,index=False)


                    #finally plot, yay

                    # n_lam = len(lambds[:-1])

                    n_lam = 1
                    lam_colors = np.linspace(0.3,1.8,n_lam)
                    lam_colors = [lighten_color('r',c) for c in lam_colors]

                    width = 6
                    size = 250

                    sns.set(style="whitegrid", font_scale=3.5)

                    plt.close('all')
                    plt.figure(figsize=(24,11))

                    plt.scatter(x=this_wit_vals_means,y=this_wit_lls_means,s=size)
                    plt.errorbar(x=this_wit_vals_means,y=this_wit_lls_means,xerr=this_wit_vals_ses,
                        yerr=this_wit_lls_ses,label='moment-match',elinewidth=width)

                    plt.scatter(x=this_EM_vals_means,y=this_EM_lls_means,s=size)
                    plt.errorbar(x=this_EM_vals_means,y=this_EM_lls_means,xerr=this_EM_vals_ses,
                        yerr=this_EM_lls_ses,label='2 stage EM',elinewidth=width)

                    plt.scatter(x=this_linf_vals_means,y=this_linf_lls_means,s=size)
                    plt.errorbar(x=this_linf_vals_means,y=this_linf_lls_means,xerr=this_linf_vals_ses,
                        yerr=this_linf_lls_ses,label='RL loss only',elinewidth=width)

                    for i in range(n_lam):
                        plt.scatter(x=this_lam_vals_means[i],y=this_lam_lls_means[i],color=lam_colors[i],s=size)
                        plt.errorbar(x=this_lam_vals_means[i],y=this_lam_lls_means[i],xerr=this_lam_vals_ses[i],color=lam_colors[i],
                            yerr=this_lam_lls_ses[i],label='lambda %d'%(lambds[:-1][i]),elinewidth=width)

                    plt.title('Generative & Decision-Making Results')
                    plt.xlabel('Avg. Policy Value')
                    plt.ylabel('Avg. HMM log marginal likelihood')
                    plt.legend()
                    plt.grid(alpha=.2)


                    plt.show()

                    plt.savefig(FIG_PATH+'results/tigergmm_results_FIXED.pdf')

                        # BEST%s_sig0-%.1f_sig1-%.1f_N%d_init-%s_prune%d_ESSpenalty%.3f.pdf' %(
                                        # env_name,sigma_0,sigma_1,N,param_init,prune_num,ESS_penalty))















                    plt.subplot(1,2,1)

                    plt.errorbar(x,this_wit_lls_means,yerr=this_wit_lls_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x,this_wit_lls_means,s=size,label='oracle')

                    plt.errorbar(x+.05,this_EM_lls_means,yerr=this_EM_lls_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x+.05,this_EM_lls_means,s=size,label='2 stage EM') 

                    plt.errorbar(x+.1,this_linf_lls_means,yerr=this_linf_lls_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x+.1,this_linf_lls_means,s=size,label='RL loss only') 

                    offset = .15
                    for i in range(num_lams):
                        plt.errorbar(x+offset,this_lam_lls_means[i],yerr=this_lam_lls_ses[i],fmt='none',elinewidth=width,color=lam_colors[i])
                        plt.scatter(x+offset,this_lam_lls_means[i],color=lam_colors[i],s=size,label='lambda %d' %lambds[i]) 
                        offset += .05
              
                    plt.legend(loc='best') #,ncol=2)
                    plt.title("Average HMM marginal likelihood on test set")
                    plt.xlabel('Number of dims')
                    plt.ylabel('Avg. marginal likelihood per dimension')

                    plt.xticks(x)
                    plt.grid(alpha=.2)

                    ############

                    plt.subplot(1,2,2)

                    plt.errorbar(x,this_wit_vals_means,yerr=this_wit_vals_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x,this_wit_vals_means,s=size,label='oracle') 

                    plt.errorbar(x+.05,this_EM_vals_means,yerr=this_EM_vals_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x+.05,this_EM_vals_means,s=size,label='2 stage EM') 

                    plt.errorbar(x+.1,this_linf_vals_means,yerr=this_linf_vals_ses,fmt='none',elinewidth=width) 
                    plt.scatter(x+.1,this_linf_vals_means,s=size,label='RL loss only') 

                    offset = .15
                    for i in range(num_lams):
                        plt.errorbar(x+offset,this_lam_vals_means[i],yerr=this_lam_vals_ses[i],fmt='none',elinewidth=width,color=lam_colors[i])
                        plt.scatter(x+offset,this_lam_vals_means[i],color=lam_colors[i],s=size,label='lambda %d' %lambds[i]) 
                        offset += .05

                    plt.title("Average return of policy")
                    plt.xlabel('Number of dims')
                    plt.ylabel('Return')
                    plt.xticks(x)
                    plt.grid(alpha=.2)

                    plt.savefig(FIG_PATH+'results/siggood%.1f_init-%s_prune%d_ESSpenalty%.3f.pdf' %(
                                        sig_good,param_init,prune_num,ESS_penalty),
                                        bbox_inches='tight', pad_inches=0.5)              
                    
                    plt.close('all')










































#####
##### make convergence plots, tracing objectives & gradient norms per iter
#####

n_seeds = len(seeds)

for env in envs:
    for sig_good in sig_goods:
        for N in Ns:
            for lambd in lambds[2:]:               
                for n_dim in n_dims:
                    for lr in lrs: 
                        
                        res = []
                        this_seeds = []
                        for seed in seeds:
                            model_str = RESULTS_PATH+'%s_ndim%d_noise%.1f_N%d_lambd%.2f_lr%.2f_seed%d.p' %(
                                        env,n_dim,sig_good,N,lambd,lr,seed)                    
                            try:
                                this_res = pickle.load(open(model_str,"rb"))       
                                res.append(this_res)
                                this_seeds.append(seed)
                            except:
                                pass
                        
                        n_runs = len(res)
                        if n_runs==0:
                            continue
                        
                        fig = plt.figure(figsize=(75,150))
                        ii = 0
                        for save_dict,seed in zip(res,this_seeds):
                            
                            try:
                                plt.subplot(n_seeds,5,ii+1)

                                plt.plot(np.array(save_dict['RL_objs'])/-lambd,label='obj est return',alpha=.5)
                                
                                returns = save_dict['avg_returns'] 
                                true_returns = np.max(save_dict['PBVI_true_results'][0])
                                plt.plot(100*np.arange(len(returns)),returns,label='actual est return')
                                plt.hlines(true_returns,0,len(save_dict['HMM_objs']),label='witness est return')

                                if ii==0:
                                    plt.title('expected return (%d)' %seed)
                                    plt.legend()
                                else: 
                                    plt.title('(%d)' %seed)
                                      
                                plt.subplot(n_seeds,5,ii+2)
                                if ii==0:
                                    plt.title('expected return gradient norm')
                                plt.plot(np.array(save_dict['RL_grad_norm']))
                                                                         
                                plt.subplot(n_seeds,5,ii+3)
                                te_HMM_obj = save_dict['obj_full_HMM']
                                plt.plot(100*np.arange(1,len(te_HMM_obj)+1),-1*np.array(te_HMM_obj),label='learned')
                                if ii==0:   
                                    plt.title('HMM test ll; wit %.4f' %save_dict['ll_te_true']) 
                                else: 
                                    plt.title('wit %.4f' %save_dict['ll_te_true']) 

                                
                                plt.subplot(n_seeds,5,ii+4)
                                if ii==0:
                                    plt.title('HMM gradient norm')
                                g = save_dict['HMM_grad_norm']
                                plt.plot(np.arange(0,len(g),1),np.array(g[::1])) 
                                
                                plt.subplot(n_seeds,5,ii+5)
                                if ii==0:
                                    plt.title('overall objective (min)')
                                objs = save_dict['objs']
                                plt.plot(np.arange(0,len(objs),1),np.array(objs[::1])) 
                       
                            except:
                                pass
                            ii+=5
                        plt.suptitle('%s_ndim%d_noise%.1f_N%d_lambd%.2f_lr%.2f'%(env,n_dim,sig_good,N,lambd,lr))
                        plt.savefig(FIG_PATH+'obj_grad_traces/%s_ndim%d_noise%.1f_N%d_lambd%.2f_lr%.2f.pdf' %(
                                            env,n_dim,sig_good,N,lambd,lr))
                        plt.close(fig)
                        # plt.show()
                         
      

#####
##### make convergence plots, showing traces of each param over time...
#####

#for env in envs:
#    for sig_good in sig_goods:
#        for N in Ns:
#            for lambd in lambds:               
#                for n_dim in n_dims:
#                    for OPE in OPEs:

                        
                        env='tiger'
                        sig_good=0.2
                        N=10000
                        lambd=1e8
                        n_dim=1       
                        OPE='CWPDIS'
                        lr=1e-6
                        
                        res = []
                        for seed in seeds:
                            model_str = RESULTS_PATH+'%s_ndim%d_noise%.1f_N%d_lambd%.0f_%s_lr10m%.0f_seed%d.p' %(
                                        env,n_dim,sig_good,N,lambd,OPE,np.log10(lr),seed)                    
                            try:
                                this_res = pickle.load(open(model_str,"rb"))       
                                res.append(this_res)
                            except:
                                pass
                    


                        if env=='tiger':
                            n_doors = 2
                            n_dims_in_door = n_dim-1
                            states_per_dim = 2
                            (O_means_wit,O_sds_wit),T_wit,R_wit,pi_wit = create_tiger_plus_witness(n_doors,n_dims_in_door,states_per_dim,
                                R_good=1,R_bad=-5,R_listen=-.1,sig_good=sig_good,sig_bad=.1)    
                        else:
                            n_S_dim1 = 2
                            n_S_dim2p = 2
                            (O_means_wit,O_sds_wit),T_wit,R_wit,pi_wit = create_chainworld_witness(n_S_dim1,n_S_dim2p,n_dim,
                                SIGNAL_T_dim1=.9,SIGNAL_T_dim2p=.8,sig_good=sig_good,sig_other=.1)
                        O_wit = (O_means_wit,O_sds_wit)
                        n_S = np.shape(R_wit)[0]
                        n_A = np.shape(R_wit)[1]
                        
#                        res = results_dicts[(env,n_dim,sig_good,N,lambd)]
#                        n_runs = len(res)
#                        if n_runs==0:
#                            continue
    
                        for i,save_dict in enumerate(res):
                            tracked_params = save_dict['tracked_params']
                            n = len(tracked_params)
                            pis = np.zeros((n,n_S))
                            Ts = np.zeros((n,n_S,n_S,n_A))
                            O_means = np.zeros((n,n_dim,n_S,n_A))
                            O_sds = np.zeros((n,n_dim,n_S,n_A))
                            Rs = np.zeros((n,n_S,n_A))
                            
                            for ii,(pi,T,O,R) in enumerate(tracked_params):
                                pis[ii,:] = pi
                                Ts[ii,:,:,:] = T
                                O_means[ii,:,:,:] = O[0]
                                O_sds[ii,:,:,:] = O[1]
                                Rs[ii,:,:] = R
    
                            RL_obj = np.array(save_dict['RL_objs'])/-lambd
                            HMM_obj = np.array(save_dict['HMM_objs'])*-1*N/save_dict['batchsize']
    
                            #make figure                                                  
                            plt.figure(figsize=(20,35))
                            
                            #first row: b0, objectives
                            plt.subplot(5,n_S,1)
                            for s in range(n_S):
                                plt.plot(pis[:,s],label='b0_'+str(s))
                            plt.legend()
                            
                            plt.subplot(5,n_S,2)
                            plt.plot(RL_obj)
                            plt.title('RL obj / lambda')
                            
                            plt.subplot(5,n_S,3)
                            plt.plot(HMM_obj)
                            plt.title('HMM obj (scaled x N/batch))')
                                
                                
                            HMM_obj_te = np.array(save_dict['obj_full_HMM'])*-1
                            plt.subplot(5,n_S,4)
                            plt.plot(HMM_obj)
                            plt.title('HMM test obj')                                
                                
                            #plot T
                            for s in range(n_S):
                                plt.subplot(5,n_S,n_S+s+1)
                                
                                for ss in range(n_S):
                                    for a in range(n_A):
                                        plt.plot(Ts[:,s,ss,a],label='T('+str(s)+','+str(ss)+','+str(a)+')')
                                plt.legend()
                                plt.title('T('+str(s)+',:,:)')
                               
                            #plot O means
                            for s in range(n_S):
                                plt.subplot(5,n_S,2*n_S+s+1)
                                
                                for d in range(n_dim):
                                    for a in range(n_A):
                                        plt.plot(O_means[:,d,s,a],label='Omean_d'+str(d)+'_s'+str(s)+'_a'+str(a))
                                plt.legend()
                                plt.title('Omean_s'+str(s))
                                
                            #plot O sds
                            for s in range(n_S):
                                plt.subplot(5,n_S,3*n_S+s+1)
                                
                                for d in range(n_dim):
                                    for a in range(n_A):
                                        plt.plot(O_sds[:,d,s,a],label='Osd_d'+str(d)+'_s'+str(s)+'_a'+str(a))
                                plt.legend()
                                plt.title('Osd_s'+str(s))
                            
                            #plot R
                            for s in range(n_S):
                                plt.subplot(5,n_S,4*n_S+s+1)
                                
                                for a in range(n_A):
                                    plt.plot(Rs[:,s,a],label='R_s'+str(s)+'_a'+str(a))
                                plt.legend()
                                plt.title('R_s'+str(s))                                                        
    
                            plt.suptitle('%s_ndim%d_noise%.1f_N%d_lambd%.0f_seed%d'%(env,n_dim,sig_good,N,lambd,seeds[i]))
                            plt.savefig(FIG_PATH+'param_traces/%s_ndim%d_noise%.1f_N%d_lambd%.0f_%s_seed%d.pdf' %(
                                                env,n_dim,sig_good,N,lambd,OPE,seeds[i]))
                            plt.show()
                        
                        

                        
                        
##### 
##### 
##### plots of EM convergence
#####
                        
                            lls_tr_EM = save_dict['lls_tr_EM']
                            lls_val_EM = save_dict['lls_val_EM'] 
                            best_EM_init = save_dict['best_EM_init']
                            
                            plt.figure(figsize=(15,15))
                            
                            for ii in range(len(lls_tr_EM)):
                                if ii != best_EM_init:
                                    plt.plot(lls_tr_EM[ii],alpha=.5,label=ii)
                                else:
                                    plt.plot(lls_tr_EM[ii],color='blue',label=ii,lw=5)

#                                plt.hlines(lls_val_EM[i],0,150)
#                                plt.show()
                            
#                            plt.legend()
                            plt.hlines(ll_true,0,15)
                            plt.title('EM inits')
                            plt.savefig(FIG_PATH+'many_EM_inits_'+'%s_ndim%d_noise%.1f_N%d_lambd%.0f_%s_seed%d.pdf' %(
                                                env,n_dim,sig_good,N,lambd,OPE,seed))
                                    







   
#####
##### make results fig showing performance vs n_dims; random restarts
#####
    

##### OLD GMM ENV WITH 3 SIGMAS

#17 jan
# lambds = np.array([0,1e0,1e1,1e2,1e3]) #-1
# seeds = 12345*np.arange(1,6)

# sigma_0s = np.array([0.1,0.3,0.5]) 
# sigma_11s = np.array([0.1,0.3,0.5]) 
# sigma_12s = np.array([0.1,0.3,0.5]) 


import seaborn as sns


#20 jan
Ns = np.array([1000])
sigma_0s = np.array([0.2,0.5]) 
sigma_11s = np.array([0.2,0.5]) 
sigma_12s = np.array([0.2,0.5]) 
lambds = np.array([0,1e0,1e1]) #-1
n_envs = np.array([1])    
lrs = np.array([1])
seeds = 12345*np.arange(1,6)


Ns = np.array([1000]); N = 1000
n_envs = np.array([1]); n_env = 1
lrs = np.array([1]); lr = 1
O_var_move_scales = np.array([1]); O_var_move_scale = 1


n_dims = n_envs *1

env_name='tigergmm'

lambds = lambds

num_ndims = len(n_envs)
num_lams = len(lambds)

for sigma_0 in sigma_0s:
    for sigma_11 in sigma_11s:
        for sigma_12 in sigma_12s:

            mean_pbvi_bestvals = [ [] for _ in range(num_lams)]
            se_pbvi_bestvals = [ [] for _ in range(num_lams)]
            
            mean_hmm_true_lls = [ [] for _ in range(num_lams)]
            se_hmm_true_lls = [ [] for _ in range(num_lams)]
            
            mean_lls = [ [] for _ in range(num_lams)]
            se_lls = [ [] for _ in range(num_lams)]
            
            mean_vals = [ [] for _ in range(num_lams)]
            se_vals = [ [] for _ in range(num_lams)]
            
            for n_env in n_envs:
                for i,lambd in enumerate(lambds):
           
                    res = []
                    for seed in seeds: 
                        model_string = RESULTS_PATH+'%s_nenv%d_sig0-%.1f_sig11-%.1f_sig12-%.1f_N%d_lambd%.2f_lr%.2f_movescale%d_seed%d.p' %(
                                    env_name,n_env,sigma_0,sigma_11,sigma_12,N,lambd,lr,O_var_move_scale,seed)

                        try:
                            this_res = pickle.load(open(model_string,"rb"))       
                            res.append(this_res)
                        except:
                            pass                    
                                     
                    pbvi_best_vals = []
                    hmm_true_lls = []
                                            
                    lls = []
                    vals = []  
                        
                    for save_dict in res: 
                        if lambd<=0:
                            try:
                                pbvi_best_vals.append(np.max(save_dict['PBVI_true_results'][0]))
                                hmm_true_lls.append(save_dict['ll_te_true'])
                                
                                lls.append(save_dict['ll_te_final'])
                                vals.append(save_dict['PBVI_HMM_results'][1])                            
                            except:
                                continue                                    
                        else:
                            try:
                                pbvi_best_vals.append(np.max(save_dict['PBVI_true_results'][0]))
                                hmm_true_lls.append(save_dict['ll_te_true'])
                                                          
                                lls.append(-1*np.array(np.min(save_dict['obj_full_HMM']))) #best ll
                                vals.append(np.max([np.max(save_dict['avg_returns']),np.max(save_dict['avg_returns_det'])])) #best return
                            except:
                                continue
                            
                    mean_pbvi_bestvals[i].append(np.nanmean(pbvi_best_vals))
                    se_pbvi_bestvals[i].append(np.nanstd(pbvi_best_vals)/np.sqrt(len(mean_pbvi_bestvals)))
                    
                    mean_hmm_true_lls[i].append(np.nanmean(hmm_true_lls))
                    se_hmm_true_lls[i].append(np.nanstd(hmm_true_lls)/np.sqrt(len(mean_hmm_true_lls)))
                                        
                    mean_lls[i].append(np.nanmean(lls))
                    se_lls[i].append(np.nanstd(lls)/np.sqrt(len(lls)))
                    
                    mean_vals[i].append(np.nanmean(vals))
                    se_vals[i].append(np.nanstd(vals)/np.sqrt(len(vals)))
    


            #to make figure, fix env, sig, N, lambd. vary n_dim
            
            plt.figure(figsize=(60,15))

            sns.set(style="whitegrid", font_scale=3)

            # plt.suptitle(env+', noise: '+str(sig_good)+', N: '+str(N)+', lr '+str(lr), )
            x = n_dims
            
            width = 6
            size = 350

            lam_colors = np.linspace(0.3,1.8,num_lams-1)
            lam_colors = [lighten_color('r',c) for c in lam_colors]
            colors = ['blue']
            colors.extend(lam_colors)

            plt.subplot(1,2,1)

            plt.title("Average HMM marginal likelihood on test set")
            plt.xlabel('Number of dims')
            plt.ylabel('Avg. marginal likelihood per dimension')

            plt.xticks(x)
            plt.grid(alpha=.2)
            plt.errorbar(x,mean_hmm_true_lls[0],yerr=se_hmm_true_lls[0],fmt='none',color='green',elinewidth=width)
            plt.scatter(x,mean_hmm_true_lls[0],color='green',s=size)

            for i in range(num_lams):
                plt.errorbar(x+(i+1)*.1,mean_lls[i],yerr=se_lls[i],fmt='none',color=colors[i],elinewidth=width)
                plt.scatter(x+(i+1)*.1,mean_lls[i],color=colors[i],s=size)
      
            leg = ['oracle','2 stage (EM)']
            tmp = ['lambda: %.2f' %lambd for lambd in lambds[1:]]
            leg.extend(tmp)
            plt.legend(leg,loc='best',ncol=2)

            plt.subplot(1,2,2)
            plt.title("Average return of PBVI policy on test set")
            plt.xlabel('Number of dims')
            plt.ylabel('Return')
            plt.xticks(x)
            plt.grid(alpha=.2)
            plt.errorbar(x,mean_pbvi_bestvals[0],yerr=se_pbvi_bestvals[0],fmt='none',color='green',elinewidth=width)
            plt.scatter(x,mean_pbvi_bestvals[0],color='green',s=size)                             
            for i in range(num_lams):
                plt.errorbar(x+(i+1)*.1,mean_vals[i],yerr=se_vals[i],fmt='none',color=colors[i],elinewidth=width)
                plt.scatter(x+(i+1)*.1,mean_vals[i],color=colors[i],s=size)


            plt.savefig(FIG_PATH+'results/sig0-%.1f_sig11-%.1f_sig12-%.1f.pdf' %(
                                sigma_0,sigma_11,sigma_12),bbox_inches='tight', pad_inches=0.5)    

            # plt.show()
            plt.close()




#################
################ NEW GMM WITH ONLY 2 SIGMAS
################# 

import seaborn as sns

#21 jan
# sigma_0s = np.array([0.1,0.3,0.5]) 
# sigma_1s = np.array([0.1,0.3,0.5]) 
# lambds = np.array([0,1e0,1e1]) #-1
# seeds = 12345*np.arange(1,6)
# Ns = np.array([1000]); N = 1000


#3 feb
sigma_0s = np.array([0.1,0.3,0.5,1.0]) 
sigma_1s = np.array([0.1,0.3,0.5,1.0]) 
lambds = np.array([0,1e0,1e1,1e2]) #-1
seeds = 12345*np.arange(1,11)
Ns = np.array([2000]); N = 2000

n_envs = np.array([1]); n_env = 1
lrs = np.array([1]); lr = 1
O_var_move_scales = np.array([1]); O_var_move_scale = 1

n_dims = n_envs *1

env_name='tigergmm'

lambds = lambds[:2]

num_ndims = len(n_envs)
num_lams = len(lambds)

for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:

        mean_pbvi_bestvals = [ [] for _ in range(num_lams)]
        se_pbvi_bestvals = [ [] for _ in range(num_lams)]
        
        mean_hmm_true_lls = [ [] for _ in range(num_lams)]
        se_hmm_true_lls = [ [] for _ in range(num_lams)]
        
        mean_lls = [ [] for _ in range(num_lams)]
        se_lls = [ [] for _ in range(num_lams)]
        
        mean_vals = [ [] for _ in range(num_lams)]
        se_vals = [ [] for _ in range(num_lams)]
        
        for n_env in n_envs:
            for i,lambd in enumerate(lambds):
       
                res = []
                for seed in seeds: 
                    model_string = RESULTS_PATH+'%s_nenv%d_sig0-%.1f_sig1-%.1f_N%d_lambd%.2f_lr%.2f_movescale%d_seed%d.p' %(
                                env_name,n_env,sigma_0,sigma_1,N,lambd,lr,O_var_move_scale,seed)

                    try:
                        this_res = pickle.load(open(model_string,"rb"))       
                        res.append(this_res)
                    except:
                        pass                    
                                 
                pbvi_best_vals = []
                hmm_true_lls = []
                                        
                lls = []
                vals = []  
                    
                for save_dict in res: 
                    if lambd<=0:
                        try:
                            pbvi_best_vals.append(np.max(save_dict['PBVI_true_results'][0]))
                            hmm_true_lls.append(save_dict['ll_te_true'])
                            
                            lls.append(save_dict['ll_te_final'])
                            vals.append(save_dict['PBVI_HMM_results'][1])                            
                        except:
                            continue                                    
                    else:
                        try:
                            pbvi_best_vals.append(np.max(save_dict['PBVI_true_results'][0]))
                            hmm_true_lls.append(save_dict['ll_te_true'])
                                                      
                            lls.append(-1*np.array(np.min(save_dict['obj_full_HMM']))) #best ll
                            vals.append(np.max([np.max(save_dict['avg_returns']),np.max(save_dict['avg_returns_det'])])) #best return
                        except:
                            continue
                        
                mean_pbvi_bestvals[i].append(np.nanmean(pbvi_best_vals))
                se_pbvi_bestvals[i].append(np.nanstd(pbvi_best_vals)/np.sqrt(len(mean_pbvi_bestvals)))
                
                mean_hmm_true_lls[i].append(np.nanmean(hmm_true_lls))
                se_hmm_true_lls[i].append(np.nanstd(hmm_true_lls)/np.sqrt(len(mean_hmm_true_lls)))
                                    
                mean_lls[i].append(np.nanmean(lls))
                se_lls[i].append(np.nanstd(lls)/np.sqrt(len(lls)))
                
                mean_vals[i].append(np.nanmean(vals))
                se_vals[i].append(np.nanstd(vals)/np.sqrt(len(vals)))



        #to make figure, fix env, sig, N, lambd. vary n_dim
        
        plt.figure(figsize=(60,15))

        sns.set(style="whitegrid", font_scale=3)

        # plt.suptitle(env+', noise: '+str(sig_good)+', N: '+str(N)+', lr '+str(lr), )
        x = n_dims
        
        width = 6
        size = 350

        lam_colors = np.linspace(0.3,1.8,num_lams-1)
        lam_colors = [lighten_color('r',c) for c in lam_colors]
        colors = ['blue']
        colors.extend(lam_colors)

        plt.subplot(1,2,1)

        plt.title("Average HMM marginal likelihood on test set")
        plt.xlabel('Number of dims')
        plt.ylabel('Avg. marginal likelihood per dimension')

        plt.xticks(x)
        plt.grid(alpha=.2)
        plt.errorbar(x,mean_hmm_true_lls[0],yerr=se_hmm_true_lls[0],fmt='none',color='green',elinewidth=width)
        plt.scatter(x,mean_hmm_true_lls[0],color='green',s=size)

        for i in range(num_lams):
            plt.errorbar(x+(i+1)*.1,mean_lls[i],yerr=se_lls[i],fmt='none',color=colors[i],elinewidth=width)
            plt.scatter(x+(i+1)*.1,mean_lls[i],color=colors[i],s=size)
  
        leg = ['oracle','2 stage (EM)']
        tmp = ['lambda: %.2f' %lambd for lambd in lambds[1:]]
        leg.extend(tmp)
        plt.legend(leg,loc='best',ncol=2)

        plt.subplot(1,2,2)
        plt.title("Average return of PBVI policy on test set")
        plt.xlabel('Number of dims')
        plt.ylabel('Return')
        plt.xticks(x)
        plt.grid(alpha=.2)
        plt.errorbar(x,mean_pbvi_bestvals[0],yerr=se_pbvi_bestvals[0],fmt='none',color='green',elinewidth=width)
        plt.scatter(x,mean_pbvi_bestvals[0],color='green',s=size)                             
        for i in range(num_lams):
            plt.errorbar(x+(i+1)*.1,mean_vals[i],yerr=se_vals[i],fmt='none',color=colors[i],elinewidth=width)
            plt.scatter(x+(i+1)*.1,mean_vals[i],color=colors[i],s=size)


        plt.savefig(FIG_PATH+'results/sig0-%.1f_sig1-%.1f.pdf' %(
                            sigma_0,sigma_1),bbox_inches='tight', pad_inches=0.5)    

        # plt.show()
        plt.close()





####################

#################### VISUALIZE LEARNED PARAMS FROM NEW RUNS (IE 2 SIGMAS; truncated GMM)

from envs_cts import (create_single_tiger_GMM_env,create_tiger_GMM_witness,create_tiger_gmm_env)

# sigma_0s = np.array([0.1,0.3,0.5]) 
# sigma_1s = np.array([0.1,0.3,0.5]) 
# lambds = np.array([0,1e0,1e1]) #-1
# Ns = np.array([1000]); N = 1000
# seeds = 12345*np.arange(1,6)


#3 feb
sigma_0s = np.array([0.1,0.3,0.5,1.0]) 
sigma_1s = np.array([0.1,0.3,0.5,1.0]) 
lambds = np.array([0,1e0,1e1,1e2]) #-1
seeds = 12345*np.arange(1,11)
Ns = np.array([2000]); N = 2000


n_envs = np.array([1]); n_env = 1
lrs = np.array([1]); lr = 1
O_var_move_scales = np.array([1]); O_var_move_scale = 1

n_dims = n_envs *1

env_name='tigergmm'

n_seeds = len(seeds)

n_A = 3
n_dim=1
n_S = 2


sns.set(font_scale=.65)

for sigma_0 in sigma_0s:
    for sigma_1 in sigma_1s:

        res = []

        for lambd in lambds:

            for seed in seeds: 
                model_string = RESULTS_PATH+'%s_nenv%d_sig0-%.1f_sig1-%.1f_N%d_lambd%.2f_lr%.2f_movescale%d_seed%d.p' %(
                            env_name,n_env,sigma_0,sigma_1,N,lambd,lr,O_var_move_scale,seed)

                try:
                    this_res = pickle.load(open(model_string,"rb"))       
                    res.append(this_res)
                except:
                    pass             


        z_prob = .5 # prob of drawing from rightmost Gaussian (vs the one at 0)
        #get the overall observations to look like 2 gaussians
        pz0 = stats.norm.cdf(0,loc=0,scale=sigma_0) / (stats.norm.cdf(0,loc=0,scale=sigma_0) + stats.norm.cdf(0,loc=1,scale=sigma_1))
        pz1 = (1-stats.norm.cdf(0,loc=0,scale=sigma_0)) / (1-stats.norm.cdf(0,loc=0,scale=sigma_0) + 1-stats.norm.cdf(0,loc=1,scale=sigma_1))
        s_prob = pz0 / (pz0+pz1) #prob of being in state 1 (vs state 0)
        true_pis,true_Ts,true_Os,true_Rs = create_tiger_gmm_env(n_env,n_S,sigma_0=sigma_0,sigma_1=sigma_1,
                R_listen=-.1,R_good=1,R_bad=-5,z_prob=z_prob,pi=np.array([1-s_prob,s_prob]))

        n_smps = 200000
        smps_0 = true_Os[0](0,0,int(n_smps*true_pis[0][0]))
        smps_1 = true_Os[0](1,0,int(n_smps*true_pis[0][1]))

        plt.figure(figsize=(15,15))

        #witness numbers may be slightly different each seed
        for i,save_dict in enumerate(res[:n_seeds]):

            params_wit = to_params(this_res['nat_params_wit'])
            pi,T,O,R = params_wit
            O_means = O[0]; O_sds = O[1]
            ll = this_res['ll_te_true']
            val = np.max(this_res['PBVI_true_results'][0])

            plt.subplot(5,10,i+1)
            plt.hist(smps_0,100,alpha=.3,label='_nolegend_'); plt.hist(smps_1,100,alpha=.3,label='_nolegend_'); 
            smps_wit_0 = np.random.normal(O_means[0,0,0],O_sds[0,0,0],int(n_smps*pi[0]))
            smps_wit_1 = np.random.normal(O_means[0,1,0],O_sds[0,1,0],int(n_smps*pi[1]))
            plt.hist(smps_wit_0,100,alpha=.3,label='%.2f,%.2f' %(O_means[0,0,0],O_sds[0,0,0])); 
            plt.hist(smps_wit_1,100,alpha=.3,label='%.2f,%.2f' %(O_means[0,1,0],O_sds[0,1,0])); 
            plt.title('wit, ll %.3f, pol %.3f' %(ll,val)); 
            plt.legend(fontsize='x-small')
            plt.xticks([]); plt.yticks([])

        for i,save_dict in enumerate(res):
            if i < n_seeds:
                params = save_dict['best_EM_params']
                ll = save_dict['ll_te_final']
                val = save_dict['PBVI_HMM_results'][1]
            else:
                try:
                    ll = -1*np.array(np.min(save_dict['obj_full_HMM']))
                    vals = np.maximum(save_dict['avg_returns'],save_dict['avg_returns_det'])
                    ind = np.argmax(vals)
                    val = vals[ind]

                    params = save_dict['tracked_params'][ind]
                except:
                    continue
            pi,T,O,R = params
            O_means = O[0]; O_sds = O[1]
            lam = lambds[i//n_seeds]
            if lam == 0:
                txt = 'EM, '
            else:
                txt = 'l'+str(int(lam))+', '

            plt.subplot(5,10,i+1+n_seeds)
            plt.hist(smps_0,100,alpha=.3,label='_nolegend_'); plt.hist(smps_1,100,alpha=.3,label='_nolegend_'); 
            smps_wit_0 = np.random.normal(O_means[0,0,2],O_sds[0,0,2],int(n_smps*pi[0]))
            smps_wit_1 = np.random.normal(O_means[0,1,2],O_sds[0,1,2],int(n_smps*pi[0]))
            plt.hist(smps_wit_0,100,alpha=.3,label='%.2f,%.2f' %(O_means[0,0,2],O_sds[0,0,2])); 
            plt.hist(smps_wit_1,100,alpha=.3,label='%.2f,%.2f' %(O_means[0,1,2],O_sds[0,1,2])); 
            plt.title(txt+'ll %.3f, pol %.3f' %(ll,val)); 
            plt.legend(fontsize='x-small')
            plt.xticks([]); plt.yticks([])

        plt.suptitle('sig0 %.1f sig1 %.1f' %(sigma_0,sigma_1)); 

        # plt.show()

        plt.savefig(FIG_PATH+'learned_param_viz/sig0-%.1f_sig1-%.1f.pdf' %(
                    sigma_0,sigma_1))

        plt.close()











     
                                                      
                 


    #plug in params of interest to see how well the policy is...

    sigma_0 = 0.3
    sigma_1 = 0.1


    gamma = .9
    n_S = 2 #by design
    R_sd_end = .1
    tiger_env = 'gmm'
    belief_with_reward = False

    O_dims = n_dim
    true_pis,true_Ts,true_Os,true_Rs = create_tiger_gmm_env(n_env,n_S,sigma_0=sigma_0,sigma_1=sigma_1,
            R_listen=-.1,R_good=1,R_bad=-5)
    n_A = np.shape(true_Rs[0])[1]

    pi_wit,T_wit,O_wit,R_wit = create_tiger_GMM_witness(true_pis,true_Ts,true_Os,true_Rs)
    O_means_wit = O_wit[0]
    O_sds_wit = O_wit[1]
    params_wit = (pi_wit,T_wit,O_wit,R_wit)

    pi = pi_wit
    T = T_wit
    O = O_wit
    R = R_wit

    # #modify O
    # O[0][0,0,2] = -0.11
    # O[0][0,1,2] = 0.68

    # O[1][0,0,2] = .06
    # O[1][0,1,2] = .50


    #try wonky EM 
    lambd = 0
    seed = seeds[0]
    model_string = RESULTS_PATH+'%s_nenv%d_sig0-%.1f_sig1-%.1f_N%d_lambd%.2f_lr%.2f_movescale%d_seed%d.p' %(
                            env_name,n_env,sigma_0,sigma_1,N,lambd,lr,O_var_move_scale,seed)
    save_dict = pickle.load(open(model_string,"rb"))  
    params = save_dict['best_EM_params']   
    pi,T,O,R = params 


    ### get value of policy from this model
    b = np.linspace(.01,.99,35)
    B = np.array([b,1-b]).T
    n_B = B.shape[0]
    V_min = np.min(R_wit)/(1-gamma)
    V = [V_min*np.ones((n_B,n_S)),-1*np.ones(n_B)]

    for _ in range(4):
        V = update_V_softmax(V,B,T,O,R,gamma,max_iter=50,verbose=False,eps=.001)



    eval_n_traj = 10000
    eval_n_steps = 20 #how long each eval trajectory should be (at most)

    print("-------")
    print("testing policy...")
    sys.stdout.flush()
    HMM_returns = []
    for ii in range(eval_n_traj):    
        #traj is just s,b,a,r,o
        traj = run_softmax_policy(true_Ts,true_Os,true_Rs,true_pis,R_sd_end,V=V,steps=eval_n_steps,
                          seed=ii,tiger_env=tiger_env,T_est=T,
                          O_est=O,belief=pi,R_est=R,temp=None,
                          belief_with_reward=belief_with_reward)
        HMM_returns.append(np.sum(traj[3]*gamma**np.arange(len(traj[3]))))
    HMM_avg_returns = np.mean(HMM_returns)
    HMM_return_quantiles = np.percentile(HMM_returns,[0,1,5,25,50,75,95,99,100])

    print("---------------------")
    print("Avg return: %.4f" %HMM_avg_returns)
    print("quantiles:")    
    print(HMM_return_quantiles)
    print("---------------------")
    sys.stdout.flush()



    max_traj_len = 250 #conservative upper bound on traj_lens; in practice should be much shorter
    min_traj_len = 20
    final_listen_prob = 0.5
    observs_tr,rewards_tr,actions_tr,_,seq_lens_tr = simdata_random_policy(
            N,true_pis,true_Ts,true_Os,true_Rs,min_traj_len,max_traj_len,final_listen_prob)
    HMM_obj_fun = MAP_objective_reward if belief_with_reward else MAP_objective
    run_E = forward_backward_Estep_rewards if belief_with_reward else forward_backward_Estep


    nat_params = to_natural_params(params)
    print("ll:")
    ll_true = -HMM_obj_fun(nat_params,observs_tr,actions_tr,rewards_tr,R_sd_end)
    print(ll_true)


    nat_params_wit = to_natural_params(params_wit)
    print("ll wit:")
    ll_true = -HMM_obj_fun(nat_params_wit,observs_tr,actions_tr,rewards_tr,R_sd_end)
    print(ll_true)


    #try wonky EM 
    lambd = 0
    seed = seeds[0]
    model_string = RESULTS_PATH+'%s_nenv%d_sig0-%.1f_sig1-%.1f_N%d_lambd%.2f_lr%.2f_movescale%d_seed%d.p' %(
                            env_name,n_env,sigma_0,sigma_1,N,lambd,lr,O_var_move_scale,seed)
    save_dict = pickle.load(open(model_string,"rb"))  
    params = save_dict['best_EM_params']   
    pi,T,O,R = params
 

    O[0][0,0,:] = 0
    O[0][0,1,:] = 1
    O[1][0,0,:] = sigma_0
    O[1][0,1,:] = sigma_1

    T[:] = .5
    T = np.array(T_wit)

    params = pi,T,O,R
    nat_params = to_natural_params((pi,T,O,R))    

    for i in range(100):           
        R_sd = R_sd_end + 10*np.exp(-i/5)  #anneal variance of R over time - doesn't really matter unless filtering w rewards
        
        gam,xi = run_E(nat_params,observs_tr,actions_tr,rewards_tr,R_sd)
        _,T,_,_ = M_step(nat_params,observs_tr,actions_tr,rewards_tr,gam,xi,R_sd)

        params = (pi,T,O,R)
        nat_params = to_natural_params((pi,T,O,R))    
        
        ll = -HMM_obj_fun(nat_params,observs_tr,actions_tr,rewards_tr,R_sd)
        lls.append(ll)  
        print(i,ll)   
        sys.stdout.flush()    
        






        # #make figure                                                  
        # plt.figure(figsize=(80,30))
        # n_row = 1+len(res)
        
        # ii = 0
        
        # plt.subplot(n_row,1+n_A+n_dim*2+1,ii+1)
        # plt.imshow(pi[None,:]); plt.colorbar(); plt.ylabel('witness'); plt.clim(0,1)
        # plt.title('b_0 = p(s_0)')
                    
        # for a in range(n_A):
        #     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+a)
        #     plt.imshow(T[:,:,a]); plt.colorbar(); plt.ylabel('s\''); plt.clim(0,1)
        #     plt.title('p(s\' | s,a=%d)'%a)

        # for d in range(n_dim):
        #     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+d)
        #     plt.imshow(O[0][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(-1,2)
        #     plt.title('O_mean: dim %d (s,a)'%d)  
            
        # for d in range(n_dim):
        #     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim+d)
        #     plt.imshow(O[1][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(0,1)
        #     plt.title('O_sd: dim %d (s,a)'%d)     
            
        # plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim*2)
        # plt.imshow(R); plt.colorbar(); plt.ylabel('s'); plt.clim(-6,2)
        # plt.title('R(s,a)')
        
        # ii += 2+n_A+n_dim*2
        
        # for i,save_dict in enumerate(res):
        #     if i <5:
        #         params = save_dict['best_EM_params']
        #     else:
        #         try:
        #             params = save_dict['params']
        #         except:
        #             continue
        #     pi,T,O,R = params
            

        #     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+1)
        #     plt.imshow(pi[None,:]); plt.colorbar(); 
        #     if i==1 or i==6 or i==11:
        #         plt.ylabel('lambda %.1f' %(lambds[i//5]))
        #     plt.clim(0,1)
                        
        #     for a in range(n_A):
        #         plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+a)
        #         plt.imshow(T[:,:,a]); plt.colorbar(); plt.ylabel('s\''); plt.clim(0,1)

        #     for d in range(n_dim):
        #         plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+d)
        #         plt.imshow(O[0][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(-1,2)
                
        #     for d in range(n_dim):
        #         plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim+d)
        #         plt.imshow(O[1][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(0,1)
                
        #     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim*2)
        #     plt.imshow(R); plt.colorbar(); plt.ylabel('s'); plt.clim(-6,2)

        #     ii += 2+n_A+n_dim*2

        # plt.suptitle('sig0 %.1f, sig1 %.1f'%(sigma_0,sigma_1))
        # plt.show()

        # plt.savefig(FIG_PATH+'learned_param_viz/sig0%.1f_sig1%.1f.pdf' %(
        #                     sigma_0,sigma_1))










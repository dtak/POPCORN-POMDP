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

import autograd
import autograd.numpy as np
from autograd import grad
from autograd import value_and_grad as vg
from autograd import make_vjp
from autograd.scipy.misc import logsumexp
from autograd.misc.flatten import flatten_func,flatten
import autograd.scipy.stats as stat

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



date='3oct'
RESULTS_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger/results_'+date+'/'
FIG_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/PC-POMDP_experiment_figs/tiger/figs_'+date+'/'

if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
    os.makedirs(FIG_PATH+'obj_grad_traces/')
    os.makedirs(FIG_PATH+'param_traces/')
    os.makedirs(FIG_PATH+'results/')
    os.makedirs(FIG_PATH+'learned_param_viz/')




import seaborn as sns

#3oct for aistats
sig_goods = np.array([0.3]) #,0.5]) 
sig_others = np.array([0.1]) 
lambds = np.array([1e0,1e1,1e2,np.inf]) 
n_envs = np.array([1,2,4,8,16])    
prune_nums = np.array([0]) #10
inits = np.array(['EM-random']) #random
ESS_penalties = np.array([0]) #0, 10
seeds = np.arange(1,26)
folds = np.arange(5)


N = 1000
sig_other = 0.1
env_name = 'tiger' 
lr = 1e-3


all_te_lls = {}
all_te_polvals = {}

all_wit_polvals = {}
all_wit_lls = {}

all_EM_polvals = {}
all_EM_lls = {}

for sig_good in sig_goods:
    for prune_num in prune_nums:
        for param_init in inits:
            for ESS_penalty in ESS_penalties:
                for n_env in n_envs:
                    for lambd in lambds:
                        for fold in folds:

                            lls = []
                            polvals = []
                            wit_lls = []
                            wit_polvals = []
                            EM_lls = []
                            EM_polvals = []

                            for seed in seeds:
                                model_string = RESULTS_PATH+'%s_nenv%d_siggood%.1f_sigother%.1f_N%d_lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d.p' %(
                                            env_name,n_env,sig_good,sig_other,N,lambd,param_init,prune_num,ESS_penalty,seed,fold)

                                try:
                                    save_dict = pickle.load(open(model_string,"rb"))       

                                    wit_lls.append(save_dict['ll_te_true'])
                                    wit_polvals.append(save_dict['PBVI_true_results'][0])

                                    lls.append(save_dict['HMM_te_objs'])
                                    polvals.append(save_dict['avg_te_returns_det'])

                                    if param_init == 'EM-random': #pull out EM results too
                                        EM_lls.append(save_dict['best_EMinit_te_ll'])
                                        EM_polvals.append(save_dict['best_EMinit_avgreturns'])

                                except:
                                    pass                    
                            
                            settings = (sig_good,prune_num,param_init,ESS_penalty,n_env,lambd,fold)
                            all_te_lls[settings] = lls
                            all_te_polvals[settings] = polvals
                            all_wit_polvals[settings] = wit_polvals
                            all_wit_lls[settings] = wit_lls
                            if param_init == 'EM-random': #pull out EM results too
                                all_EM_polvals[settings] = EM_polvals
                                all_EM_lls[settings] = EM_lls

pickle.dump([all_te_lls,all_te_polvals,all_wit_polvals,all_wit_lls,all_EM_polvals,all_EM_lls],
    open(RESULTS_PATH+'all_aggregate_results.p','wb'))


te_lls = {}
te_polvals = {}

wit_polvals = {}
wit_lls = {}

EM_polvals = {}
EM_lls = {}

for sig_good in sig_goods:
    for prune_num in prune_nums:
        for param_init in inits:
            for ESS_penalty in ESS_penalties:
                for n_env in n_envs:
                    for lambd in lambds:

                        setting = (sig_good,prune_num,param_init,ESS_penalty,n_env,lambd)
                        te_lls[setting] = []
                        te_polvals[setting] = []
                        wit_polvals[setting] = []
                        wit_lls[setting] = []
                        if param_init == 'EM-random':
                            EM_polvals[setting] = []
                            EM_lls[setting] = []

                        for fold in folds:
                            run = (sig_good,prune_num,param_init,ESS_penalty,n_env,lambd,fold)

                            if len(all_te_polvals[run]) > 0:

                                # a bit optimistic...
                                best_vals = np.array([np.max(x) for x in all_te_polvals[run]])
                                best_val_inds = np.array([np.argmax(x) for x in all_te_polvals[run]])
                                best_seed = np.where(best_vals==np.max(best_vals))[0][0]

                                te_lls[setting].append(all_te_lls[run][best_seed][best_val_inds[best_seed]])
                                te_polvals[setting].append(all_te_polvals[run][best_seed][best_val_inds[best_seed]])

                                wit_lls[setting].append(all_wit_lls[run][best_seed])
                                wit_polvals[setting].append(all_wit_polvals[run][best_seed])

                                if param_init == 'EM-random':
                                    EM_lls[setting].append(all_EM_lls[run][best_seed])
                                    EM_polvals[setting].append(all_EM_polvals[run][best_seed])

# add in extra EM results for random init for easier plotting
for sig_good in sig_goods:
    for prune_num in prune_nums:
        for ESS_penalty in ESS_penalties:
            for n_env in n_envs:
                for lambd in lambds:
                    setting = (sig_good,prune_num,'random',ESS_penalty,n_env,lambd)
                    setting2 = (sig_good,prune_num,'EM-random',ESS_penalty,n_env,lambd)
                    EM_lls[setting] = EM_lls[setting2]
                    EM_polvals[setting] = EM_polvals[setting2]




for sig_good in sig_goods:
    for prune_num in prune_nums:
        for param_init in inits:
            for ESS_penalty in ESS_penalties:


                sig_good = 0.3
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

                for n_env in n_envs:
                    v = wit_polvals[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_wit_vals_means.append(np.mean(v))
                    this_wit_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = wit_lls[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_wit_lls_means.append(np.mean(v))
                    this_wit_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                    v = EM_polvals[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_EM_vals_means.append(np.mean(v))
                    this_EM_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = EM_lls[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_EM_lls_means.append(np.mean(v))
                    this_EM_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                    v = te_polvals[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_linf_vals_means.append(np.mean(v))
                    this_linf_vals_ses.append(np.std(v)/np.sqrt(len(v)))
                    v = te_lls[(sig_good,prune_num,param_init,ESS_penalty,n_env,np.inf)]
                    this_linf_lls_means.append(np.mean(v))
                    this_linf_lls_ses.append(np.std(v)/np.sqrt(len(v)))

                for lambd in lambds[:-1]: #[1.]
                    ll_means = []; ll_ses = []
                    val_means = []; val_ses = []
                    for n_env in n_envs:
                        v = te_polvals[(sig_good,prune_num,param_init,ESS_penalty,n_env,lambd)]
                        val_means.append(np.mean(v))
                        val_ses.append(np.std(v)/np.sqrt(len(v)))
                        v = te_lls[(sig_good,prune_num,param_init,ESS_penalty,n_env,lambd)]
                        ll_means.append(np.mean(v))
                        ll_ses.append(np.std(v)/np.sqrt(len(v)))
                    this_lam_vals_means.append(val_means); this_lam_vals_ses.append(val_ses)
                    this_lam_lls_means.append(ll_means); this_lam_lls_ses.append(ll_ses)


                #EDIT 10/3/19: get and write out the results for the final results figure we use:

                import pandas as pd
                pd.set_option("display.max_columns",101)
                dat = pd.DataFrame()

                dat['obs_dim'] = n_envs

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

                out_path = RESULTS_PATH+'mean_stderr_results_%s_siggood-%.1f_init-%s_prune%d_ESSpenalty%.3f.csv' %(
                                    env_name,sig_good,param_init,prune_num,ESS_penalty)

                dat.to_csv(out_path,index=False)


                #finally plot, yay

                num_lams = 1
                x = n_envs
                width = 6
                size = 350
                lam_colors = np.linspace(0.3,1.8,3)
                lam_colors = [lighten_color('r',c) for c in lam_colors]


                plt.close('all')
                plt.figure(figsize=(60,15))
                sns.set(style="whitegrid", font_scale=1.5)

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

                plt.show()

                # plt.savefig(FIG_PATH+'results/siggood%.1f_init-%s_prune%d_ESSpenalty%.3f.pdf' %(
                #                     sig_good,param_init,prune_num,ESS_penalty),
                #                     bbox_inches='tight', pad_inches=0.5)              
                
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
                                    











##########
########## visualize learned params from EM  / our method
##########



# for env in envs:
#     for sig_good in sig_goods:
#         for N in Ns:
#             for lambd in lambds[:1]:               
#                 for n_dim in n_dims:
#                     for OPE in OPEs:
#                         for lr in lrs:
                        
# #                            env='tiger'
# #                            sig_good=0.2
# #                            N=10000
# #                            lambd=0
# #                            n_dim=1       
# #                            OPE='CWPDIS'
# #                            lr = 1e-5
                            
#                             res = []
#                             for seed in seeds:
#                                 model_str = RESULTS_PATH+'%s_ndim%d_noise%.1f_N%d_lambd%.0f_%s_lr10m%.0f_seed%d.p' %(
#                                             env,n_dim,sig_good,N,lambd,OPE,np.log10(lr),seed)                    
#                                 try:
#                                     this_res = pickle.load(open(model_str,"rb"))       
#                                     res.append(this_res)
#                                 except:
#                                     pass
                               
    
#                             if env=='tiger':
#                                 n_doors = 2
#                                 n_dims_in_door = n_dim-1
#                                 states_per_dim = 2
#                                 (O_means_wit,O_sds_wit),T_wit,R_wit,pi_wit = create_tiger_plus_witness(n_doors,n_dims_in_door,states_per_dim,
#                                     R_good=1,R_bad=-5,R_listen=-.1,sig_good=sig_good,sig_bad=.1)    
#                             else:
#                                 n_S_dim1 = 2
#                                 n_S_dim2p = 2
#                                 (O_means_wit,O_sds_wit),T_wit,R_wit,pi_wit = create_chainworld_witness(n_S_dim1,n_S_dim2p,n_dim,
#                                     SIGNAL_T_dim1=.9,SIGNAL_T_dim2p=.8,sig_good=sig_good,sig_other=.1)
#                             O_wit = (O_means_wit,O_sds_wit)
#                             n_S = np.shape(R_wit)[0]
#                             n_A = np.shape(R_wit)[1]

                            
                            
#                             pi,T,O,R = pi_wit,T_wit,O_wit,R_wit

#                             #make figure                                                  
#                             plt.figure(figsize=(35+n_dim*10,35))
#                             n_row = 1+len(res)
                            
#                             ii = 0
                            
#                             plt.subplot(n_row,1+n_A+n_dim*2+1,ii+1)
#                             plt.imshow(pi_wit[None,:]); plt.colorbar(); plt.ylabel('witness'); plt.clim(0,1)
#                             plt.title('b_0 = p(s_0)')
                                        
#                             for a in range(n_A):
#                                 plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+a)
#                                 plt.imshow(T[:,:,a]); plt.colorbar(); plt.ylabel('s\''); plt.clim(0,1)
#                                 plt.title('p(s\' | s,a=%d)'%a)

#                             for d in range(n_dim):
#                                 plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+d)
#                                 plt.imshow(O[0][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(-1,2)
#                                 plt.title('O_mean: dim %d (s,a)'%d)  
                                
#                             for d in range(n_dim):
#                                 plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim+d)
#                                 plt.imshow(O[1][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(0,1)
#                                 plt.title('O_sd: dim %d (s,a)'%d)     
                                
#                             plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim*2)
#                             plt.imshow(R); plt.colorbar(); plt.ylabel('s'); plt.clim(-6,2)
#                             plt.title('R(s,a)')
                            
#                             ii += 2+n_A+n_dim*2
                            
#                             for i,save_dict in enumerate(res):
#                                 params = save_dict['best_EM_params']
#                                 pi,T,O,R = params
                                

#                                 plt.subplot(n_row,1+n_A+n_dim*2+1,ii+1)
#                                 plt.imshow(pi_wit[None,:]); plt.colorbar(); plt.ylabel('EM best init %d' %i); plt.clim(0,1)
                                            
#                                 for a in range(n_A):
#                                     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+a)
#                                     plt.imshow(T[:,:,a]); plt.colorbar(); plt.ylabel('s\''); plt.clim(0,1)
    
#                                 for d in range(n_dim):
#                                     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+d)
#                                     plt.imshow(O[0][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(-1,2)
                                    
#                                 for d in range(n_dim):
#                                     plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim+d)
#                                     plt.imshow(O[1][d,:,:]); plt.colorbar(); plt.ylabel('s'); plt.clim(0,1)
                                    
#                                 plt.subplot(n_row,1+n_A+n_dim*2+1,ii+2+n_A+n_dim*2)
#                                 plt.imshow(R); plt.colorbar(); plt.ylabel('s'); plt.clim(-6,2)

#                                 ii += 2+n_A+n_dim*2

#                             plt.suptitle('%s_ndim%d_noise%.1f_N%d'%(env,n_dim,sig_good,N))
#                             plt.savefig(FIG_PATH+'learned_param_viz/EMinit_%s_ndim%d_noise%.1f_N%d.pdf' %(
#                                                 env,n_dim,sig_good,N))
#                             plt.show()
    


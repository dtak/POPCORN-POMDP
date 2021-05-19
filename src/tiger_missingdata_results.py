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



date='5oct'
RESULTS_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger_miss/results_'+date+'/'

EM_RESULTS_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/results/tiger_miss/results_'+date+'-EM/'

FIG_PATH = '/Users/josephfutoma/Dropbox/research/prediction_constrained_RL/experiments/PC-POMDP_experiment_figs/tiger_miss/figs_'+date+'/'

if not os.path.exists(FIG_PATH):
    os.makedirs(FIG_PATH)
    os.makedirs(FIG_PATH+'obj_grad_traces/')
    os.makedirs(FIG_PATH+'param_traces/')
    os.makedirs(FIG_PATH+'results/')
    os.makedirs(FIG_PATH+'learned_param_viz/')




import seaborn as sns

#5oct for aistats
sig_goods = np.array([0.3]) 
sig_others = np.array([0.3]) 
good_dim_meas_probs = np.array([0.05,0.1,0.2,0.3,0.5,0.7,0.9])
lambds = np.array([1e-1,1e0,1e1,1e2,np.inf]) 
n_envs = np.array([2])    
prune_nums = np.array([0]) #10
inits = np.array(['random','EM-random']) #random
ESS_penalties = np.array([0]) #0, 10
seeds = np.arange(1,11)
folds = np.arange(5)

N = 2500
env_name = 'tiger-miss' 
tiger_env = True

sig_other = 0.3
sig_good = 0.3
n_env = 2
prune_num = 0
ESS_penalty = 0

model_string = '%s_nenv%d_good-dim-meas-prob%.2f_siggood%.1f_sigother%.1f_N%d_\
lambd%.8f_init-%s_prune%d_ESSpenalty%.3f_seed%d_fold%d' 

all_te_lls = {}
all_te_polvals = {}

all_wit_polvals = {}
all_wit_lls = {}

all_EM_polvals = {}
all_EM_lls = {}

for good_dim_meas_prob in good_dim_meas_probs:
    for param_init in inits:
        for lambd in lambds:
            for fold in folds:

                lls = []
                polvals = []
                wit_lls = []
                wit_polvals = []
                EM_lls = []
                EM_polvals = []

                for seed in seeds:
                    model_path = RESULTS_PATH+model_string %(env_name,n_env,
                        good_dim_meas_prob,sig_good,sig_other,N,lambd,param_init,
                        prune_num,ESS_penalty,seed,fold) +'.p'

                    try:
                        save_dict = pickle.load(open(model_path,"rb"))       

                        wit_lls.append(save_dict['ll_te_true'])
                        wit_polvals.append(save_dict['PBVI_true_results'][0])

                        lls.append(save_dict['HMM_te_objs'])
                        polvals.append(save_dict['avg_te_returns_det'])

                        if param_init == 'EM-random': #pull out EM results too
                            EM_lls.append(save_dict['best_EMinit_te_ll'])
                            EM_polvals.append(save_dict['best_EMinit_avgreturns'])

                    except:
                        pass                    
                
                settings = (good_dim_meas_prob,param_init,lambd,fold)
                all_te_lls[settings] = lls
                all_te_polvals[settings] = polvals
                all_wit_polvals[settings] = wit_polvals
                all_wit_lls[settings] = wit_lls
                if param_init == 'EM-random': #pull out EM results too
                    all_EM_polvals[settings] = EM_polvals
                    all_EM_lls[settings] = EM_lls

pickle.dump([all_te_lls,all_te_polvals,all_wit_polvals,all_wit_lls,all_EM_polvals,all_EM_lls],
    open(RESULTS_PATH+'../'+date+'_all_aggregate_results.p','wb'))

[all_te_lls,all_te_polvals,all_wit_polvals,all_wit_lls,
    all_EM_polvals,all_EM_lls] = pickle.load(
    open(RESULTS_PATH+'../'+date+'_all_aggregate_results.p','rb'))



###### get updated EM results....

all_EM_objs = {}
all_EM_polvals = {}
all_EM_lls = {}

for good_dim_meas_prob in good_dim_meas_probs:
    for param_init in inits:
        for lambd in lambds:
            for fold in folds:

                EM_objs = []
                EM_lls = []
                EM_polvals = []

                for seed in seeds:
                    model_path = EM_RESULTS_PATH+'EMonly-'+model_string %(env_name,n_env,
                        good_dim_meas_prob,sig_good,sig_other,N,lambd,param_init,
                        prune_num,ESS_penalty,seed,fold) +'.p'

                    try:
                        save_dict = pickle.load(open(model_path,"rb"))       

                        if param_init == 'EM-random': #pull out EM results too
                            EM_lls.append(save_dict['best_EMinit_te_ll'])
                            EM_polvals.append(save_dict['best_EMinit_avgreturns'])
                            EM_objs.append(save_dict['best_EM_obj'])
                    except:
                        pass                    
                
                settings = (good_dim_meas_prob,param_init,lambd,fold)
                if param_init == 'EM-random': #pull out EM results too
                    all_EM_polvals[settings] = EM_polvals
                    all_EM_lls[settings] = EM_lls
                    all_EM_objs[settings] = EM_objs





#####

te_lls = {}
te_polvals = {}

wit_polvals = {}
wit_lls = {}

EM_polvals = {}
EM_lls = {}

for good_dim_meas_prob in good_dim_meas_probs:
    for prune_num in prune_nums:
        for param_init in inits:
            for ESS_penalty in ESS_penalties:
                for n_env in n_envs:
                    for lambd in lambds:

                        setting = (good_dim_meas_prob,param_init,lambd)
                        te_lls[setting] = []
                        te_polvals[setting] = []
                        wit_polvals[setting] = []
                        wit_lls[setting] = []
                        if param_init == 'EM-random':
                            EM_polvals[setting] = []
                            EM_lls[setting] = []

                        for fold in folds:
                            run =  (good_dim_meas_prob,param_init,lambd,fold)

                            if len(all_te_polvals[run]) > 0:

                                # a bit optimistic...get *best* policyy from all of each run
                                # best_vals = np.array([np.max(x) for x in all_te_polvals[run]])
                                # best_val_inds = np.array([np.argmax(x) for x in all_te_polvals[run]])
                                # best_seed = np.where(best_vals==np.max(best_vals))[0][0]
                                # te_lls[setting].append(all_te_lls[run][best_seed][best_val_inds[best_seed]])
                                # te_polvals[setting].append(all_te_polvals[run][best_seed][best_val_inds[best_seed]])

                                # a bit more realistic...take last result
                                vals = np.array([x[-1] for x in all_te_polvals[run]])
                                best_seed = np.where(vals==np.max(vals))[0][0]
                                te_lls[setting].append(all_te_lls[run][best_seed][-1])
                                te_polvals[setting].append(all_te_polvals[run][best_seed][-1])


                                wit_lls[setting].append(all_wit_lls[run][best_seed])
                                wit_polvals[setting].append(all_wit_polvals[run][best_seed])

                                if param_init == 'EM-random':
                                    objs = np.array(all_EM_objs[(good_dim_meas_prob,param_init,1.0,fold)])
                                    best_seed = np.where(objs==np.max(objs))[0][0]

                                    EM_lls[setting].append(all_EM_lls[(good_dim_meas_prob,param_init,1.0,fold)][best_seed])
                                    EM_polvals[setting].append(all_EM_polvals[(good_dim_meas_prob,param_init,1.0,fold)][best_seed])


# copy over extra EM results from EM-random init for random init for easier plotting
for good_dim_meas_prob in good_dim_meas_probs:
    for lambd in lambds:
        setting = (good_dim_meas_prob,'random',lambd)
        setting2 = (good_dim_meas_prob,'EM-random',1.0)
        EM_lls[setting] = EM_lls[setting2]
        EM_polvals[setting] = EM_polvals[setting2]



##### df to write out

import pandas as pd
pd.set_option("display.max_columns",101)

col_names = ['good_dim_meas_prob',
    'oracle_policy_value_mean',
    'oracle_policy_value_stderrs',
    'EM_policy_value_mean',
    'EM_policy_value_stderrs',
    'RLonly_policy_value_mean',
    'RLonly_policy_value_stderrs',
    ]            
for i,lambd in enumerate(lambds[:-1]):
    col_names.append('lambd'+str(lambd)+'_policy_value_mean')
    col_names.append('lambd'+str(lambd)+'_policy_value_stderrs')
col_names.append('oracle_policy_lls_mean')
col_names.append('oracle_policy_lls_stderrs')
col_names.append('EM_policy_lls_mean')
col_names.append('EM_policy_lls_stderrs')
col_names.append('RLonly_policy_lls_mean')
col_names.append('RLonly_policy_lls_stderrs')
for i,lambd in enumerate(lambds[:-1]):
    col_names.append('lambd'+str(lambd)+'_policy_lls_mean')
    col_names.append('lambd'+str(lambd)+'_policy_lls_stderrs')

#####




all_rows = []

for good_dim_meas_prob in good_dim_meas_probs:
    for param_init in inits:

        this_wit_vals_means = []; this_wit_vals_ses = []
        this_EM_vals_means = []; this_EM_vals_ses = []
        this_lam_vals_means = []; this_lam_vals_ses = []
        this_linf_vals_means = []; this_linf_vals_ses = []

        this_wit_lls_means = []; this_wit_lls_ses = []
        this_EM_lls_means = []; this_EM_lls_ses = []
        this_lam_lls_means = []; this_lam_lls_ses = []
        this_linf_lls_means = []; this_linf_lls_ses = []

        for n_env in n_envs:
            v = wit_polvals[(good_dim_meas_prob,param_init,np.inf)]
            this_wit_vals_means.append(np.mean(v))
            this_wit_vals_ses.append(np.std(v)/np.sqrt(len(v)))
            v = wit_lls[(good_dim_meas_prob,param_init,np.inf)]
            this_wit_lls_means.append(np.mean(v))
            this_wit_lls_ses.append(np.std(v)/np.sqrt(len(v)))

            v = EM_polvals[(good_dim_meas_prob,param_init,np.inf)]
            this_EM_vals_means.append(np.mean(v))
            this_EM_vals_ses.append(np.std(v)/np.sqrt(len(v)))
            v = EM_lls[(good_dim_meas_prob,param_init,np.inf)]
            this_EM_lls_means.append(np.mean(v))
            this_EM_lls_ses.append(np.std(v)/np.sqrt(len(v)))

            v = te_polvals[(good_dim_meas_prob,param_init,np.inf)]
            this_linf_vals_means.append(np.mean(v))
            this_linf_vals_ses.append(np.std(v)/np.sqrt(len(v)))
            v = te_lls[(good_dim_meas_prob,param_init,np.inf)]
            this_linf_lls_means.append(np.mean(v))
            this_linf_lls_ses.append(np.std(v)/np.sqrt(len(v)))

        for lambd in lambds[:-1]: 
            ll_means = []; ll_ses = []
            val_means = []; val_ses = []
            for n_env in n_envs:
                v = te_polvals[(good_dim_meas_prob,param_init,lambd)]
                val_means.append(np.mean(v))
                val_ses.append(np.std(v)/np.sqrt(len(v)))
                v = te_lls[(good_dim_meas_prob,param_init,lambd)]
                ll_means.append(np.mean(v))
                ll_ses.append(np.std(v)/np.sqrt(len(v)))
            this_lam_vals_means.append(val_means); this_lam_vals_ses.append(val_ses)
            this_lam_lls_means.append(ll_means); this_lam_lls_ses.append(ll_ses)



        #EDIT 10/6/19: get and write out the results for the final results figure we use:

        if param_init=='random':

            this_row = [[good_dim_meas_prob],this_wit_vals_means,
                this_wit_vals_ses,this_EM_vals_means,this_EM_vals_ses,
                this_linf_vals_means,this_linf_vals_ses]
            for i,lambd in enumerate(lambds[:-1]):
                this_row.append(this_lam_vals_means[i])
                this_row.append(this_lam_vals_ses[i])
            this_row.append(this_wit_lls_means)
            this_row.append(this_wit_lls_ses)
            this_row.append(this_EM_lls_means)
            this_row.append(this_EM_lls_ses)
            this_row.append(this_linf_lls_means)
            this_row.append(this_linf_lls_ses)
            for i,lambd in enumerate(lambds[:-1]):
                this_row.append(this_lam_lls_means[i])
                this_row.append(this_lam_lls_ses[i])
            this_row = [x[0] for x in this_row]
            all_rows.append(this_row)



#######################

dat = pd.DataFrame(all_rows,columns=col_names)

out_path = RESULTS_PATH+'../mean_stderr_results_%s_init-%s.csv' %(
                    env_name,param_init)
dat.to_csv(out_path,index=False)





        #still mid-loop....

        #finally plot, yay

        n_lam = 4
        width = 6
        size = 350
        lam_colors = np.linspace(0.3,1.8,n_lam)
        lam_colors = [lighten_color('r',c) for c in lam_colors]


        plt.close('all')
        plt.figure(figsize=(24,11))

        plt.scatter(y=this_wit_vals_means,x=this_wit_lls_means,s=size)
        plt.errorbar(y=this_wit_vals_means,x=this_wit_lls_means,yerr=this_wit_vals_ses,
            xerr=this_wit_lls_ses,label='oracle',elinewidth=width)

        plt.scatter(y=this_EM_vals_means,x=this_EM_lls_means,s=size)
        plt.errorbar(y=this_EM_vals_means,x=this_EM_lls_means,yerr=this_EM_vals_ses,
            xerr=this_EM_lls_ses,label='2 stage EM',elinewidth=width)

        plt.scatter(y=this_linf_vals_means,x=this_linf_lls_means,s=size)
        plt.errorbar(y=this_linf_vals_means,x=this_linf_lls_means,yerr=this_linf_vals_ses,
            xerr=this_linf_lls_ses,label='RL loss only',elinewidth=width)

        for i in range(n_lam):
            plt.scatter(y=this_lam_vals_means[i],x=this_lam_lls_means[i],color=lam_colors[i],s=size)
            plt.errorbar(y=this_lam_vals_means[i],x=this_lam_lls_means[i],yerr=this_lam_vals_ses[i],color=lam_colors[i],
                xerr=this_lam_lls_ses[i],label='lambda %.1f'%(lambds[:-1][i]),elinewidth=width)

        plt.title('Generative & Decision-Making Results')
        plt.ylabel('Avg. Policy Value')
        plt.xlabel('Avg. HMM log marginal likelihood')
        plt.legend()
        plt.grid(alpha=.2)

        # plt.show()

        plt.savefig(FIG_PATH+'results/measprob%.2f_init-%s_last.pdf' %( #last vs best
                            good_dim_meas_prob,param_init),
                            bbox_inches='tight', pad_inches=0.5)              
        
        plt.close('all')





#####
##### make convergence plots, tracing objectives & gradient norms per iter
#####

n_seeds = len(seeds)

for good_dim_meas_prob in good_dim_meas_probs:
    for param_init in inits:
        for lambd in lambds:
            for fold in folds:

                res = []
                this_seeds = []
                for seed in seeds:
                    model_path = RESULTS_PATH+model_string %(env_name,n_env,
                        good_dim_meas_prob,sig_good,sig_other,N,lambd,param_init,
                        prune_num,ESS_penalty,seed,fold) +'.p'                 
                    try:
                        this_res = pickle.load(open(model_path,"rb"))       
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
                        
                        returns = save_dict['avg_te_returns_det'] 
                        true_returns = np.max(save_dict['PBVI_true_results'][0])
                        plt.plot(50*np.arange(len(returns)),returns,label='actual est return')
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
                        te_HMM_obj = save_dict['HMM_te_objs']
                        plt.plot(50*np.arange(1,len(te_HMM_obj)+1),np.array(te_HMM_obj),label='learned')
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

                plt.savefig(FIG_PATH+'obj_grad_traces/measprob%.1f_init-%s_lambd%.4f_fold%d.pdf' %(
                    good_dim_meas_prob,param_init,lambd,fold),
                    bbox_inches='tight', pad_inches=0.5)   

                plt.close(fig)

                # plt.show()
                 




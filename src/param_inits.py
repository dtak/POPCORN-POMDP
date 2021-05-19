
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



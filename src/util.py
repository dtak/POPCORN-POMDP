#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:22:50 2018
@author: josephfutoma
"""

import autograd.numpy as np
from autograd.core import make_vjp
from autograd.builtins import tuple as atuple
from autograd.extend import vspace
from autograd.wrap_util import unary_to_nary

import autograd.extend
from autograd.scipy.misc import logsumexp
from autograd import elementwise_grad, grad
from autograd.extend import primitive, defvjp
import autograd.scipy.stats as stat

EPS = 1e-16
MIN_EPS = 1e-11

MIN_VAL=1e-200
MAX_VAL=1 - 1e-14


def adam(x,g,iter_num,step_size,m,v,b1=0.9, b2=0.999, eps=10**-8):
    """Take gradient step for adam & update m,v
    """
    # m = np.zeros(len(x))
    # v = np.zeros(len(x))
    m = (1 - b1) * g      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(iter_num + 1))    # Bias correction.
    vhat = v / (1 - b2**(iter_num + 1))
    x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x,m,v


def rprop(x,g,last_g,step_sizes,last_steps,obj,last_obj,upper_mult=1.2,lower_mult=0.5,upper_cap=.1,lower_cap=1e-8):
    """
    take gradient step for rprop
    we use iRprop+ from Fig 2, https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf
    x: flattened param vec to update
    g: current gradient
    last_g: grad from previous time step
    step_sizes: current set of step sizes
    last_steps: last param update that was added to get current x
    obj: current obj at this param vec
    last_obj: previous obj 
    """
    steps = np.zeros(len(x)) #final step to take

    #get 3 cases
    same_signs = g*last_g>0
    opp_signs = g*last_g<0
    zero_signs = g*last_g==0

    #grad same sign; take step & increase step size
    step_sizes[same_signs] = np.clip(step_sizes[same_signs]*upper_mult,lower_cap,upper_cap)
    steps[same_signs] = -np.sign(g[same_signs])*step_sizes[same_signs]

    #grads opposite sign; decrease step size and maybe revert last step
    step_sizes[opp_signs] = np.clip(step_sizes[opp_signs]*lower_mult,lower_cap,upper_cap)
    if obj > last_obj: #objective got worse at this iter, take back last step
        steps[opp_signs] = -last_steps[opp_signs]
    g[opp_signs] = 0

    #last time grads were opposite sign, so now product is 0...
    steps[zero_signs] = -np.sign(g[zero_signs])*step_sizes[zero_signs]

    #finally, take the step
    x = x + steps

    return x,g,step_sizes,steps

def gd_nag(x,g,step_size,v,mom_decay=0.9):
    """
    take gradient step for Nesterov accelerated gradient
    Note that g is not the gradient at x, but rather at x - mom_decay*v
    """
    v = mom_decay*v + step_size*g
    x = x - v
    return x,v


@primitive
def logistic_sigmoid(x_real):
    ''' Compute logistic sigmoid transform from real line to unit interval.
    Numerically stable and fully vectorized.
    Args
    ----
    x_real : array-like, with values in (-infty, +infty)
    Returns
    -------
    p_real : array-like, size of x_real, with values in (0, 1)
    Examples
    --------
    >>> logistic_sigmoid(-55555.)
    0.0
    >>> logistic_sigmoid(0.0)
    0.5
    >>> logistic_sigmoid(55555.)
    1.0
    >>> logistic_sigmoid(np.asarray([-999999, 0, 999999.]))
    array([ 0. ,  0.5,  1. ])
    '''
    if not isinstance(x_real, float):
        out = np.zeros_like(x_real)
        mask1 = x_real > 50.0
        out[mask1] = 1.0 / (1.0 + np.exp(-x_real[mask1]))
        mask0 = np.logical_not(mask1)
        out[mask0] = np.exp(x_real[mask0])
        out[mask0] /= (1.0 + out[mask0])
        return out
    if x_real > 50.0:
        pos_real = np.exp(-x_real)
        return 1.0 / (1.0 + pos_real)
    else:
        pos_real = np.exp(x_real)
        return pos_real / (1.0 + pos_real)

def _logistic_sigmoid_not_vectorized(x_real):
    if x_real > 50.0:
        pos_real = np.exp(-x_real)
        return 1.0 / (1.0 + pos_real)
    else:
        pos_real = np.exp(x_real)
        return pos_real / (1.0 + pos_real)


# Definite gradient function via manual formula
def _vjp__logistic_sigmoid(ans, x):
    def _my_gradient(g, x=x, ans=ans):
        x = np.asarray(x)
        return np.full(x.shape, g) * ans * (1.0 - ans)
    return _my_gradient
defvjp(
    logistic_sigmoid,
    _vjp__logistic_sigmoid,
    )


def inv_logistic_sigmoid(
        p, do_force_safe=True):
    ''' Compute inverse logistic sigmoid from unit interval to reals.
    Numerically stable and fully vectorized.
    Args
    ----
    p : array-like, with values in (0, 1)
    Returns
    -------
    x : array-like, size of p, with values in (-infty, infty)
    Examples
    --------
    >>> np.round(inv_logistic_sigmoid(0.11), 6)
    -2.090741
    >>> np.round(inv_logistic_sigmoid(0.5), 6)
    0.0
    >>> np.round(inv_logistic_sigmoid(0.89), 6)
    2.090741
    >>> p_vec = np.asarray([
    ...     1e-100, 1e-10, 1e-5,
    ...     0.25, 0.75, .9999, 1-1e-14])
    >>> np.round(inv_logistic_sigmoid(p_vec), 2)
    array([-230.26,  -23.03,  -11.51,   -1.1 ,    1.1 ,    9.21,   32.24])
    '''
    if do_force_safe:
        p = np.minimum(np.maximum(p, MIN_VAL), MAX_VAL)
    return np.log(p) - np.log1p(-p)

def to_safe_common_arr(p):
    p = np.minimum(np.maximum(p, MIN_VAL), MAX_VAL)
    return p    


def softplus(x):
    """
    map back from unconstrained space to positive reals
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

def inv_softplus(x):
    """
    map from positive reals to unconstrained space
    """
    return np.log1p(-np.exp(-np.abs(x))+1e-16) + np.maximum(x,0)


def to_proba_vec(reals_X):
    assert reals_X.ndim==1
    
    Nm, =  reals_X.shape
    N = Nm + 1
    
    offset_N = -1.0 * np.log(N - np.arange(1.0, N))

    fracs = logistic_sigmoid(reals_X + offset_N)
    tmp = 1.0 - fracs
    
    cumprod = np.array([np.prod(tmp[:(nn+1)], axis=0) for nn in range(N-1)])
    proba_X = np.concatenate([fracs[:1],fracs[1:] * cumprod[:-1],cumprod[-1:]],0)
    
    assert np.allclose(1.0, np.sum(proba_X))
    return proba_X


def from_proba_vec(proba_X):
    assert proba_X.ndim==1
    
    N, = proba_X.shape
    
    offset_N = -1.0 * np.log(N - np.arange(1.0, N))

    cumsum_probs = np.maximum(1e-16,1.0 - np.cumsum(proba_X[:-1], axis=0))
    
    fracs = np.concatenate([proba_X[:1],proba_X[1:] / cumsum_probs],0)
      
    reals_X = (inv_logistic_sigmoid(fracs[:-1]) - offset_N)
    return reals_X



def to_proba_3darr(reals_X):
    ''' Convert unconstrained probabilities back to 3D probability array
    Should handle any non-nan, non-inf input without numerical problems.
    Args
    ----
    reals_X : 3D array, size N1-1 x N2 x N3
        Contains real values.
    Returns
    -------
    proba_X : 3D array, size N1 x N2 x N3
        Minimum value of any entry will be min_eps
        Maximum value will be 1.0 - min_eps
        Each row will sum to 1.0 (+/- min_eps)
    '''
    assert reals_X.ndim == 3
    N1m, N2, N3 = reals_X.shape
    N1 = N1m + 1
    
    offset_N1 = -1.0 * np.log(N1 - np.arange(1.0, N1))

    fracs = logistic_sigmoid(reals_X + offset_N1[:,None,None])
    tmp = 1.0 - fracs
    
    
    cumprod = np.concatenate([
        np.prod(tmp[:(nn+1),:,:], axis=0)[None,:,:]
            for nn in range(N1-1)] ,0)

    proba_X = np.concatenate([
        fracs[:1,:,:],
        fracs[1:,:,:] * cumprod[:-1,:,:],
        cumprod[-1:,:,:],
        ],0)
    assert np.allclose(1.0, np.sum(proba_X, axis=0))
    return proba_X


def from_proba_3darr(proba_X):
    ''' Transform normalized probabilities to unconstrained space.
    Args
    ----
    proba_X : 3D array, size N1 x N2 x N3
        first dim sums to 1, ie sum(X[:,i,j]) = 1
    Returns
    -------
    reals_X : 3D array, size (N1-1) x N2 x N3
        unconstrained real values
    '''
    assert proba_X.ndim == 3
    N1, N2, N3 = proba_X.shape
    
    offset_N1 = -1.0 * np.log(N1 - np.arange(1.0, N1))

    cumsum_probs = np.maximum(1e-16,1.0 - np.cumsum(proba_X[:-1,:,:], axis=0))
    
    fracs = np.concatenate([proba_X[:1,:,:],proba_X[1:,:,:] / cumsum_probs],0)
      
    reals_X = (inv_logistic_sigmoid(fracs[:-1,:,:]) - offset_N1[:,None,None])
    return reals_X



@unary_to_nary
def value_and_output_and_grad(fun, x):
    """Builds a function that returns:
            the value of the first output
            the gradient of the first output
            and the second output (that was presumably modified in the function)
    of a function that returns two outputs."""
    vjp, (ans, aux) = make_vjp(lambda x: atuple(fun(x)), x)
    return ans, aux, vjp((vspace(ans).ones(), vspace(aux).zeros()))

def draw_discrete(p,N=1,rng=np.random):
    """
    draw from a multinomial to get a discrete; but has to be same multinomial
    """
    draws = np.where(rng.multinomial(1,p,N))[1]
    if N==1:
        return draws[0]
    return draws


def vect_draw_discrete(prob_matrix,rng=np.random):
    """
    fast way to get draws from a bunch of different multinomials, with different parameters
    prob_matrix should be of dim N x S.
    N is number of different multinomials
    S is common dimension size of multinomials 
    """
    s = prob_matrix.cumsum(axis=1)
    r = rng.rand(prob_matrix.shape[0])[:,None]
    k = (s < r).sum(axis=1)
    return k



def draw_discrete_gumbeltrick(p,rng=None):
    """
    Gumbel trick for drawing a discrete
    """
    if rng is None:  
        U = np.random.uniform(0,1,len(p))
    else:
        U = rng.uniform(0,1,len(p))
    return np.argmax(np.log(p+EPS)-np.log(-np.log(U+EPS)+EPS))
    


def is_close(x,y):
    return np.mean(np.abs(x-y)<1e-6)


def round_(x,num=2):
    return np.round(x*10**num)/10**num  


def nd(f, x, eps=1e-8):
    """
    numerical derivative, for debugging.
    f: function that takes in x 
    x: flattened params argument to function f
    eps: tunable param for numerical diff 
    """
    n = len(x)
    g = np.zeros(n)
    x_ = np.copy(x)
    
    for i in range(n):
        #more accurate: 4th order         
        tmp = 0 #grad in this dim 

        x_[i] -= 2*eps 
        tmp += 1/12*f(x_)
        x_[i] = x[i]
        
        x_[i] -= eps 
        tmp += -2/3*f(x_)
        x_[i] = x[i]

        x_[i] += eps 
        tmp += 2/3*f(x_)
        x_[i] = x[i]

        x_[i] += 2*eps 
        tmp += -1/12*f(x_)
        x_[i] = x[i]
        
        g[i] = tmp/eps        
    return g
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Wrapper around the sepsis simulator to get trajectories & optimal policy out.
Behavior for our purposes will be eps-greedy of optimal.

Lots of code here is directly copied from the original gumbel-max-scm repo.s

@author: josephfutoma
"""

import numpy as np
import sepsis_simulator.cf.counterfactual as cf
import sepsis_simulator.cf.utils as utils
import pandas as pd
import pickle
import itertools as it
from scipy.linalg import block_diag

# Sepsis Simulator code
from sepsis_simulator.sepsisSimDiabetes.State import State
from sepsis_simulator.sepsisSimDiabetes.Action import Action
from sepsis_simulator.sepsisSimDiabetes.DataGenerator import DataGenerator

import matplotlib.pyplot as plt
import seaborn as sns



def get_optimal_and_soft_policy(PHYS_EPSILON,DISCOUNT):

	# These are properties of the simulator, do not change
	n_actions = Action.NUM_ACTIONS_TOTAL
	n_components = 2

	# Get the transition and reward matrix from file
	with open("../data/sepsisSimData/diab_txr_mats-replication.p", "rb") as f:
	    mdict = pickle.load(f)

	tx_mat = mdict["tx_mat"]
	r_mat = mdict["r_mat"]

	tx_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
	r_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

	for a in range(n_actions):
	    tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
	    r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])


	fullMDP = cf.MatrixMDP(tx_mat_full, r_mat_full)
	fullPol = fullMDP.policyIteration(discount=DISCOUNT, eval_type=1)

	physPolSoft = np.copy(fullPol)
	physPolSoft[physPolSoft == 1] = 1 - PHYS_EPSILON
	physPolSoft[physPolSoft == 0] = PHYS_EPSILON / (n_actions - 1)

	return fullPol, physPolSoft


DISCOUNT = 0.99 # Used for computing optimal policies
PHYS_EPSILON = 0.14 #.05 # Used for sampling using physician pol as eps greedy
PROB_DIAB = 0.2 #what proportion to have diabetes; for our purposes treat as an obs we see each time

fullPol, physPolSoft = get_optimal_and_soft_policy(PHYS_EPSILON,DISCOUNT)

pickle.dump({'optimal_policy':fullPol, 'eps_greedy_policy_0.14':physPolSoft},
	open("../data/sepsisSimData/sepsis_simulator_policies.p", "wb"))
	





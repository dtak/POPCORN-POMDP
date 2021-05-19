Hypotension Management Dataset Readme

Date Last Modified: 5/31/2020
Author: Joe Futoma (@Joe on dtak slack, jfutoma14@gmail.com)

If you run into any issues, see if you have similar versions of numpy & pandas:
I built everything using software versions:
	python 3.7
	numpy 1.18.4
	pandas 1.0.4

This dataset is all derived from mimic-iii-v1.4.

------------------------------------------------------------------------------------------

Processed data that is ready for RL or other kinds of modeling can be found here, in the model_data folder:
	/n/dtak/mimic-iii-v1-4/hypotension_management_dataset/model_data

Scripts I used to convert the files in mimic-iii-v1-4/query-data to model-ready data can be found in the data_cleaning dir:
	- data_clean.py: does most of the heavy lifting
	- data_load_utils.py: helper functions to load in data files from query-data and merge as appropriate. 
	- dataclean_EDA.py: pretty messy file with a bunch of random EDA / visualization bits from various parts of the data cleaning process; may be helpful.

There are also some scripts in behavior_policy that outline a very simple means to learning a behavior policy in order to use this data for off-policy RL:
	- learn_beh_policy_knn.py: my current (relatively unprincipled) way of learning a behavior policy for downstream RL modeling.
	- beh_policy_knn_weights.py: set of weights I was using in the past for KNN, for different / older versions of this dataset. 

------------------------------------------------------------------------------------------

Description of files in model_data. But first, some general notes about how things are setup:

 
	- NOTE: there is 1 more state than action & rewards. this is because at the end of the trajectory, we're assuming no further action is taken. but, we still need that last state to compute the final reward from the second-to-last state & last action, and we may want that last state if we're trying to learn a transition model, and need to know what state we wound up in after taking the last action.
	- I'm assuming that we start at time t=1 hour into ICU stay. So at time t=1, we receive state s_1 (i.e. variables measured in first hour of ICU admit or pre-ICU), take action a_1, get reward r_1, move to s_2, etc. 
	- Note that in the actions files, actions at t=0 are listed even though states start at t=1; this is so we're able to say what the inital action was, taken before the first state at t=1. We make the assumption that a0 can be conditioned on, so that RL will be starting at t=1. You might want a0 so you can add features to the state that depend on the last action, you also might want if you want to learn an initial state dist that depends on the initial action that we condition on, i.e. p(s_1 | a_0); and then later learn transitions p(s_t+1 | s_t, a_t)
	- Trajectories END at t=72. The last state used for RL is time t=71. So in total, at most there are 71 actions to consider, from times 1 to 71 (a_0 exists but is conditioned on). There are 71 rewards, r_1 to r_71; in practice, these are a deterministic function of the next state (i.e. r_1 = f(s_2), ..., r_71 = f(s_72)).  States run up to t=72, but s_72 is terminal and we don't consider which actions were taken there (as stated, need s_72 to get the last reward, r_71).


	------ STATES

	* states_1hr.p: a pickled python dict. 
		keys are ICU IDs, there should be 15,653 in total. values are pandas dataframes with the following state variables (note some of these are imputed values):

		Static / Demographic / Other vars:
			- Times: time, from 1 up to 72 at most 
			- normed_time: just Times/72; the fraction of the first 72 hours we're at. Can be useful as explicit feature in state space, since knowing absolute time we're at is informative.
			- age: continuous, capped at 90
			- is_F: binary sex variable
			- surg_ICU: binary variable for if in surgical ICU
			- is_not_white: binary ethnicity var
			- is_emergency: binary, was admission emergency
			- is_urgent: binary, was admission urgent
			- hrs_from_admit_to_icu: how long between hospital admission & ICU admission, if short they basicallyy were immediately sent to ICU

		Labs:
			- bicarbonate: 
			- bun: measure of kidney & liver health
			- creatinine: too high indicates kidney problems
			- fio2: (fraction inspired oxygen): 21% if on room air, otherwise may be higher if on ventilator
			- glucose:
			- hct (hematocrit):
			- lactate: high values (eg >2) are bad & indicative of potential organ damage
			- magnesium:
			- platelets:
			- potassium:
			- sodium:
			- wbc (white blood cell count): too high means likely an infection
			- alt: liver marker
			- ast: liver marker
			- bilirubin_total: liver marker
			- hgb (hemoglobin):
			- pco2:
			- po2:
			- weight: may be taken at various times during admission; when taken there's probably some reason.

		Vitals:
			- hr (heart rate):
			- spo2 (pulse oximetry):
			- temp (temperature in C):
			- urine (urine output):
			- spontaneousrr (respiration rate):
			- dbp (diastolic BP):
			- map (mean arterial pressure):
			- sbp (systolic BP):
			- GCS (glasgow coma score; 1-15, higher = more awake & alert):

		Indicators: variables that denote how recently the measurement was taken, since the act of measuring something is often informative.
			- X_ind: indicator for was variable measured in last hour. Eg at s_t, X_ind says if X was measured between t-1 & t. Exists for all labs/vitals except gfr, since gfr is a direct function of creatinine. If the indicator is 0, then either the population median is used (if it's never been measured for this patient), or else it's imputed using the most recent value.
			- X_8ind: indicator if X has been measured in the last 8 hours. Only included for labs, since vitals are taken very often. 
			- X_everind: indicator if X has ever been measured in this ICU stay so far. Once this indicator turns on, it stays on indefinitely. Again only exists for labs, and helpful as some labs (e.g. ALT/AST) are not always taken, and may only be ordered once; knowing there was a reason to take them at *some* past point may be useful.

		NOTE: the continuous-valued variables are all on their ORIGINAL scales, and extreme physiologically infeasible values have been pruned. For many variables, you probably want to standardize before using in modeling. Also, for some you may want to log-transform as well; some labs (e.g. ALT & AST) can take on quite extreme values that vary by multiple orders of magnitude from normal.


	------ ACTIONS

	* actions_raw_1hr.p: similar in structure to states_1hr.p, a pickled python dict.
		keys ICU IDs. values are pandas dataframes with these fields:
			Times: unlike states, starts at 0, ends at 71 (at most).
			Fluids: total fluid amount given between current time t & next time t+1
			Vasopressors: total vasopressor amount given between current time t & next time t+1
			total_all_prev_vasos: total cumulative amount of vasopressors given from time 0 up until now. May be useful to append this as an additional state feature in modeling, as it's crucial to know how much treatment has already been given up until now. 
			total_all_prev_fluids: total cumulative amount of IV fluid boluses given from time 0 up until now. May be useful to append this as an additional state feature in modeling, as it's crucial to know how much treatment has already been given up until now.
			total_last_8hrs_vasos: total amount of vasopressor given over last 8 hours.
			total_last_8hrs_fluids: total amount of IV fluid boluses given over last 8 hours.

		NOTE: these values have *not* been filtered yet, and are still raw. Ie for fluids, when we eventually bin & discretize, we will toss anything < 200, but keeping here to preserve the raw values. Also: due to the arbitrary way time is discretized, some actions may be split across time windows, eg a 1L bolud might get split into two separate actions where 500mL each is given due to the crude discretization setup. 


	* actions_discretized_vaso5_fluid4_1hr.p.p: similar but with a few additional fields after binning actions into 20 total types, 5 vasopressor doses & 4 fluid doses. New fields:
			OVERALL_ACTION_ID: a discrete action id, between 0 and 19:
				0 = no action, 1 = 1 fluid 0 vaso, 2 = 2 fluid 0 vaso, 3 = 3 fluid 0 vaso;
				4 = 0 fluid 1 vaso, 5 = 1 fluid 1 vaso, ...  
				19 = 4 fluid 5 vaso
			FLUID_ID: discrete action id for just fluids, 0-4 (0=none, 4=largest)
			VASO_ID: discrete action id for just vaso, 0-5


	------ REWARDS

	* rewards_1hr.p: similar structure, dict with keys ICU IDs & values a pandas df, with just Times & Rewards.

		Rewards: A direct function of MAP values and urine values, using a piecewise linear function (see data_clean.pyy for details). Want to maintain a target of at least 65, <55 or so is very bad. MAP 65 or above results in reward of 1. For moderately low values in 55-65, moderate penalty is assessed, unless urine output is high (>30). Values 40-55 have a heavier penalty, dropping to reward of 0 at 40.


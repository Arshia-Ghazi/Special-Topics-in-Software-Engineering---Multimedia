Overview
--------
This repository provides a PyTorch implementation of a temporal-difference
actor-critic algorithm for the corticosteroid-in-sepsis RL problem described
in the paper I've translated.

Files
-----
- requirements.txt
- models.py: encoder + actor & critic
- data_loader.py: loader and synthetic-data generator
- agent.py: A2C/T-D actor-critic agent
- train.py: training loop over offline trajectories
- eval_offpolicy.py: Weighted Importance Sampling estimator + bootstrap for approximating HCOPE
- feature_importance.py: Random Forest clinician model & permutation importance for agent

How to prepare your data
------------------------
The code expects preprocessed trajectories. The paper converts clinical time series into 24h steps
and constructs trajectories from sepsis onset until ICU discharge. Each trajectory should be a dict with:
 - 'states' : np.array shape (T, F) where F is number of features (paper: 379)
 - 'actions': np.array shape (T,) int in {0,1,2,3,4} mapping corticosteroid dose bins
 - 'rewards': np.array shape (T,) floats (paper used ICU mortality as the reward signal - likely sparse final reward)
 - 'dones'  : np.array shape (T,) booleans with last element True

Example to save:
  import numpy as np
  trajectories = [ {'states':s,'actions':a,'rewards':r,'dones':d}, ... ]
  np.savez('amsterdam_trajs.npz', trajectories=trajectories)

Quickstart (synthetic demo)
---------------------------
1) Install dependencies:
   pip install -r requirements.txt

2) Train on synthetic demo:
   python train.py --data None --epochs 50 --input_dim 379
   Note: by None, I mean that you should not include "--data" in your command line.

   This will use the synthetic dataset in data_loader.build_synthetic_demo. Replace --data with your .npz to train on real data.

4) After training, evaluate off-policy using eval_offpolicy.wis_estimate. In a python session:
   from eval_offpolicy import fit_behavior_model, wis_estimate
   from data_loader import load_trajectories_npz
   from agent import A2CAgent
   trajs = load_trajectories_npz('amsterdam_trajs.npz')
   rf, behavior_prob_fn = fit_behavior_model(trajs)
   agent = A2CAgent(input_dim=379); agent.load('checkpoints/best_model.pth', map_location='cpu')
   res = wis_estimate(agent, trajs, behavior_prob_fn=behavior_prob_fn)
   print(res['wis'], res['lower95'])


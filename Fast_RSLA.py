# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:40:56 2024

@author: gangu
"""
from Robust_Safe_RL import Robust_Safe_RLA
from Machine_Rep import MachineReplacementEnv,RiverSwimEnv
import pickle
from itertools import product
import numpy as np


env_type = ["MR","RS"]
model_type=["DQN","A2C"]
model_chosen = 0
env_chosen = 1

with open(env_type[env_chosen]+"_all_policies_"+model_type[model_chosen],"rb") as f:
    total_policies = pickle.load(f)
f.close()

dist = 0
T = 1000
alpha = 0.1
lambda_,lambda_hat = 0.5,0.5
b = 3
eta = 0.1
zi = 0.5

obj = None
init_state = 0
if(env_chosen==0):
    obj = MachineReplacementEnv()
elif(env_chosen==1):
    obj = RiverSwimEnv()
    init_state = 3

#env = gym_MR_env(mr_obj, init_state, T)
nS,nA = obj.nS,obj.nA
rsla = Robust_Safe_RLA(nS)
R,C = obj.R,obj.C

p0 = np.ones(len(total_policies))*1/len(total_policies)
P0_hat = np.ones(len(total_policies))*1/len(total_policies)

rsla = Robust_Safe_RLA()

for t in range(T):
    v_list,c_list = [],[]
    for pol in range(len(total_policies)):
        vf,cf = rsla.find_min_vf_cf(total_policies[pol],us,R,C,init_state,dist)
        c_list.append(cf)
        v_list.append(vf)
    p0 = p0_hat*np.exp(alpha*(np.array(v_list)+lambda_*np.array(c_list)))
    lambda_ = np.min([np.max([lambda_hat+eta*(b-np.dot()),0]),zi])
    p0_hat[pol] = 
        


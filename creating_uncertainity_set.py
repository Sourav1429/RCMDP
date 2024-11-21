# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:15:29 2024

@author: gangu
"""

import numpy as np
from Machine_Rep import MachineReplacementEnv,RiverSwimEnv
#from perturbations import perturb_nominal_
import pickle

class perturb_nominal_:
    def __init__(self,accessed_vector,epsilon,is_function = 1):
        self.accessed_vector = accessed_vector
        self.eps = epsilon
        self.is_function = is_function
    def fit(self,d=1):
        if self.is_function == 1:
            return self.perturb_function()
        else:
            return self.perturb_vector(d)
    def perturb_vector(self,set_size = 1):
        vect_set = []
        for _ in range(set_size):
            dir_ = np.random.choice([-1,1],size = len(self.accessed_vector))
            perturb_val_vect = self.eps*np.random.random(len(self.accessed_vector))*dir_
            vect_set.append(self.accessed_vector + perturb_val_vect)
        return vect_set

class uncertainity:
    def __init__(self):
        pass
    def create_prob_space(self,P,nS,nA):
        for s in range(nS):
            for a in range(nA):
                P[a,s,: ] = P[a,s,:]/np.sum(P[a,s])
        return P
    def build_uncertainity_set(self,P_dict,nS,nA,sets):
        uncertain_set = []
        for s_a_set in range(sets):
            P = np.zeros((nA,nS,nS))
            for s in range(nS):
                for a in range(nA):
                    sel_P = P_dict[(s,a)][s_a_set]
                    sel_P = sel_P/np.sum(sel_P)
                    P[a,s] = sel_P
            uncertain_set.append(P)
        return uncertain_set



env_type = "RS"   #Enter env of model here RS or MR
env_ch = 1 #If env is RS this should be 1 or 0 if env = MR
model_ch = 0 # If model is DQN the 0 else if model is A2C then 1

env=None
if(env_ch==0):
    env = MachineReplacementEnv() 
elif(env_ch==1):
    env = RiverSwimEnv()
nS,nA = env.nS,env.nA
us = uncertainity()
P0 = np.ones((nA,nS,nS))
P0 = us.create_prob_space(P0, nS, nA)
print(P0)


un_eps = 0.1
P_dict = {}
no_of_perturb_Set = 10
for s in range(nS):
  for a in range(nA):
    pr_nom = perturb_nominal_(P0[a,s],un_eps,0)
    P_dict[(s,a)] = pr_nom.perturb_vector(no_of_perturb_Set)

uncertain_set = us.build_uncertainity_set(P_dict, nS, nA, no_of_perturb_Set)
with open("Uncertainity_set_"+env_type,"wb") as f:
    pickle.dump(uncertain_set, f)
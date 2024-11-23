# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 06:28:24 2024

@author: Sourav
"""

from Machine_Rep import MachineReplacementEnv,RiverSwimEnv
import pickle
import numpy as np

class Robust_Safe_RLA:
    def __init__(self,nS):
        self.nS = nS
    def form_probability_transition(self,P,policy,nS,nA):
        ret_mat = np.zeros((nS,nS))
        #print(policy.shape)
        for a in range(nA):
            for s in range(nS):
                for s_dash in range(nS):
                    #print("(s,a)=",s,a,s_dash)
                    #print(policy[int(s),int(a)])
                    #print(P[int(a),int(s),int(s_dash)])
                    ret_mat[int(s),int(s_dash)]+= policy[int(s),int(a)]*P[int(a),int(s),int(s_dash)]
        return ret_mat
    def get_vf(self,P, R, pi, gamma=0.9):
        """
        Compute the value function for an MDP under a given policy using matrix operations.

        Parameters:
            P (np.ndarray): Transition probability matrix of shape (nS, nS, nA).
            R (np.ndarray): Reward matrix of shape (nS, nA).
            pi (np.ndarray): Policy matrix of shape (nS, nA).
            gamma (float): Discount factor (0 <= gamma < 1).

        Returns:
            np.ndarray: Value function vector of shape (nS,).
        """

        # Compute the policy transition matrix P_pi
        nS = self.nS
        P_pi = P
        #print("pi=",pi)
        # Compute the policy reward vector R_pi
        R_pi = np.sum(pi * np.transpose(R), axis=1)
        #print(R_pi)

        # Solve the linear system (I - gamma * P_pi) V = R_pi
        I = np.eye(nS)
        V = np.linalg.solve(I - gamma * P_pi, R_pi)
        #print(V)
        return V
    def find_min_vf_cf(self,policy,us,R,C,init,dist=0):
        nA,nS = R.shape
        v_list=[]
        c_list=[]
        policy = np.array(policy)
        for P in us:
            #print("P=",P)
            #print("policy=",policy)
            Tr = self.form_probability_transition(P, policy, nS, nA)
            #print("Transform_prob=",Tr)
            vf,cf = self.get_vf(Tr,R,policy),self.get_vf(Tr,C,policy)
            #print("VF=",vf)
            #print("CF=",cf)
            #break
            if(dist==0):
                vf,cf = vf[init],cf[init]
            else:
                vf,cf = np.dot(vf,init),np.dot(cf,init)
            v_list.append(vf)
            c_list.append(cf)
        return np.min(v_list),np.min(c_list)
    def cross_product_of_keys(self,dictionary):
        """
        Finds the cross product of the elements of keys in a Python dictionary.

        Args:
            dictionary (dict): The dictionary to process.

        Returns:
            list: A list of tuples representing the cross product.
        """

        key_values = [dictionary[key] for key in dictionary]
        return list(product(*key_values))

env_type = ["MR","RS"]
model_type=["DQN","A2C"]
model_chosen = 1
env_chosen = 0

with open(env_type[env_chosen]+"_all_policies_"+model_type[model_chosen],"rb") as f:
    total_policies = pickle.load(f)
f.close()

with open("Uncertainity_set_"+env_type[env_chosen],"rb") as f:
    us = pickle.load(f)
#print(us)
f.close()

dist = 0
T = 10
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
p0_hat = np.ones(len(total_policies))*1/len(total_policies)

rsla = Robust_Safe_RLA(nS)

vf_list,cf_list = [],[]
for t in range(T):
    v_list,c_list = [],[]
    prev_probs = p0
    for pol in range(len(total_policies)):
        vf,cf = rsla.find_min_vf_cf(total_policies[pol],us,R,C,init_state,dist)
        c_list.append(cf)
        v_list.append(vf)
        p0[pol] = p0_hat[pol]*np.exp(alpha*(vf+lambda_*cf))
    c_list = np.array(c_list)
    v_list = np.array(v_list)
    lambda_ = np.min([np.max([lambda_hat+eta*(b-np.dot(prev_probs,c_list)),0]),zi])
    p0_hat = p0_hat*np.exp(alpha*(v_list + lambda_*c_list))
    lambda_hat = np.min([np.max([lambda_+eta*(b-np.dot(p0,c_list)),0]),zi])
    cf_,vf_ = np.dot(p0,c_list),np.dot(p0,v_list)
    vf_list.append(vf_)
    cf_list.append(cf_)
data_dict={'vf':np.array(vf_list),'cf':np.array(cf_list)}
import pandas as pd
df = pd.DataFrame(data_dict)
df.to_excel('VF_CF_values_Fast_RSLA_'+env_type[env_chosen]+"_"+model_type[model_chosen]+".xlsx")

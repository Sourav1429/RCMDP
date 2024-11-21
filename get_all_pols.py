# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:07:41 2024

@author: gangu
"""

import numpy as np
from itertools import product
from Machine_Rep import MachineReplacementEnv,RiverSwimEnv
import torch
import pickle

import torch
import numpy as np
import torch.nn as nn

class CustomA2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CustomA2C, self).__init__()
        
        # Shared network for feature extraction
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Actor network
        self.actor = nn.Linear(128, action_dim)
        
        # Critic network
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared_network(state)
        
        # Actor: action probabilities
        action_logits = self.actor(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Critic: state value
        state_value = self.critic(features)
        
        return action_probs, state_value


class CustomDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CustomDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)  # Output Q-values for each action
        )

    def forward(self, state):
        q_values = self.network(state)  # Q-values
        probabilities = torch.softmax(q_values, dim=-1)  # Convert to probabilities
        return probabilities

class get_policy_combinations:
    def __init__(self,model,states,nS,nA,d,epsilon):
        self.model = model
        self.states = states
        self.nS = nS
        self.nA = nA
        self.d = d
        self.k = int(epsilon/d)
    def one_hot(self,nS,s):
        ret_val= np.zeros(nS)
        ret_val[s]=1
        return ret_val
    
    def make_probs(self,pr):
        pr = np.exp(pr)
        return pr/np.sum(pr)
    
    def create_prob_space(P,nS,nA):
        for s in range(nS):
            for a in range(nA):
                P[a,s,: ] = P[a,s,:]/np.sum(P[a,s])
        return P
    
    def generate_epsilon_close_distributions(self,model, states, nS,nA, d, k,ch = 0):
        epsilon_close_distributions = {}

        for s in states:
            # Get the original probability distribution for state s
            s1 = torch.tensor(self.one_hot(nS, s)).float()
            original_probs = None
            if(ch==1):
                original_probs = model(s1)[0].cpu().detach().numpy()
            else:
                original_probs = model(s1).cpu().detach().numpy()
            original_probs = self.make_probs(original_probs)
            #print(original_probs)
            
            # Generate perturbed ranges for each action
            action_ranges = [
                np.clip(
                    np.arange(min(p, p - k*d), max(p, p + k * d) + d, d),
                    p-k*d, p+k*d
                )
                for p in original_probs
            ]
            #print("============")
            #print(action_ranges)
            
            # Generate all combinations of perturbed probabilities
            all_combinations = list(product(*action_ranges))
            
            # Filter combinations to ensure they sum to 1 (within a tolerance)
            #valid_distributions = make_distribution(all_combination)
            valid_distributions = [
                np.array(comb)
                for comb in all_combinations
                if np.isclose(sum(comb), 1, atol=1e-6)
            ]
            #print("********************")
            #print(valid_distributions)
            
            epsilon_close_distributions[s] = valid_distributions

        return epsilon_close_distributions
    
    def __ret_policies__(self,ch):
        return self.generate_epsilon_close_distributions(self.model, self.states, self.nS,self.nA, self.d, self.k,ch)

env_type = "RS"   #Enter env of model here RS or MR
model_nm = "River_swim_DQN" #Enter the complete model name whether it is RS env DQn or RS A2C or MR DQN or MR A2C
env_ch = 1 #If env is RS this should be 1 or 0 if env = MR
model_ch = 0 # If model is DQN the 0 else if model is A2C then 1
policy_function = torch.load(model_nm)
policy_perturbation_epsilon = 0.1
policy_discretization_rate = 0.05

env=None
if(env_ch==0):
    env = MachineReplacementEnv() 
elif(env_ch==1):
    env = RiverSwimEnv()
nS,nA =env.nS,env.nA
gp = get_policy_combinations(policy_function, np.arange(nS,dtype=np.int16),nS, nA, policy_discretization_rate, policy_perturbation_epsilon)
#P0 = np.ones((nA,env.observation_space_size(),env.observation_space_size()))
#P0 = gp.create_prob_space(P0, nS, nA)

possible_policies = gp.__ret_policies__(model_ch)

print(possible_policies)
with open("all_policies_"+env_type+"_"+model_nm,"wb") as f:
    pickle.dump(possible_policies,f)







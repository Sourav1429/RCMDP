# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:15:51 2024

@author: gangu
"""

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

def one_hot(nS,s):
    ret_val = np.zeros(nS,dtype=np.double)
    ret_val[s]=1
    return ret_val

nS = 6
model = torch.load("River_swim_A2C")
for i in range(nS):
    print(model(torch.tensor(one_hot(nS,i),dtype=torch.float32)).cpu().detach().numpy())

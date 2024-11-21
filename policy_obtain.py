# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:04:04 2024

@author: gangu
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Machine_Rep import MachineReplacementEnv,RiverSwimEnv

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


class genEnvWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        return np.array([self.env.reset()], dtype=np.float32)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.array([state], dtype=np.float32), reward, done, info
    

def model_choice(wrapped_env,ch,env_nm,N):
    state_dim = wrapped_env.env.nS
    action_dim = wrapped_env.env.nA
    if(ch==0):
        # DQN setup
        dqn_model = CustomDQN(state_dim, action_dim)
        optimizer = optim.Adam(dqn_model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        # Training loop (simplified example)
        state = wrapped_env.reset()
        print(state)
        for _ in range(N):  # Training iterations
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Get action probabilities
            action_probs = dqn_model(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # Step environment
            next_state, reward, done, _ = wrapped_env.step(action)
            
            # Compute target Q-value
            target = reward + 0.99 * torch.max(dqn_model(torch.tensor(next_state, dtype=torch.float32))).item()
            target = torch.tensor([target], dtype=torch.float32)
            
            # Compute loss and backpropagate
            q_value = dqn_model(state_tensor)[0, action]
            loss = loss_fn(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            if done:
                state = wrapped_env.reset()
        torch.save(dqn_model,env_nm+"_DQN")
    else:
        a2c_model = CustomA2C(state_dim, action_dim)
        optimizer = optim.Adam(a2c_model.parameters(), lr=1e-3)
        
        # Training loop (simplified example)
        state = wrapped_env.reset()
        for _ in range(N):  # Training iterations
            state_tensor = torch.tensor(state, dtype=torch.float32)
            
            # Get action probabilities and state value
            action_probs, state_value = a2c_model(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # Step environment
            next_state, reward, done, _ = wrapped_env.step(action)
            
            # Compute advantage
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            _, next_state_value = a2c_model(next_state_tensor)
            advantage = reward + 0.99 * next_state_value - state_value
            
            # Compute loss
            critic_loss = advantage.pow(2).mean()
            actor_loss = -(torch.log(action_probs[0, action]) * advantage.detach()).mean()
            loss = critic_loss + actor_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            if done:
                state = wrapped_env.reset()
        torch.save(a2c_model,env_nm+"_A2C")

env = MachineReplacementEnv()
wrapped_env = genEnvWrapper(env)
mod_ch = 0
N=10000
env_nm = "Machine_Replacement"
model_choice(wrapped_env,mod_ch,env_nm,N)
mod_ch=1
model_choice(wrapped_env,mod_ch,env_nm,N)

print("MR done")

env = RiverSwimEnv()
wrapped_env = genEnvWrapper(env)
mod_ch = 0
N=10000
env_nm = "River_swim"
model_choice(wrapped_env,mod_ch,env_nm,N)
mod_ch=1
model_choice(wrapped_env,mod_ch,env_nm,N)






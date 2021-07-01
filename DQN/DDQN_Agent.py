from DQN.DQN_Agent import DQN_Agent
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from libs.replay_memory import ReplayMemory, Experience
import os
from .DQN import DQN, DQN_cartpole
import resource

class DDQN_Agent(DQN_Agent):
    def __init__(self, env, env_name, config):
        super().__init__(env, env_name, config)

    def _calc_loss(self, batch):
        """
            Calculate L1-Loss for given batch.
        """
        states, actions, rewards, dones, next_states = batch
    
        states_v = torch.from_numpy(states).to(self.device)
        next_states_v = torch.from_numpy(next_states).to(self.device)
        actions_v = torch.from_numpy(actions).to(self.device)
        rewards_v = torch.from_numpy(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        
        state_action_values = self.policy_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states_v).max(1)[1]
            next_state_values = self.target_net(next_states_v).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = (next_state_values * self.hp_gamma) + rewards_v

        return F.smooth_l1_loss(state_action_values, expected_state_action_values)
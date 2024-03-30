import torch
import torch.nn.functional as F 
import copy
import torch.nn as nn
import numpy as np

from ssd.networks.util import create_mlp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, n_layers, hidden_size):
        super().__init__()
        self.actor_net = create_mlp(state_dim, action_dim, n_layers, hidden_size)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = self.actor_net(state)
        return self.max_action * torch.tanh(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_size):
        super().__init__()

        # Q1 architecture
        self.q1_net = create_mlp(state_dim + action_dim, 1, n_layers, hidden_size)
        
        # Q2 architecture
        self.q2_net = create_mlp(state_dim + action_dim, 1, n_layers, hidden_size)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1_net(sa)
        return q1


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        n_layers,
        hidden_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action, n_layers, hidden_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, n_layers, hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def sample_action(self, state):
        noise = np.random.normal(0, self.max_action * self.policy_noise, size=self.action_dim)
        action = (
            self.select_action(state) + noise
            ).clip(-self.max_action, self.max_action)
        return action
    

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, state, action, next_state, reward, not_done):
        self.total_it += 1

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None
        
        return critic_loss, actor_loss


    def save(self, save_file):
        model_dict =  {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict()
        }
        
        torch.save(model_dict, save_file)

    def load(self, filename):
        model_dict = torch.load(filename)

        self.critic.load_state_dict(model_dict["critic"])
        self.critic_target.load_state_dict(model_dict["critic_target"])
        self.critic_optimizer.load_state_dict(model_dict["critic_optimizer"])
        self.actor.load_state_dict(model_dict["actor"])
        self.actor_target.load_state_dict(model_dict["actor_target"])
        self.actor_optimizer.load_state_dict(model_dict["actor_optimizer"])

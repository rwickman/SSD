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
    def __init__(self, state_dim, action_dim, action_bounds, n_layers, hidden_size):
        super().__init__()
        self.actor_net = create_mlp(state_dim, action_dim, n_layers, hidden_size)
        
        self.action_bounds = action_bounds

        self.register_buffer(
            "action_scale", torch.tensor((action_bounds[1] - action_bounds[0]) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_bounds[1] + action_bounds[0]) / 2.0, dtype=torch.float32)
        )


    def forward(self, state):
        a = self.actor_net(state)
        return torch.tanh(a) * self.action_scale + self.action_bias 
        # return self.max_action * torch.tanh(a)


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
        action_bounds,
        n_layers,
        hidden_size,
        actor_hidden_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        expl_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4,
        optim="adam",
        max_norm=2.0,
    ):

        self.actor = Actor(state_dim, action_dim, action_bounds, n_layers, actor_hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_bounds, n_layers, actor_hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, n_layers, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, n_layers, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if optim == "adam":
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        else:
            print("USING RADAM")
            self.actor_optimizer = torch.optim.RAdam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = torch.optim.RAdam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.expl_noise = expl_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_norm = max_norm

        self.total_it = 0

    def sample_action(self, state):
        with torch.no_grad():
            noise = torch.normal(0, self.actor.action_scale * self.expl_noise)
            action = (
                self.select_action(state) + noise
                ).cpu().numpy().flatten().clip(self.action_bounds[0], self.action_bounds[1])
            
            return action

    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state)

    def train(self, state, action, next_state, reward, not_done):
        self.total_it += 1

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip) * self.actor_target.action_scale
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(self.action_bounds[0][0], self.action_bounds[1][0])

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q
            target_Q = target_Q.view(-1)


        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1.view(-1), target_Q) + F.mse_loss(current_Q2.view(-1), target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.total_it % 2048 == 0:
                print("current_Q1", current_Q1[:10].view(-1))
                print("current_Q2", current_Q2[:10].view(-1))
                print("target_Q", target_Q[:10].view(-1))
                print("not_done", not_done.sum())
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

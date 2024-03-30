import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import Counter
from ssd.networks.util import create_mlp


class StateDiscriminator(nn.Module):
    def __init__(self,
                 state_dim: int,
                 skill_dim: int,
                 num_species: int,
                 n_layers: int,
                 hidden_size: int,
                 device: str,
                 temperature: float,
                 tau: float,
                 use_skill_enc: bool = True):
        super().__init__()
        self.device = device        
        self.temperature = temperature
        self.tau = tau
        self.use_skill_enc = use_skill_enc
        
        self.latent_dim = skill_dim + num_species
        self.num_iter = 0

        self.num_species = num_species
        self._state_encoder = create_mlp(state_dim, self.latent_dim, n_layers, hidden_size)
        self._state_encoder_target = copy.deepcopy(self._state_encoder)
        if self.use_skill_enc:
            self._skill_species_encoder = create_mlp(skill_dim + num_species, self.latent_dim, n_layers, hidden_size)        

    def update_target(self):
        for param, target_param in zip(self._state_encoder.parameters(), self._state_encoder_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)        

    def encode_state(self, obs):
        with torch.no_grad():
            return self._state_encoder_target(obs)

    def encode_species(self, species_id):
        species_one_hot = F.one_hot(species_id.view(-1), self.num_species).to(self.device)
        return species_one_hot.float()

    def get_diversity_reward(self, next_obs, skill, species_id):
        self.num_iter += 1


        obs_latent, context_latent = self.forward(next_obs, skill, species_id)

        pred = F.normalize(obs_latent, dim=1) * F.normalize(context_latent, dim=1)

        rewards = pred.sum(dim=-1, keepdim=True)

        # if self.num_iter % 2048 == 0:
        #     print("next_obs", next_obs, next_obs.shape)
        #     print("skill", skill)
        #     print("species_id", species_id)
        #     print("species_id.unique()", Counter(species_id.tolist()))
        #     print("obs_latent", obs_latent, obs_latent.shape)
        #     print("context_latent", context_latent, context_latent)
        #     print("pred", pred )
        #     print("rewards.view(-1)", rewards.view(-1), rewards.min(), rewards.max())


        return rewards

    def compute_loss(self, obs, skill, species_id):
        batch_size = obs.shape[0]
        obs_latent, context_latent = self.forward(obs, skill, species_id)

        obs_latent = F.normalize(obs_latent, dim=1)
        context_latent = F.normalize(context_latent, dim=1)
        sim_matrix = torch.matmul(obs_latent, context_latent.T)

        # Extract the positive and negative pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positives = sim_matrix[mask.bool()].view(batch_size, -1)

        negatives = sim_matrix[~mask.bool()].view(batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, obs, skill, species_id):
        obs_latent =  self._state_encoder(obs)

        species_one_hot = self.encode_species(species_id)
        obs_latent = self._state_encoder(obs)
        if self.use_skill_enc:
            context_latent = self._skill_species_encoder(torch.cat((skill, species_one_hot), dim=1))
        else:
            context_latent = torch.cat((skill, species_one_hot), dim=1)

        return obs_latent, context_latent


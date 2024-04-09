import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
import hnswlib
import numpy as np

from ssd.networks.diversity.state_discriminator import StateDiscriminator
from ssd.networks.util import RunningMeanStd


class NoveltyBuffer:
    def __init__(self, latent_dim, running_mean_std, device, memory_capacity, k, max_novelty=10.0):
        self.latent_dim = latent_dim
        self.rms = running_mean_std
        self.device = device
        self.memory_capacity = memory_capacity
        self.k = k
        self.max_novelty = max_novelty
        if self.memory_capacity > 0:
            self.knn = hnswlib.Index(space='l2', dim=latent_dim)
            self.knn.init_index(
                max_elements=self.memory_capacity, ef_construction=100, M=32,
            )
            self.knn.set_num_threads(4)
            self.knn.set_ef(50)

        self._idx = 0
    
    def add_state(self, state):
        if self.memory_capacity > 0:
            self.knn.add_items(state, np.array([self._idx]))
            self._idx = (self._idx + 1) % self.memory_capacity
    
    def add_states(self, state):
        if self.memory_capacity > 0:
            idxs = np.arange(self._idx, self._idx + state.shape[0]) % self.memory_capacity
            self.knn.add_items(state, idxs)
            self._idx = (idxs[-1] + 1) % self.memory_capacity

    def apt_novelty(self, state):
        def local_dist(state):
            sim_matrix = torch.norm(state[:, None, :] - state[None, :, :], dim=-1, p=2)
            cur_state_dists, _ = sim_matrix.topk(min(self.k, state.shape[0]), dim=1, largest=False, sorted=True)  # (b1, k)
            cur_state_dists = cur_state_dists.detach().cpu().numpy()
            state = state.detach().cpu().numpy()

            dists = cur_state_dists
            return dists
        
        def rms_dists(dists):
            self.rms.update(torch.tensor(dists.reshape(-1)))
            dists = dists / ((self.rms.var.item() ** 0.5) + 1e-6)
            return dists
        
        try:
            if self.memory_capacity > 0:
                # Get distance from the buffer
                _, dists = self.knn.knn_query(state.detach().cpu().numpy(), k=min(self.k, state.shape[0]))
            else:
                dists = local_dist(state)

            dists = rms_dists(dists)
            self.rms.update(torch.tensor(dists.reshape(-1)))
            dists = dists / ((self.rms.var.item() ** 0.5) + 1e-6)

            return torch.tensor(np.log(1 + dists.mean(axis=1)), dtype=torch.float32, device=self.device).unsqueeze(1).clamp(max=self.max_novelty)
        except Exception as e:
            print(f"GIVEN {state.shape} self._idx {self._idx} Current Count {self.knn.get_current_count()} Max Elements {self.knn.get_max_elements()}")
            print(e)
        
        try:
            dists = local_dist(state)
            dists = rms_dists(dists)
            return torch.tensor(np.log(1 + dists.mean(axis=1)), dtype=torch.float32, device=self.device).unsqueeze(1)
        except Exception as e:
            return torch.zeros(state.shape[0],1, dtype=torch.float32, device=self.device)

class DiversityRewardNet(nn.Module):
    """Aggregates all the different networks used for the unsupervised reward."""
    def __init__(self, args, state_dim: int, device: str):
        super().__init__()
        self.args = args
        self.device = device
        self.state_dim = state_dim
        self.total_it = 0
        self.rms = RunningMeanStd()
        print("self.args.novelty_lam", self.args.novelty_lam)

        self.novelty_buffer = NoveltyBuffer(
            self.args.skill_dim + self.args.num_species,
            self.rms,
            self.device,
            self.args.novelty_capacity,
            self.args.knn_k,
        )

        self.state_disc = StateDiscriminator(
            state_dim,
            self.args.skill_dim,
            self.args.num_species,
            self.args.n_layers,
            self.args.hidden_size,
            device,
            self.args.temperature,
            self.args.state_enc_tau,
            not self.args.no_use_skill_enc)

        if self.args.no_use_skill_enc:
            params = list(self.state_disc._state_encoder.parameters())
        else:
            params = list(self.state_disc._state_encoder.parameters()) + list(self.state_disc._skill_species_encoder.parameters())
        
        if self.args.optim == "adam":
            self.state_disc_optimizer = Adam(params, lr=self.args.lr)
        else:
            self.state_disc_optimizer = RAdam(params, lr=self.args.lr)


    def generate_reward(self, next_obs, skill, species_id, should_log=True):
        with torch.no_grad():
            state_disc_reward = self.state_disc.get_diversity_reward(next_obs, skill, species_id)
            next_obs_latent = self.state_disc.encode_state(next_obs)
            if self.args.novelty_lam != 0.0:
                novelty = self.novelty_buffer.apt_novelty(next_obs_latent)
            else:
                novelty = torch.zeros(state_disc_reward.shape, dtype=torch.float32, device=self.device)

            # print("novelty", novelty)
            diversity_reward = self.args.novelty_lam * novelty + self.args.state_disc_lam * state_disc_reward
            if not self.args.render and self.total_it % 4096 == 0 and should_log:
                print("state_disc_reward", state_disc_reward.view(-1)[:10], state_disc_reward.mean())
                print("novelty", novelty.view(-1)[:10], novelty.mean(), f"RANGE [{novelty.min()}, {novelty.max()}]")
                print("diversity_reward", diversity_reward.view(-1)[:10], diversity_reward.mean())
                print(f"RUNNING MEAN {self.rms.mean} AND STD {self.rms.var ** 0.5}")
                print("ENCODED STATE", self.state_disc.encode_state(
                    torch.tensor([0.5] * self.state_dim, device=self.device)).detach().cpu())

                if self.args.novelty_capacity > 0:
                    print("self.novelty_buffer._idx", self.novelty_buffer._idx)
            return diversity_reward

    def add_state(self, obs):
        if self.args.novelty_lam != 0 and self.args.novelty_capacity > 0:
            obs_latent = self.state_disc.encode_state(
                torch.tensor(obs, device=self.device, dtype=torch.float32)).detach().cpu().numpy()
            self.novelty_buffer.add_state(obs_latent)
    
    def add_states(self, obs):
        """Add multiple states."""
        if self.args.novelty_lam != 0 and self.args.novelty_capacity > 0:
            obs_latent = self.state_disc.encode_state(
                torch.tensor(obs, device=self.device, dtype=torch.float32)).detach().cpu().numpy()
            self.novelty_buffer.add_states(obs_latent)

    def update(self, obs_next, skill, species_id):
        # Update the state discriminator
        state_disc_loss = self.state_disc.compute_loss(obs_next, skill, species_id)
        self.state_disc_optimizer.zero_grad()
        state_disc_loss.backward()
        nn.utils.clip_grad_norm_(list(self.state_disc._state_encoder.parameters()) + list(self.state_disc._skill_species_encoder.parameters()), self.args.max_norm)
        self.state_disc_optimizer.step()
        self.state_disc.update_target()

        self.total_it += 1

        return state_disc_loss.item()

    def save(self, filename):
        model_dict = {
            "state_disc": self.state_disc.state_dict(),
            "state_disc_optimizer": self.state_disc_optimizer.state_dict(),
            "rms": self.rms.state_dict(),
        }

        torch.save(model_dict, filename)


    def load(self, filename):
        model_dict = torch.load(filename)

        self.state_disc.load_state_dict(model_dict["state_disc"])
        self.state_disc_optimizer.load_state_dict(model_dict["state_disc_optimizer"])
        self.rms.load_state_dict(model_dict["rms"])

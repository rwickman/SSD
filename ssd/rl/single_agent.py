import torch
import torch.nn.functional as F
import os
import json

from ssd.rl.td3 import TD3
from ssd.networks.diversity.diversity_reward_net import DiversityRewardNet
from ssd.rl.species_replay_buffer import SpeciesReplayBuffer


class SingleAgent:
    def __init__(self, args, state_dim, action_dim, max_action):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_it = 0

        # Create the species, where each species is a skill-conditioned TD3 instance
        self.td3 = TD3(
                state_dim + self.args.num_species + self.args.skill_dim,
                action_dim,
                max_action,
                self.args.n_layers,
                self.args.hidden_size,
                self.args.gamma,
                self.args.tau,
                self.args.policy_noise,
                self.args.noise_clip,
                self.args.policy_freq
            )

        # Create the diversity reward network for generating the unsupervised reward
        self.diversity_reward_net = DiversityRewardNet(
            self.args, state_dim, self.device).to(self.device)

        # Create the species replay buffer that partitions the buffer by species 
        self.replay_buffer = SpeciesReplayBuffer(
            state_dim, action_dim,  self.args.skill_dim, self.args.num_species, self.args.replay_capacity)
        
        self.train_dict = {
            "state_disc_loss": [],
            "ep_return": [],
        }

        self.train_dict[f"critic_loss_{0}"] = []
        self.train_dict[f"actor_loss_{0}"] = []

    def add_ep_return(self, ep_return):
        self.train_dict["ep_return"].append(ep_return)

    def store(self, species_id, state, action, next_state, reward, skill, done):
        """Add an experience into the replay buffer."""
        self.replay_buffer.add(species_id, state, action, next_state, reward, skill, done)
        self.diversity_reward_net.add_state(next_state)

    def train(self, species_id: int):
        self.train_diversity()
        self.total_it += 1

    def train_diversity(self):
        """Update the diversity networks."""
        if self.replay_buffer.min_size < self.args.learning_starts:
            return
        
        state, action, next_state, _, skill, not_done, species_id = self.replay_buffer.sample(self.args.batch_size)
        state_disc_loss = self.diversity_reward_net.update(
            next_state, skill, species_id)

        # Generate the reward
        species_id_tensor = F.one_hot(species_id.view(-1), self.args.num_species).to(self.device)
        diversity_reward = self.diversity_reward_net.generate_reward(next_state, skill, species_id)

        # Add the skills to the state
        skill_state = torch.cat((state, skill, species_id_tensor), dim=-1)
        skill_next_state = torch.cat((next_state, skill, species_id_tensor), dim=-1)

        # Train the species' network
        critic_loss, actor_loss = self.td3.train(
            skill_state, action, skill_next_state, diversity_reward, not_done)

        # Log the losses
        self.train_dict[f"critic_loss_{0}"].append(critic_loss.item())
        if actor_loss is not None:
            self.train_dict[f"actor_loss_{0}"].append(actor_loss.item())

        # Log the losses
        self.train_dict["state_disc_loss"].append(state_disc_loss)

    def sample_action(self, species_id, state):
        species_id_tensor = F.one_hot(torch.tensor(species_id, dtype=torch.long), self.args.num_species).to(self.device)
        state = torch.cat((torch.tensor(state, dtype=torch.float32, device=self.device), species_id_tensor), dim=-1)
        return self.td3.sample_action(state.view(-1))

    def select_action(self, species_id, state):
        species_id_tensor = F.one_hot(torch.tensor(species_id, dtype=torch.long), self.args.num_species).to(self.device)
        state = torch.cat((torch.tensor(state, dtype=torch.float32, device=self.device), species_id_tensor), dim=-1)
        return self.td3.select_action(state.view(-1))

    def save(self):
        """Save the networks."""
        # Save the species network
        save_file = os.path.join(self.args.model_dir, "td3.pt")
        self.td3.save(save_file)
        
        # Save the diversity reward networks
        diversity_save_file = os.path.join(self.args.model_dir, "diversity_reward_net.pt")
        self.diversity_reward_net.save(diversity_save_file)

        # Save the replay buffer
        self.replay_buffer.save(self.args.model_dir)

        # Save the JSON
        with open(os.path.join(self.args.model_dir, "train_dict.json"), "w") as f:
            json.dump(self.train_dict, f)

    def load(self):
        """Load the networks."""
        # Load the species
        save_file = os.path.join(self.args.model_dir, "td3.pt")
        self.td3.load(save_file)
        
        # Load the diversity reward networks
        diversity_save_file = os.path.join(self.args.model_dir, "diversity_reward_net.pt")
        self.diversity_reward_net.load(diversity_save_file)

        # Load the replay buffer
        if not self.args.render and not self.args.evaluate:
            self.replay_buffer.load(self.args.model_dir)

            # Load the JSON
            with open(os.path.join(self.args.model_dir, "train_dict.json")) as f:
                self.train_dict = json.load(f)

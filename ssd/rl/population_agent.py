import torch
import os
import json

from ssd.rl.td3 import TD3
from ssd.networks.diversity.diversity_reward_net import DiversityRewardNet
from ssd.rl.species_replay_buffer import SpeciesReplayBuffer


class PopulationAgent:
    def __init__(self, args, state_dim, action_dim, action_bounds, behavior_dim):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_it = 0

        # Create the species, where each species is a skill-conditioned TD3 instance
        self.species = [
            TD3(
                state_dim + self.args.skill_dim,
                action_dim,
                action_bounds,
                self.args.n_layers,
                self.args.hidden_size,
                self.args.actor_hidden_size,
                self.args.gamma,
                self.args.tau,
                self.args.policy_noise,
                self.args.expl_noise,
                self.args.noise_clip,
                self.args.policy_freq,
                self.args.lr,
                self.args.optim,
                self.args.max_norm,
            )
            for _ in range(self.args.num_species)
        ]

        # Create the diversity reward network for generating the unsupervised reward
        self.diversity_reward_net = DiversityRewardNet(
            self.args,
            behavior_dim if self.args.use_behavior_disc else state_dim,
            self.device
        ).to(self.device)

        # Create the species replay buffer that partitions the buffer by species 
        self.replay_buffer = SpeciesReplayBuffer(
            state_dim, action_dim,  self.args.skill_dim, self.args.num_species, behavior_dim, self.args.replay_capacity)
        
        self.train_dict = {
            "state_disc_loss": [],
            "ep_return": [],
        }

        # Add an entry for each species
        for i in range(self.args.num_species):
            self.train_dict[f"critic_loss_{i}"] = []
            self.train_dict[f"actor_loss_{i}"] = []

    def add_ep_return(self, ep_return):
        self.train_dict["ep_return"].append(ep_return)

    def store(self, species_id, state, action, next_state, reward, skill, behavior, done):
        """Add an experience into the replay buffer."""
        self.replay_buffer.add(
            species_id, state, action, next_state, reward, skill, behavior, done)

    def store_behaviors(self, behavior):
        self.diversity_reward_net.add_states(behavior)        

    def get_diversity(self, behavior, species_id, skill):
        with torch.no_grad():
            behavior =  torch.tensor(behavior, device=self.device, dtype=torch.float32)
            skill = torch.tensor(skill, device=self.device, dtype=torch.float32)
            species_id_tensor = torch.tensor(
                species_id,
                dtype=torch.long,
                device=self.device)

            diversity_reward = self.diversity_reward_net.generate_reward(behavior, skill, species_id_tensor, should_log=False)
            return diversity_reward

    def train(self, species_id: int):
        if self.replay_buffer.min_size < self.args.learning_starts:
            return
        self.train_diversity()
        cur_species_id = self.total_it % self.args.num_species 
        self.train_species(cur_species_id)
        self.total_it += 1

    def train_diversity(self):
        """Update the diversity networks."""
        if self.replay_buffer.min_size < self.args.learning_starts:
            return
        
        state, action, next_state, _, skill, behavior, _, species_id = self.replay_buffer.sample(self.args.batch_size)
        state_disc_loss = self.diversity_reward_net.update(
            behavior if self.args.use_behavior_disc else next_state, skill, species_id)

        # Log the losses
        self.train_dict["state_disc_loss"].append(state_disc_loss)

    def train_species(self, species_id: int):
        """Update a species' TD3 networks."""
        if self.replay_buffer.min_size < self.args.learning_starts:
            return

        # Sample from the species buffer
        state, action, next_state, env_reward, skill, behavior, not_done = self.replay_buffer.sample_species(
            self.args.batch_size, species_id)

        # Generate the reward
        species_id_tensor = torch.tensor([species_id], dtype=torch.long, device=self.device).repeat(state.shape[0])
        diversity_reward = self.diversity_reward_net.generate_reward(
            behavior if self.args.use_behavior_disc else next_state,
            skill,
            species_id_tensor
        )

        reward = self.args.env_reward_lam * env_reward + diversity_reward
        if self.total_it % 2048 == 0:
            print("diversity_reward", diversity_reward.view(-1)[:10], diversity_reward.mean())
            print("env_reward", env_reward.view(-1)[:10], env_reward.mean())
            print("reward", reward.view(-1)[:10], reward.mean())

        # Add the skills to the state
        skill_state = torch.cat((state, skill), dim=-1)
        skill_next_state = torch.cat((next_state, skill), dim=-1)

        # Train the species' networks
        critic_loss, actor_loss = self.species[species_id].train(
            skill_state, action, skill_next_state, reward, not_done)

        # Log the losses
        self.train_dict[f"critic_loss_{species_id}"].append(critic_loss.item())
        if actor_loss is not None:
            self.train_dict[f"actor_loss_{species_id}"].append(actor_loss.item())

    def sample_action(self, species_id, state):
        return self.species[species_id].sample_action(state)

    def select_action(self, species_id, state):
        return self.species[species_id].select_action(state)

    def save(self):
        """Save the networks."""
        # Save the species
        for i in range(self.args.num_species):
            save_file = os.path.join(self.args.model_dir, "species", f"species_{i}_td3.pt")
            self.species[i].save(save_file)
        
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
        for i in range(self.args.num_species):
            save_file = os.path.join(self.args.model_dir, "species", f"species_{i}_td3.pt")
            self.species[i].load(save_file)
        
        # Load the diversity reward networks
        diversity_save_file = os.path.join(self.args.model_dir, "diversity_reward_net.pt")
        self.diversity_reward_net.load(diversity_save_file)

        # Load the replay buffer
        if not self.args.render and not self.args.evaluate:
            self.replay_buffer.load(self.args.model_dir)

            # Load the JSON
            with open(os.path.join(self.args.model_dir, "train_dict.json")) as f:
                self.train_dict = json.load(f)

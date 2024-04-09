import numpy as np
import torch
import os

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, skill_dim, behavior_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.skill_dim = skill_dim
        self.ptr = 0
        self.size = 0
        

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.skill = np.zeros((max_size, skill_dim))
        self.behavior = np.zeros((max_size, behavior_dim))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, skill, behavior, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.skill[self.ptr] = skill
        self.behavior[self.ptr] = behavior
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return [
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.skill[ind],
            self.behavior[ind],
            self.not_done[ind],
        ]

    def get_save_dict(self):
        d = {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "skill": self.skill,
            "behavior": self.behavior,
            "not_done": self.not_done,
            "ptr": self.ptr,
            "size": self.size,
        }
        if self.skill_dim > 0:
            d["skill"] = self.skill

        return d


    def load(self, data):
        """Load the replay buffer for a particular species."""
        self.state = data["state"]
        self.action = data["action"]
        self.reward = data["reward"]
        if self.skill_dim > 0:
            self.skill = data["skill"]
        self.behavior = data["behavior"]
        self.next_state = data["next_state"]
        self.not_done = data["not_done"]
        self.ptr = int(data["ptr"])
        self.size = int(data["size"])


class SpeciesReplayBuffer:
    def __init__(self, state_dim, action_dim, skill_dim, num_species, behavior_dim, max_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_species = num_species

        # Create a replay buffer for each species
        species_buffer_maze_size = max_size//self.num_species # Partition replay buffer into equally-sized chunks
        self._buffers = [
            ReplayBuffer(state_dim, action_dim, skill_dim, behavior_dim, species_buffer_maze_size)
            for _ in range(self.num_species)
        ]

    @property
    def size(self):
        return sum(b.size for b in self._buffers)
    
    @property
    def min_size(self):
        """The minimum species buffer size ."""
        return min(b.size for b in self._buffers)

    def add(self, species_id, state, action, next_state, reward,  skill,  behavior, done):
        """Add an experience to a species buffer."""
        self._buffers[species_id].add(state, action, next_state, reward, skill, behavior, done)

    def sample(self, batch_size: int):
        """Equally sample experiences from each species"""
        species_ids = list(range(self.num_species))

        # Number of examples per species to sample
        ex_per_species = batch_size // len(species_ids)

        # states, actions, next_states, rewards, skills, behaviors, not_dones, species_ids
        batch = [[], [], [], [], [], [], [], []]

        # Sample all the species
        for i in species_ids:
            if self._buffers[i].size >= ex_per_species:
                species_batch = self._buffers[i].sample(ex_per_species)
                
                for j in range(len(species_batch)):
                    batch[j].append(species_batch[j])
                
                # Add the species
                batch[-1] += [i] * ex_per_species

        # Join them into a single batch
        for i in range(len(batch) - 1):
            batch[i] = torch.FloatTensor(
                np.concatenate(batch[i])).to(self.device)

        batch[-1] = torch.LongTensor(batch[-1]).to(self.device)

        return batch

    def sample_species(self, batch_size: int, species_id: int):
        """Sample directly from a species buffer."""
        batch = self._buffers[species_id].sample(batch_size)
        for i in range(len(batch)):
            batch[i] = torch.tensor(
                batch[i], dtype=torch.float32, device=self.device)
        
        return batch

    def save(self, save_dir: str):
        """Save the replay buffer."""
        out_file = os.path.join(save_dir, "replay_buffer.npz")
        save_dicts = {}
        for i in range(self.num_species):
            save_dicts[str(i)] = self._buffers[i].get_save_dict()

        np.savez(out_file, **save_dicts)

    def load(self, save_dir: str):
        """Load the replay buffer."""
        out_file = os.path.join(save_dir, "replay_buffer.npz")
        if os.path.exists(out_file):
            data = np.load(out_file, allow_pickle=True)
            for i in range(len(data)):
                species_data = data[str(i)].item()
                self._buffers[i].load(species_data)
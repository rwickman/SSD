import numpy as np
import random
from tqdm import tqdm

def sample_skill(skill_dim):
    skill = np.random.uniform(low=-1, high=1.0, size=(skill_dim, ))
    # skill = skill / np.linalg.norm(skill)
    return skill

def sample_species(num_species):
    return random.randint(0, num_species-1)

class EnvRunner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
    
    def run(self, agent, species_id=None, skill=None, train=False):
        ep_skills = [] # Skills used in the episode
        ep_species = []
 
        obs, _ = self.env.reset()
        if skill is None:
            skill = sample_skill(self.args.skill_dim)
            ep_skills.append(skill)
        if species_id is None:    
            species_id = sample_species(self.args.num_species)
            
        ep_skills.append(skill)
        total_reward = 0
        
        for t in range(self.args.max_episode_len):
            if t != 0 and t % self.args.resample_skill_freq == 0:
                skill = sample_skill(self.args.skill_dim)
                ep_skills.append(skill)
            if t != 0 and t % self.args.resample_species_freq == 0:
                species_id = sample_species(self.args.num_species)
                ep_species.append(species_id)

            state = np.concatenate((obs, skill))
            if train:
                if agent.replay_buffer.min_size < self.args.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = agent.sample_action(species_id, state)
            else:
                action = agent.select_action(species_id, state)
            
            next_obs, reward, done, terminated, _ = self.env.step(action)
            total_reward += reward
            if train:
                agent.store(species_id, obs, action, next_obs, reward, skill, done or terminated)

            obs = next_obs

            if train and t % self.args.update_freq == 0:
                agent.train(species_id)
            if done or terminated:
                break

        if self.args.render:
            print("Episode Reward: ", total_reward)
        
        ep_info = {
            "species_id": species_id,
            "ep_reward": total_reward,
            "steps": t + 1,
            "skills": ep_skills,
            "species": ep_species,
        }
        return t + 1, ep_info

    def train(self, agent):
        elapsed_timesteps = 0
        pbar = tqdm(total=self.args.timesteps)
        ep_idx = 0
        last_save = 0
        while elapsed_timesteps < self.args.timesteps:
            species_id = ep_idx % self.args.num_species

            episode_steps, ep_info = self.run(agent, species_id, train=not self.args.evaluate and not self.args.render)
            elapsed_timesteps += episode_steps
            last_save += episode_steps
            agent.add_ep_return(ep_info["ep_reward"])
            pbar.update(episode_steps)

            if last_save >= self.args.save_iter and not self.args.evaluate and not self.args.render:
                print(f"SAVING {last_save}")
                agent.save()
                last_save = 0
            
            ep_idx += 1

        if not not self.args.evaluate and not self.args.render:
            agent.save()
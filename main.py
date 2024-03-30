import argparse
import gym
import os, json, time
import maze_envs
from tqdm import tqdm
from maze_envs.wrappers.state_frequency import StateFrequencyWrapper

from ssd.args import update_parser
from ssd.util import set_seed
from ssd.rl.env_runner import EnvRunner
from ssd.rl.population_agent import PopulationAgent
from ssd.rl.single_agent import SingleAgent

def main(args):
    set_seed(args.seed)
    if not os.path.exists(args.model_dir):
        os.makedirs(os.path.join(args.model_dir, "species"))
    elif not args.load:
        # Prevent saving over an existing model
        raise Exception(f"{args.model_dir} already exists!")

    # Create the environment
    if "maze" in args.env:
        gym_env = gym.make(
            "maze_envs/ContinuousGridWorld-v0",
            grid_img_path=args.maze_map,
            init_pos=args.init_pos,
            render_mode="human" if args.render else None)
    else:
        gym_env = gym.make(args.env, render_mode="human" if args.render else None)

    # Create the Environment Runner
    env_runner = EnvRunner(args, gym_env)

    # Create the agent
    if args.single_agent:
        agent = SingleAgent(
            args,
            gym_env.observation_space.shape[0],
            gym_env.action_space.shape[0],
            gym_env.action_space.high[0]
        )
    else:
        agent = PopulationAgent(
            args,
            gym_env.observation_space.shape[0],
            gym_env.action_space.shape[0],
            gym_env.action_space.high[0]
        )

    if args.load:
        agent.load()
    
    if not args.evaluate:
        # Train the models
        env_runner.train(agent)
    elif "maze" in args.env:
        # Evaluate the models
        env = StateFrequencyWrapper(gym_env)
        env_runner.env = env
        if args.results_dir == "":
            raise Exception("Please provide a location to save the results.")
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        for i in tqdm(range(args.evals)):
            species_id = i % args.num_species
            env.set_skill(species_id)
            env_runner.run(agent, species_id)
            if i % 64 == 0:
                state_coverage = env.get_state_coverage()
                print("state_coverage", state_coverage)

        state_coverage = env.get_state_coverage()
        state_skill_frequency = env.get_state_skill_frequency()
        with open(os.path.join(args.results_dir, f"state_skill_frequency_seed_{args.seed}.json"), "w") as f:
            json.dump(state_skill_frequency, f)
        with open(os.path.join(args.results_dir, f"state_coverage_seed_{args.seed}.json"), "w") as f:
            json.dump({"state_coverage": state_coverage}, f)

        print("state_coverage", state_coverage)
    else:
        max_reward = -1e6
        for i in tqdm(range(args.evals)):
            species_id = i % args.num_species
            _, ep_info = env_runner.run(agent, species_id)
            max_reward = max(max_reward, ep_info["ep_reward"])
            if i % 64 == 0:
                print(max_reward)

        print("max_reward", max_reward)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    update_parser(parser)
    parser.add_argument("--results_dir", default="",
        help="Directory to save the results.")
    parser.add_argument("--evals", type=int, default=1024,
        help="Number of times to evaluate the environment.")

    args = parser.parse_args()
    main(args)
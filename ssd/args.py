
def update_parser(parser):
    parser.add_argument("--model_dir", required=True,
        help="Directory to save the modelsa and results.")
    parser.add_argument("--load", action="store_true",
        help="Load the models.")
    parser.add_argument("--render", action="store_true",
        help="Render frames.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the model.")
    parser.add_argument("--save_iter", type=int, default=10000,
                        help="How many timesteps elapsed before saving the networks.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed to run models on.")
    parser.add_argument("--single_agent", action="store_true",
                        help="Use a single agent instead of the population of agents.")


    env_args = parser.add_argument_group("Environment")
    env_args.add_argument("--env", required=True,
        help="Environment to test.")
    env_args.add_argument("--timesteps", type=int, default=1000000,
        help="Number of timesteps to run.")
    env_args.add_argument("--max_episode_len", type=int, default=200,
        help="Maximum number of steps in an episode.")

    pop_args = parser.add_argument_group("Population")
    pop_args.add_argument("--num_species", type=int, default=8,
        help="Number of species (i.e., skill-conditioned)")
    pop_args.add_argument("--skill_dim", type=int, default=2,
        help="Dimension of the skill.")
    pop_args.add_argument("--resample_skill_freq", type=int, default=50,
        help="How often to resample a skill.")
    pop_args.add_argument("--resample_species_freq", type=int, default=50,
        help="How often to resample a species.")

    net_args = parser.add_argument_group("Network")
    net_args.add_argument("--lr", type=float, default=3e-4,
        help="Learning rate for actor and critic networks.")
    net_args.add_argument("--gamma", type=float, default=0.99,
        help="Gamma used for learning rate discount.")
    net_args.add_argument("--tau", type=float, default=0.005,
        help="Value used for interpolating target policy.")
    net_args.add_argument("--batch_size", type=int, default=256,
        help="Batch size for updating networks.")
    net_args.add_argument("--replay_capacity", type=int, default=1048576,
        help="Maximum size of replay memory.")
    net_args.add_argument("--n_layers", type=int, default=2,
        help="Number of hidden layers.")
    net_args.add_argument("--hidden_size", type=int, default=256,
        help="Hidden size of network.")
    net_args.add_argument("--policy_noise", type=float, default=0.2,
        help="Noise used in policy prediction.")
    net_args.add_argument("--noise_clip", type=float, default=0.5,
        help="Noise used in policy prediction.")
    net_args.add_argument("--policy_freq", type=int, default=2,
        help="How often to update the policy w.r.t. the critic.")
    net_args.add_argument("--learning_starts", type=int, default=1000,
        help="Number of timesteps to elapse before training.")
    net_args.add_argument("--update_freq", type=int, default=2,
        help="How often to update the models.")

    reward_args = parser.add_argument_group("Unsupervised Reward")
    reward_args.add_argument("--state_disc_lam", type=float, default=1.0,
        help="Reward scaling for state discriminator reward.")
    reward_args.add_argument("--env_reward_lam", type=float, default=0.0,
        help="Reward scaling for environment reward.")
    reward_args.add_argument("--novelty_lam", type=float, default=1.0,
        help="Reward scaling for novelty reward.")
    reward_args.add_argument("--temperature", type=float, default=0.5,
        help="Temperature to use for contrastive learning.")
    reward_args.add_argument("--knn_k", type=int, default=10,
        help="Number of nearest-neighbors to take the mean for the novelty of a sample.")
    reward_args.add_argument("--knn_rms", action="store_true",
        help="Normalize KNN distances by running mean.")
    reward_args.add_argument("--knn_clip", type=float, default=0.0001,
        help="Lipschitz constant for the spectral normalization.")
    reward_args.add_argument("--novelty_capacity", type=int, default=6400,
        help="Capacity of the novelty buffer.")
    reward_args.add_argument("--max_novelty", type=int, default=10,
        help="Clip the novelty at a max value.")
    reward_args.add_argument("--state_enc_tau", type=float, default=0.001,
        help="Value used for interpolating target policy.")
    reward_args.add_argument("--no_use_skill_enc", action="store_true",
        help="Disable the skill, species encoder.")


    maze_args = parser.add_argument_group("Maze")
    maze_args.add_argument("--init_pos", nargs=2, type=float, default=[2.0, 48.0],
        help="Initial position for the maze map.")
    maze_args.add_argument("--maze_map", default="", 
        help="Location of maze map if running the maze-env.")

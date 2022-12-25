# default experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-non-stationary-vel',
    env_params=dict(
        n_train_tasks=100,  # number train tasks
        n_eval_tasks=30,  # number evaluation tasks tasks
        n_grid_tasks=8,
        use_normalized_env=True,  # if normalized env should be used
        scripted_policy=False,  # if true, a scripted oracle policy will be used for data collection, only supported for metaworld
    ),
    path_to_weights=None, # path to pre-trained weights to load into networks
    train_or_showcase='train',  # 'train' for train new policy, 'showcase' to load trained policy and showcase
    showcase_itr=None,  # training epoch from which to use weights of policy to showcase
    util_params=dict(
        base_log_dir='output',  # name of output directory
        use_gpu=True,  # set True if GPU available and should be used
        use_multiprocessing=False,  # set True if data collection should be parallelized across CPUs
        num_workers=1,  # number of CPU workers for data collection
        gpu_id=0,  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        plot=False,  # plot figures of progress for reconstruction and policy training
        tb_log_interval=100  # interval in which to log to tensorboard. Set to -1 to disable logging
    ),

    algo_params=dict(
        use_relabeler=True, # if data should be relabeled
        use_combination_trainer=False,  # if combination trainer (gradients from Decoder and SAC should be used, currently broken
        use_data_normalization=True,  # if data become normalized, set in correspondence to use_combination_trainer
        use_parametrized_alpha=False,  # alpha conditioned on z
        encoding_mode="transitionIndividualY",  # choose encoder design: trajectory, transitionSharedY, transitionIndividualY
        merge_mode="add",  # if transitionSharedY: how to merge y infos: "add", "add_softmax", "multiply", "linear", "mlp"
        use_fixed_seeding=True,  # seeding, make comparison more robust
        seed=0,  # seed for torch and numpy
        batch_size_reconstruction=4096,  # batch size reconstruction trainer
        batch_size_combination=256,  # batch size combination trainer
        batch_size_policy=256,  # batch size policy trainer
        batch_size_relabel=1024,  # batch size relabeler
        batch_size_rollout=512,
        time_steps=30,  # timesteps before current to be considered for determine current task
        latent_size=1,  # dimension of the latent context vector z. For use_latent_grid=True, this represents the grid dimension and mus be either 1 or 2.
        sac_layer_size=300,  # layer size for SAC networks, value 300 taken from PEARL
        max_replay_buffer_size=10000000,  # write as integer!
        data_usage_reconstruction=None,  # mode of data prioritization for reconstruction trainer, values: None, 'cut', 'linear, 'tree_sampling'
        data_usage_sac=None,  # mode of data prioritization for policy trainer, values: None, 'cut', 'linear, 'tree_sampling'
        num_last_samples=10000000,  # if data_usage_sac == 'cut, number of previous samples to be used
        permute_samples=False,  # if order of samples from previous timesteps should be permuted (avoid learning by heart)
        num_train_epochs=250,  # number of overall training epochs
        snapshot_gap=20,  # interval to store weights of all networks like encoder, decoder, policy etc.
        num_reconstruction_steps=5000,  # number of training steps in reconstruction trainer per training epoch
        num_policy_steps=3000,  # number of training steps in policy trainer per training epoch
        num_train_tasks_per_episode=100,  # number of training tasks from which data is collected per training epoch
        num_transitions_initial=200,  # number of overall transitions per task while initial data collection
        num_transitions_per_episode=200,  # number of overall transitions per task while each epoch's data collection
        num_eval_trajectories=3,  # number evaluation trajectories per test task
        showcase_every=0,  # interval between training epochs in which trained policy is showcased
        num_showcase_deterministic=1,  # number showcase evaluation trajectories per test task, encoder deterministic
        num_showcase_non_deterministic=1,  # number showcase evaluation trajectories per test task, encoder deterministic
        max_path_length=200,  # maximum number of transitions per episode in the environment
        target_entropy_factor=1.0,  # target entropy from SAC
        sac_alpha=1.0,  # fixed alpha value in SAC when not using automatic entropy tuning,
        automatic_entropy_tuning=True,
        latent_probabilities=False,  # whether to pass probability of each y along with mean and variance of the z for each y instead of just the sampled overall z to the policy
        exploration_elbo_reward_factor=0.0,  # weight for the elbo_reward. If 0.0, elbo_rewards won't be calculated at all
        elbo_reward_decay_rate=0.95,  # elbo reward is multiplied by (elbo_reward_decay_rate ^ epoch) * exploration_elbo_reward_factor
        latent_channels=1,  # Number of tasks for the latent_grid setting. For each task, one channel/grid with num_classes clusters is created
        use_latent_grid=False,  # Whether to represent the latent distribution of the latent space as a grid.
        policy_equivariance="none",  # the equivariance to use for the policy [toy1D, toy2D, ...]
        latent_grid_resolution=32,  # If latent_grid=True, this represents the grid's number of cells in one dimension. For latent_size=2, the grid size would be latent_grid_resolution x latent_grid_resolution.
        latent_grid_range=6,  # The latent grid will go from -latent_grid_range to + latent_grid_range in each direction
        restrict_latent_mean=False,  # If True, the encoder mean output will be forced to the range [-latent_grid_range, latent_grid_range] using tanh activation. No activation function for means otherwise.
        all_channels_to_policy=True, # Whether to pass the likelihood of all channels along with the encoding for each channel to policy instead of just the most likely channel along with its encoding. Required to learn Bayes-optimal policy
        random_policy=False,  # overwrite policy actions with random behavior for debugging
        task_specs_to_policy=False  # pass true task specifications instead of encoder predictions to policy for debugging
    ),

    reconstruction_params=dict(
        use_state_diff=False,  # determines if decoder uses state or state difference as target
        component_constraint_learning=False,  # enables providing base class information to the class encoder
        prior_mode='fixedOnY',  # options: 'fixedOnY' and 'network, determine if prior comes from a linear layer or is fixed on y
        prior_sigma=0.5,  # simga on prior when using fixedOnY prior
        num_classes=1,  # number of base classes in the class encoder
        lr_encoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,
        lr_decoder=3e-4,  # learning rate decoder (ADAM) 3e-4 when combine with combination trainer,
        alpha_kl_z=1e-3,  # weighting factor KL loss of z distribution
        beta_kl_y=1e-3,  # # weighting factor KL loss of y distribution
        net_complex_enc_dec=10.0,  # determines overall net complextity in encoder and decoder
        decoder_equivariance="none",  # if not none, the decoder will be equivariant as specified by the given experiment
        encoder_equivariance="none",  # if not none, the encoder will be equivariant as specified by the given experiment. Only for GRU encoder.
        reconstruct_all_timesteps=False,  # If True, the encoding of a trajectory is used to predict next state/reward for each step in the trajectory instead of just the last one. This prevents the encoding to represent the reward prediction directly.
        factor_qf_loss=1.0,  # weighting of state and reward loss compared to Qfunction in combination trainer
        train_val_percent=0.8,  # percentage of train samples vs. validation samples
        eval_interval=50,  # interval for evaluation with validation data and possible early stopping
        early_stopping_threshold=500,  # minimal epochs before early stopping after new minimum was found
        temp_folder='.temp_cemrl',  # helper folder for storing encoder and decoder weights while training
        use_state_decoder=True  # state decoder can be disabled for debugging as the task only depends on the reward functions in our settings
    ),


    ablation_params = dict(
        encoding_noise=0.0,  # Weight of additional gaussian noise to be added to the encoding to prove that the encoding is necessary
        constant_encoding=False,  # Whether to feed constant 0 instead of the encoding to the decoder to prove the encoding's necessity
    )
)

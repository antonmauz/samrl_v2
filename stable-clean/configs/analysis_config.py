analysis_config = dict(
    env_name='placeholder',
    # path_to_weights="output/toy-goal/2022_01_18_18_23_35",
    # path_to_weights=["output/cheetah-stationary-multi-task/2021_12_20_08_22_41"],# "output/cheetah-stationary-multi-task/2021_12_16_16_18_52"],  # CONFIGURE! path to experiment folder
    # path_to_weights="../cemrl_server_data/cheetah-stationary-multi-task/2021_09_23_15_37_36_dense_elbo_decay_098",  # CONFIGURE! path to experiment folder
    # path_to_weights=["../cemrl_server_data/cemrl_ablation/cheetah-stationary-multi-task/2022_02_02_15_47_26", "../cemrl_server_data/cemrl_ablation/cheetah-stationary-multi-task/2022_02_02_15_46_40"], #, "../cemrl_server_data/cemrl_ablation/cheetah-stationary-multi-task/2022_02_02_15_40_52"],
    path_to_weights="../cemrl_server_data/cemrl_ablation/cheetah-stationary-multi-task/2020_12_02_23_41_18",
    train_or_showcase='showcase',  # 'showcase' to load trained policy and showcase
    showcase_itr=100,
    #showcase_itr=[1, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 750, 1000, 1250],  # CONFIGURE! epoch for which analysis is performed. Can also be list. If none, most recent epoch is used.
    util_params=dict(
        base_log_dir='output',  # name of output directory
        use_gpu=True,  # set True if GPU available and should be used
        use_multiprocessing=False,  # set True if data collection should be parallelized across CPUs
        num_workers=8,  # number of CPU workers for data collection
        gpu_id=0,  # number of GPU if machine with multiple GPUs
        debug=False,  # debugging triggers printing and writes logs to debug directory
        plot=False  # plot figures of progress for reconstruction and policy training
    ),
    analysis_params=dict(
        example_case=4,  # CONFIGURE! choose a test task
        log_and_plot_progress=False,  # CONFIGURE! If True: experiment will be logged to the experiment_database.json and plotted, If already logged: plot only
        save=True,  # CONFIGURE! If True: plots of following options will be saved to the experiment folder
        visualize_run=False,  # CONFIGURE! If True: learned policy of the showcase_itr will be played on example_case
        plot_time_response=True,  # CONFIGURE! If True: plot of time response
        plot_velocity_multi=False,  # CONFIGURE! If True: (only for velocity envs) plots time responses for multiple tasks
        plot_encoding=False,  # CONFIGURE! If True: plots encoding for showcase_itr
        produce_video=False,  # CONFIGURE! If True: produces videos of learned policy for all test tasks
        wandb=False,  # whether produced output should be logged to wandb
        manipulate_time_steps=False,  # CONFIGURE! If True: uses time_steps different recent context, as specified below
        time_steps=10,  # CONFIGURE! time_steps for recent context if manipulated is enabled
        manipulate_change_trigger=False,  # CONFIGURE! for non-stationary envs: change the trigger for task changes
        change_params=dict(  # CONFIGURE! If manipulation enabled: set new trigger specification
            change_mode="time",
            change_prob=1.0,
            change_steps=100,
        ),
        manipulate_max_path_length=False,  # CONFIGURE! change max_path_length to value specified below
        max_path_length=800,
        manipulate_test_task_number=True,  # CONFIGURE! change test_task_number to value specified below
        test_task_number=6,
    ),
    env_params=dict(
        scripted_policy=False,
    ),
    algo_params=dict(
        merge_mode="mlp",
        use_fixed_seeding=True,
        seed=0,
    ),
    reconstruction_params=dict()
)

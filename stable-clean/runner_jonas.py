# Continuous Environment Meta Reinforcement Learning (CEMRL)

import os
import numpy as np
import click
import json

from wandb import Config

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from analysis import logger
from symmetrizer.jonas.symmetric_networks_metaworld_v1 import MetaWorldv1Policy, MetaWorldv1QNet
from symmetrizer.jonas.symmetric_networks_toy1d import Toy1DPolicy, Toy1DQNet
from symmetrizer.jonas.symmetric_networks_toy2d import Toy2DPolicy, Toy2DQNet
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy, MlpPolicyNetwork
from rlkit.torch.networks import Mlp, FlattenMlp
from rlkit.launchers.launcher_util import setup_logger, create_log_dir
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from cerml.encoder_decoder_networks import PriorPz, EncoderMixtureModelTrajectory, EncoderMixtureModelTransitionSharedY, \
    EncoderMixtureModelTransitionIndividualY, DecoderMDP, EncoderMixtureModelGRU, MultiChannelDecoderMDP
from cerml.sac import PolicyTrainer
from cerml.stacked_replay_buffer import StackedReplayBuffer
from cerml.reconstruction_trainer import ReconstructionTrainer
from cerml.combination_trainer import CombinationTrainer
from cerml.rollout_worker import RolloutCoordinator
from cerml.agent import CEMRLAgent, ScriptedPolicyAgent
from cerml.relabeler import Relabeler
from cerml.cemrl_algorithm import CEMRLAlgorithm


def experiment(variant):
    # create logging directory
    encoding_save_epochs = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 225, 250, 275, 300, 325, 350,
                            375, 400, 425, 450, 475, 500, 600, 750, 1000, 1250, 1500, 1750, 2000, 3000, 5000, 10000,
                            15000, 20000]

    if logger.USE_WANDB:
        # Create log dir earlier to include it in config
        log_dir = create_log_dir(variant['env_name'], exp_id=None, seed=0,
                                 base_log_dir=variant['util_params']['base_log_dir'])
        exp_name = log_dir.split('/')[-1]
        variant["exp_name"] = exp_name

        wandb_variant = deep_dict_to_point(variant)
        if variant['util_params']['resume_wandb']:
            wandb.init(id=variant['wandb_run_id'], resume="must")
        else:
            wandb.init(project="cemrl", config=wandb_variant, dir=variant['util_params']['base_log_dir'])
            variant['wandb_run_id'] = wandb.run.id
        wandb.define_metric("rl/step")
        wandb.define_metric("rl/*", step_metric="rl/step", summary="last")
        wandb.define_metric("reconst/step")
        wandb.define_metric("reconst/*", step_metric="reconst/step", summary="last")
        wandb.define_metric("epoch",  summary="last")
        wandb.define_metric("env_step_total", summary="last")
        wandb.define_metric("evaluation.*", step_metric="env_steps_total", summary="last")
        wandb.define_metric("encoding", step_metric="epoch")
        fix_dict_in_wandb_config()
        variant = wandb.config  # changes config in wandb sweeps
        experiment_log_dir = setup_logger(variant['env_name'], variant=variant.__dict__["_items"], exp_id=None,
                                          base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                          snapshot_gap=variant['algo_params']['snapshot_gap'],
                                          log_dir=log_dir,
                                          snapshot_points=encoding_save_epochs)
    else:
        experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=None,
                                          base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                          snapshot_gap=variant['algo_params']['snapshot_gap'],
                                          snapshot_points=encoding_save_epochs)

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)
    if (variant['algo_params']['use_fixed_seeding']):
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])

    if logger.USE_TENSORBOARD:
        logger.TENSORBOARD_LOGGER = SummaryWriter(log_dir=os.path.join(experiment_log_dir, 'tensorboard'))
    logger.LOG_INTERVAL = variant['util_params']['tb_log_interval']
    logger.TRAINING_LOG_STEP = 0
    # tb_logger.AUGMENTATION_LOG_STEP = 0
    logger.TI_LOG_STEP = 0
    # tb_logger.DEBUG_LOG_STEP = 0

    algorithm, _, _, _, _, _, _, _ = setup_algorithm(variant, experiment_log_dir, False)

    # run the algorithm
    algorithm.train()

def setup_algorithm(variant, experiment_log_dir, is_analysis):
    # create multi-task environment and sample tasks
    env = ENVS[variant['env_name']](**variant['env_params'])
    if variant['env_params']['use_normalized_env']:
        env = NormalizedBoxEnv(env)
    if variant['train_or_showcase'] == 'showcase':
        env = CameraWrapper(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    tasks = list(range(len(env.tasks)))
    # train_tasks = tasks[:variant['env_params']['n_train_tasks']]
    train_tasks = list(range(len(env.train_tasks)))
    test_tasks = tasks[
                 variant['env_params']['n_train_tasks']:variant['env_params']['n_train_tasks'] + variant['env_params'][
                     'n_eval_tasks']]
    grid_tasks = tasks[variant['env_params']['n_train_tasks'] + variant['env_params']['n_eval_tasks']:]

    # instantiate networks
    net_complex_enc_dec = variant['reconstruction_params']['net_complex_enc_dec']
    latent_dim = variant['algo_params']['latent_size']
    time_steps = variant['algo_params']['time_steps']
    num_classes = variant['reconstruction_params']['num_classes']
    use_latent_grid = variant['algo_params']['use_latent_grid']
    latent_probabilities = variant['algo_params']['latent_probabilities']
    # if use_latent_grid and not latent_probabilities:
    #     # Potentially implement this for ablation
    #     raise ValueError("Discretizing a sampled latent variable is currently not implemented. The latent grid "
    #                      "represents a probability distribution")
    num_channels = variant['algo_params']['latent_channels']
    task_indicator_dim = (latent_dim * 2 + 1) * (num_classes * num_channels)\
        if latent_probabilities else latent_dim

    # encoder used: single transitions or trajectories
    if variant['algo_params']['encoding_mode'] == 'transitionSharedY':
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTransitionSharedY
    elif variant['algo_params']['encoding_mode'] == 'transitionIndividualY':
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTransitionIndividualY
    elif variant['algo_params']['encoding_mode'] == 'trajectory':
        encoder_input_dim = time_steps * (obs_dim + action_dim + reward_dim + obs_dim)
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelTrajectory
    elif variant['algo_params']['encoding_mode'] == 'GRU':
        if variant['algo_params']['permute_samples']:
            print("#" * 76)
            print("# WARNING: Are you sure you meant to use GRU encoder with permute_samples? #")
            print("#" * 76)
        encoder_input_dim = obs_dim + action_dim + reward_dim + obs_dim
        shared_dim = int(encoder_input_dim * net_complex_enc_dec)  # dimension of shared encoder output
        encoder_model = EncoderMixtureModelGRU
    else:
        raise NotImplementedError

    encoder = encoder_model(
        shared_dim,
        encoder_input_dim,
        latent_dim,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        num_channels,
        time_steps,
        variant['algo_params']['merge_mode'],
        variant['algo_params']['use_latent_grid'],
        variant['algo_params']['latent_grid_resolution'],
        variant['algo_params']['latent_grid_range'],
        variant['algo_params']['restrict_latent_mean'],
        variant['ablation_params']['encoding_noise'],
        variant['ablation_params']['constant_encoding'],
        variant['reconstruction_params']['encoder_equivariance']
    )

    # TODO the MultiChannel setup could be used decoupled from the GRU but I would need to implement multi-channel for
    #  the other encoders
    decoder_class = MultiChannelDecoderMDP if variant['algo_params']['encoding_mode'] == 'GRU' else DecoderMDP
    decoder = decoder_class(
        action_dim,
        obs_dim,
        reward_dim,
        latent_dim,
        num_channels,
        num_classes,
        net_complex_enc_dec,
        variant['env_params']['state_reconstruction_clip'],
        variant['reconstruction_params']['decoder_equivariance']
    )

    if variant['reconstruction_params']['encoder_equivariance'] in ["toy1D", "MetaWorldv1"]:
        class_divisor = 2
    elif variant['reconstruction_params']['encoder_equivariance'] == "toy2D":
        class_divisor = 4
    elif variant['reconstruction_params']['encoder_equivariance'] == "none":
        class_divisor = 1
    else:
        raise ValueError("Unknown encoder equivariance: " +
                         str(variant['reconstruction_params']['encoder_equivariance']))

    if num_classes % class_divisor != 0:
        raise ValueError("For an equivariant encoder, the number of classes (" + str(num_classes) + ") must be "
                         "divisible by the amount of equivariance groups (" + str(class_divisor) + ") in order to match each class "
                         "with equivariant corresponding classes!")
    prior_pz = PriorPz(num_classes // class_divisor, num_channels, latent_dim)

    M = variant['algo_params']['sac_layer_size']
    if use_latent_grid:
        conv_channels_in = num_channels if variant['algo_params']['all_channels_to_policy'] else 1,
        if variant['algo_params']['policy_equivariance'] == "toy2D":
            M = M // 2  # sqrt(4)
            policy_net = Toy2DPolicy(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            qf1 = Toy2DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            qf2 = Toy2DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            target_qf1 = Toy2DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            target_qf2 = Toy2DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
        elif variant['algo_params']['policy_equivariance'] == "toy1D":
            M = int(M / np.sqrt(2))
            policy_net = Toy1DPolicy(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            qf1 = Toy1DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            qf2 = Toy1DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            target_qf1 = Toy1DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
            target_qf2 = Toy1DQNet(conv_channels_in, hidden_sizes=[M, M, M], filters=[3, 5], channels=[8, 16])
        elif variant['algo_params']['policy_equivariance'] == "MetaWorldv1":
            M = int(M / np.sqrt(2))  # ensure roughly same amount of parameters
            policy_net = MetaWorldv1Policy(not latent_probabilities, num_channels, num_classes, hidden_sizes=[M, M, M])
            qf1 = MetaWorldv1QNet(not latent_probabilities, num_channels, num_classes, hidden_sizes=[M, M, M])
            qf2 = MetaWorldv1QNet(not latent_probabilities, num_channels, num_classes, hidden_sizes=[M, M, M])
            target_qf1 = MetaWorldv1QNet(not latent_probabilities, num_channels, num_classes, hidden_sizes=[M, M, M])
            target_qf2 = MetaWorldv1QNet(not latent_probabilities, num_channels, num_classes, hidden_sizes=[M, M, M])
        elif variant['algo_params']['policy_equivariance'] == "none":
            if latent_probabilities:
                raise NotImplementedError("TODO: implement normal CNN as baseline")
            else:
                policy_net = MlpPolicyNetwork(hidden_sizes=[M, M, M],
                                              obs_dim=(obs_dim + (latent_dim + 1) * num_channels),
                                              action_dim=action_dim)
                qf1 = FlattenMlp(
                    input_size=(obs_dim + (latent_dim + 1) * num_channels) + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                )
                qf2 = FlattenMlp(
                    input_size=(obs_dim + (latent_dim + 1) * num_channels) + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                )
                target_qf1 = FlattenMlp(
                    input_size=(obs_dim + (latent_dim + 1) * num_channels) + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                )
                target_qf2 = FlattenMlp(
                    input_size=(obs_dim + (latent_dim + 1) * num_channels) + action_dim,
                    output_size=1,
                    hidden_sizes=[M, M, M],
                )
        else:
            raise ValueError("Unknown policy grid equivariance: " + variant['algo_params']['policy_equivariance'])
    else:
        if variant['algo_params']['policy_equivariance'] == "none":
            policy_net = MlpPolicyNetwork(hidden_sizes=[M, M, M],
                                          obs_dim=(obs_dim + task_indicator_dim),
                                          action_dim=action_dim)
            qf1 = FlattenMlp(
                input_size=(obs_dim + task_indicator_dim) + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            qf2 = FlattenMlp(
                input_size=(obs_dim + task_indicator_dim) + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            target_qf1 = FlattenMlp(
                input_size=(obs_dim + task_indicator_dim) + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            target_qf2 = FlattenMlp(
                input_size=(obs_dim + task_indicator_dim) + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
        else:
            # TODO implement them as baseline which should be quite easy
            raise NotImplementedError("Currently, the equivariance \"" + variant['algo_params']['policy_equivariance'] +
                                      "\" of the latent variable without latent grid are not yet implemented!")

    policy = TanhGaussianPolicy(policy_net, variant['algo_params']['random_policy'])
    if variant['util_params']['watch_policy']:
        if not logger.USE_WANDB:
            raise ValueError("Cannot watch policy if not logging to wandb!")
        # wandb.watch(policy, log='all')

    alpha_net = Mlp(
        hidden_sizes=[latent_dim * 10],
        input_size=latent_dim,
        output_size=1
    )

    networks = {'encoder': encoder,
                'prior_pz': prior_pz,
                'decoder': decoder,
                'qf1': qf1,
                'qf2': qf2,
                'target_qf1': target_qf1,
                'target_qf2': target_qf2,
                'policy': policy,
                'alpha_net': alpha_net}

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        itr = variant['showcase_itr']
        path = variant['path_to_weights']
        for name, net in networks.items():
            net.load_state_dict(torch.load(os.path.join(path, name + '_itr_' + str(itr) + '.pth'), map_location='cpu'))

    replay_buffer = StackedReplayBuffer(
        variant['algo_params']['max_replay_buffer_size'],
        time_steps,
        obs_dim,
        action_dim,
        task_indicator_dim,
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['algo_params']['max_path_length'],
        variant['algo_params']['exploration_elbo_reward_factor'] != 0.0
    )

    # Agent
    agent_class = ScriptedPolicyAgent if variant['env_params']['scripted_policy'] else CEMRLAgent
    agent = agent_class(
        encoder,
        prior_pz,
        policy,
        latent_probabilities,
        use_latent_grid,
        variant['algo_params']['latent_grid_resolution'],
        variant['algo_params']['latent_grid_range'],
        latent_dim,
        variant['algo_params']['all_channels_to_policy']
    )

    # Rollout Coordinator
    rollout_coordinator = RolloutCoordinator(
        env,
        variant['env_name'],
        variant['env_params'],
        variant['train_or_showcase'],
        agent,
        replay_buffer,
        variant['algo_params']['batch_size_rollout'],
        time_steps,

        variant['algo_params']['max_path_length'],
        variant['algo_params']['permute_samples'],
        variant['algo_params']['encoding_mode'],
        variant['util_params']['use_multiprocessing'],
        variant['algo_params']['use_data_normalization'],
        variant['util_params']['num_workers'],
        variant['util_params']['gpu_id'],
        variant['env_params']['scripted_policy'],
        variant['algo_params']['exploration_elbo_reward_factor']
    )

    # ReconstructionTrainer
    reconstruction_trainer = ReconstructionTrainer(
        encoder,
        decoder,
        prior_pz,
        replay_buffer,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        num_channels,
        latent_dim,
        time_steps,
        variant['reconstruction_params']['reconstruct_all_timesteps'],
        variant['algo_params']['max_path_length'],
        variant['reconstruction_params']['lr_decoder'],
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['alpha_kl_z'],
        variant['reconstruction_params']['beta_kl_y'],
        variant['reconstruction_params']['use_state_diff'],
        variant['reconstruction_params']['component_constraint_learning'],
        variant['env_params']['state_reconstruction_clip'],
        variant['reconstruction_params']['train_val_percent'],
        variant['reconstruction_params']['eval_interval'],
        variant['reconstruction_params']['early_stopping_threshold'],
        experiment_log_dir,
        variant['reconstruction_params']['prior_mode'],
        variant['reconstruction_params']['prior_sigma'],
        variant['algo_params']['encoding_mode'] == 'transitionIndividualY',
        variant['algo_params']['data_usage_reconstruction'],
        variant['reconstruction_params']['use_state_decoder']
    )

    rollout_coordinator.reconstruction_trainer = reconstruction_trainer

    # PolicyTrainer
    policy_trainer = PolicyTrainer(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        alpha_net,
        replay_buffer,
        variant['algo_params']['batch_size_policy'],
        action_dim,
        variant['algo_params']['data_usage_sac'],
        latent_probabilities,
        use_latent_grid,
        variant['algo_params']['latent_grid_resolution'],
        variant['algo_params']['latent_grid_range'],
        latent_dim,
        num_channels,
        num_classes,
        variant['algo_params']['all_channels_to_policy'],
        variant['algo_params']['elbo_reward_decay_rate'],
        initial_epoch=0 if variant['path_to_weights'] is None else variant['showcase_itr'],
        exploration_elbo_reward_factor=variant['algo_params']['exploration_elbo_reward_factor'],
        use_parametrized_alpha=variant['algo_params']['use_parametrized_alpha'],
        target_entropy_factor=variant['algo_params']['target_entropy_factor'],
        alpha=variant['algo_params']['sac_alpha'],
        use_automatic_entropy_tuning=variant['algo_params']['sac_alpha'],
    )

    combination_trainer = CombinationTrainer(
        # from reconstruction trainer
        encoder,
        decoder,
        prior_pz,
        replay_buffer,
        variant['algo_params']['batch_size_reconstruction'],
        num_classes,
        latent_dim,
        variant['reconstruction_params']['lr_decoder'],
        variant['reconstruction_params']['lr_encoder'],
        variant['reconstruction_params']['alpha_kl_z'],
        variant['reconstruction_params']['beta_kl_y'],
        variant['reconstruction_params']['use_state_diff'],
        variant['env_params']['state_reconstruction_clip'],
        variant['reconstruction_params']['factor_qf_loss'],
        variant['reconstruction_params']['train_val_percent'],
        variant['reconstruction_params']['eval_interval'],
        variant['reconstruction_params']['early_stopping_threshold'],
        experiment_log_dir,

        # from policy trainer
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        action_dim,
        target_entropy_factor=variant['algo_params']['target_entropy_factor']
        # stuff missing
    )

    relabeler = Relabeler(
        encoder,
        replay_buffer,
        variant['algo_params']['batch_size_relabel'],
        action_dim,
        obs_dim,
        variant['algo_params']['use_data_normalization'],
        latent_probabilities
    )

    algorithm = CEMRLAlgorithm(
        replay_buffer,
        rollout_coordinator,
        reconstruction_trainer,
        combination_trainer,
        policy_trainer,
        relabeler,
        agent,
        networks,

        train_tasks,
        test_tasks,
        grid_tasks,

        variant['algo_params']['num_train_epochs'],
        0 if variant['path_to_weights'] is None else variant['showcase_itr'],
        variant['algo_params']['num_reconstruction_steps'],
        variant['algo_params']['num_policy_steps'],
        variant['algo_params']['num_train_tasks_per_episode'],
        variant['algo_params']['num_transitions_initial'],
        variant['algo_params']['num_transitions_per_episode'],
        variant['algo_params']['num_eval_trajectories'],
        variant['algo_params']['showcase_every'],
        variant['algo_params']['snapshot_gap'],
        variant['algo_params']['num_showcase_deterministic'],
        variant['algo_params']['num_showcase_non_deterministic'],
        variant['algo_params']['use_relabeler'],
        variant['algo_params']['use_combination_trainer'],
        experiment_log_dir,
        latent_dim,
        is_analysis,
        variant['algo_params']['task_specs_to_policy']
    )

    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    PLOT = variant['util_params']['plot']
    os.environ['DEBUG'] = str(int(DEBUG))
    os.environ['PLOT'] = str(int(PLOT))

    # create temp folder
    if not os.path.exists(variant['reconstruction_params']['temp_folder']):
        os.makedirs(variant['reconstruction_params']['temp_folder'])
    return algorithm, encoder, decoder, replay_buffer, rollout_coordinator, env, time_steps, test_tasks

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            if k not in to:
                to[k] = dict()
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def deep_dict_to_point(d):
    res = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    v2 = deep_dict_to_point(v2)
                res[k + "." + k2] = v2
        else:
            res[k] = v
    return res

def fix_dict_in_wandb_config():
    """
    Should be fixed by wandb in the future. Source: https://github.com/wandb/client/issues/982
    Not recursive -> only works for at most one "." in a key, but that should be fine in the current config structure
    """
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]

    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v


@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=None)
@click.option('--num_workers', default=4)
@click.option('--use_mp', is_flag=True, default=False)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--resume_wandb', is_flag=True, default=False)
@click.option('--watch_policy', is_flag=True, default=False)
def main(config, gpu, use_mp, num_workers, docker, debug, resume_wandb, watch_policy):
    variant = default_config

    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        if 'path_to_weights' in exp_params:
            path = exp_params['path_to_weights']
            with open(os.path.join(path, 'variant.json')) as f:
                orig_variant = json.load(f)
            variant = deep_update_dict(orig_variant, variant)
            print("Found weights, using its variant.json as default...")
        variant = deep_update_dict(exp_params, variant)

        if variant['path_to_weights'] is not None:
            path = variant['path_to_weights']
            files = [d for d in os.listdir(path) if d.startswith('policy_itr_')]
            iterations = [int(f.split('_')[-1].split('.')[0]) for f in files]
            if variant['showcase_itr'] is None:
                variant['showcase_itr'] = max(iterations)
                print("Detected iteration " + str(variant['showcase_itr']) + "...")
            elif variant['showcase_itr'] not in iterations:
                print("There is no data for the targeted iteration " + str(variant['showcase_itr']))
                exit()
            if resume_wandb:
                if variant['showcase_itr'] == max(iterations):
                    print("Resuming wandb run. Note, that this only makes sense if the run saved its training state "
                          "in the epoch of termination.")
                else:
                    print("Resuming the wandb run with a model state from epoch " + str(variant['showcase_itr']) +
                          " doesn't make sense as the state from epoch" + str(max(iterations)) + " is also available.")
                    exit()
    if gpu is not None:
        variant['util_params']['gpu_id'] = gpu
    variant['util_params']['use_multiprocessing'] = use_mp
    variant['util_params']['num_workers'] = num_workers
    variant['util_params']['resume_wandb'] = resume_wandb
    variant['util_params']['watch_policy'] = watch_policy

    experiment(variant)


if __name__ == "__main__":
    main()

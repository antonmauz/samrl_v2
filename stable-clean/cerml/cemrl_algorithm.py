import numpy as np

import torch
from collections import OrderedDict

import wandb

from analysis import logger
from analysis.request_handler import RequestHandler
from rlkit.core import logger as rlkit_logger
import gtimer as gt
import pickle
import os
import ray
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu
from analysis.encoding import plot_encodings_split
from cerml.utils import generate_latent_grid


class CEMRLAlgorithm:
    def __init__(self,
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

                 num_epochs,
                 initial_epoch,
                 num_reconstruction_steps,
                 num_policy_steps,
                 num_train_tasks_per_episode,
                 num_transitions_initial,
                 num_transistions_per_episode,
                 num_eval_trajectories,
                 showcase_every,
                 snapshot_gap,
                 num_showcase_deterministic,
                 num_showcase_non_deterministic,
                 use_relabeler,
                 use_combination_trainer,
                 experiment_log_dir,
                 latent_dim,
                 is_analysis,
                 task_specs_to_policy,
                 ):
        self.replay_buffer = replay_buffer
        self.rollout_coordinator = rollout_coordinator
        self.reconstruction_trainer = reconstruction_trainer
        self.combination_trainer = combination_trainer
        self.policy_trainer = policy_trainer
        self.relabeler = relabeler
        self.agent = agent
        self.networks = networks

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.grid_tasks = grid_tasks

        self.num_epochs = num_epochs
        self.initial_epoch = initial_epoch
        self.num_reconstruction_steps = num_reconstruction_steps
        self.num_policy_steps = num_policy_steps
        self.num_transitions_initial = num_transitions_initial
        self.num_train_tasks_per_episode = num_train_tasks_per_episode
        self.num_transitions_per_episode = num_transistions_per_episode
        self.num_eval_trajectories = num_eval_trajectories
        self.use_relabeler = use_relabeler
        self.use_combination_trainer = use_combination_trainer
        self.experiment_log_dir = experiment_log_dir
        self.latent_dim = latent_dim
        self.task_specs_to_policy = task_specs_to_policy

        self.showcase_every = showcase_every
        self.snapshot_gap = snapshot_gap
        self.num_showcase_deterministic = num_showcase_deterministic
        self.num_showcase_non_deterministic = num_showcase_non_deterministic

        self._n_env_steps_total = 0

        self.request_handler = RequestHandler(self) if not is_analysis and logger.USE_WANDB else None

    def train(self):
        params = self.get_epoch_snapshot()
        rlkit_logger.save_itr_params(-1, params)
        previous_epoch_end = 0

        print("Collecting initial samples ...")
        if self.num_transitions_initial > 0:
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(self.train_tasks,
                                                                                    num_samples_per_task=
                                                                                    self.num_transitions_per_episode,
                                                                                    task_specs_to_policy=
                                                                                    self.task_specs_to_policy)

        for epoch in gt.timed_for(range(self.initial_epoch, self.initial_epoch + self.num_epochs), save_itrs=True):
            tabular_statistics = OrderedDict()

            # 1. collect data with rollout coordinator
            print("Collecting samples ...")
            data_collection_tasks = np.random.permutation(self.train_tasks)[:self.num_train_tasks_per_episode]
            self._n_env_steps_total += self.rollout_coordinator.collect_replay_data(data_collection_tasks,
                                                                                    num_samples_per_task=
                                                                                    self.num_transitions_per_episode,
                                                                                    task_specs_to_policy=
                                                                                    self.task_specs_to_policy
                                                                                    )
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            gt.stamp('data_collection')

            # replay buffer stats
            self.replay_buffer.stats_dict = self.replay_buffer.get_stats()

            if self.use_combination_trainer:
                # 2. combination trainer
                print("Combination Trainer ...")
                temp, sac_stats = self.combination_trainer.train(self.num_reconstruction_steps)
                tabular_statistics.update(sac_stats)
                gt.stamp('reconstruction_trainer')

                # 3. relabel the data regarding z with relabeler
                if self.use_relabeler:
                    self.relabeler.relabel()
                gt.stamp('relabeler')

                # 4. train policy via SAC with data from the replay buffer
                print("Policy Trainer ...")
                temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
                tabular_statistics.update(sac_stats)

                # alpha optimized through policy trainer should be used in combination trainer as well
                self.combination_trainer.alpha = self.policy_trainer.log_alpha.exp()

            else:
                # 2. encoder - decoder training with reconstruction trainer
                print("Reconstruction Trainer ...")
                self.reconstruction_trainer.train(self.num_reconstruction_steps)
                gt.stamp('reconstruction_trainer')

                # 3. relabel the data regarding z with relabeler
                if self.use_relabeler:
                    self.relabeler.relabel()
                gt.stamp('relabeler')

                # 4. train policy via SAC with data from the replay buffer
                print("Policy Trainer ...")
                temp, sac_stats = self.policy_trainer.train(self.num_policy_steps)
                tabular_statistics.update(sac_stats)
            gt.stamp('policy_trainer')

            # 5. Evaluation
            print("Evaluation ...")
            eval_output = self.rollout_coordinator.evaluate('train', data_collection_tasks, self.num_eval_trajectories,
                                                            deterministic=True, animated=False)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            if logger.USE_TENSORBOARD:
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/average_reward', average_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/std_reward', std_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/max_reward', max_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/min_reward', min_test_reward,
                                                     global_step=self._n_env_steps_total)

                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/average_reward_elbo',
                                                         eval_stats['train_eval_avg_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/std_reward_elbo',
                                                         eval_stats['train_eval_std_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/max_reward_elbo',
                                                         eval_stats['train_eval_max_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/train/det/min_reward_elbo',
                                                         eval_stats['train_eval_min_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
            if logger.USE_WANDB:
                wandb_data = {'evaluation': {'train': {'deterministic': {'reward': {'avg': average_test_reward,
                                                                                    'std': std_test_reward,
                                                                                    'max': max_test_reward,
                                                                                    'min': min_test_reward}}}},
                              'epoch': epoch,
                              'env_steps_total': self._n_env_steps_total}
                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    wandb_data['evaluation']['train']['deterministic']['reward_elbo'] = \
                        {'avg': eval_stats['train_eval_avg_reward_elbo_deterministic'],
                         'std': eval_stats['train_eval_std_reward_elbo_deterministic'],
                         'max': eval_stats['train_eval_max_reward_elbo_deterministic'],
                         'min': eval_stats['train_eval_min_reward_elbo_deterministic']
                         }
                if 'train_eval_success_rates_per_base_task' in eval_stats:
                    wandb_data['evaluation']['train']['deterministic']['success_rate'] =\
                        dict_keys_to_string(eval_stats['train_eval_success_rates_per_base_task'])


            eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories,
                                                            deterministic=False, animated=False)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)
            if logger.USE_TENSORBOARD:
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/average_reward', average_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/std_reward', std_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/max_reward', max_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/min_reward', min_test_reward,
                                                     global_step=self._n_env_steps_total)

                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/average_reward_elbo',
                                                         eval_stats['test_eval_avg_reward_elbo_non_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/std_reward_elbo',
                                                         eval_stats['test_eval_std_reward_elbo_non_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/max_reward_elbo',
                                                         eval_stats['test_eval_max_reward_elbo_non_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/nondet/min_reward_elbo',
                                                         eval_stats['test_eval_min_reward_elbo_non_deterministic'],
                                                         global_step=self._n_env_steps_total)
            if logger.USE_WANDB:
                wandb_data['evaluation']['test'] = {'non_deterministic': {}}
                wandb_data['evaluation']['test']['non_deterministic']['reward'] = \
                    {'avg': average_test_reward,
                     'std': std_test_reward,
                     'max': max_test_reward,
                     'min': min_test_reward
                     }
                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    wandb_data['evaluation']['test']['non_deterministic']['reward_elbo'] = \
                        {'avg': eval_stats['test_eval_avg_reward_elbo_non_deterministic'],
                         'std': eval_stats['test_eval_std_reward_elbo_non_deterministic'],
                         'max': eval_stats['test_eval_max_reward_elbo_non_deterministic'],
                         'min': eval_stats['test_eval_min_reward_elbo_non_deterministic']
                         }
                if 'test_eval_success_rates_per_base_task' in eval_stats:
                    wandb_data['evaluation']['test']['non_deterministic']['success_rate'] =\
                        dict_keys_to_string(eval_stats['test_eval_success_rates_per_base_task'])

            eval_output = self.rollout_coordinator.evaluate('test', self.test_tasks, self.num_eval_trajectories,
                                                            deterministic=True, animated=False)
            average_test_reward, std_test_reward, max_test_reward, min_test_reward, eval_stats = eval_output
            tabular_statistics.update(eval_stats)

            if logger.USE_TENSORBOARD:
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/average_reward', average_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/std_reward', std_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/max_reward', max_test_reward,
                                                     global_step=self._n_env_steps_total)
                logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/min_reward', min_test_reward,
                                                     global_step=self._n_env_steps_total)

                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/average_reward_elbo',
                                                         eval_stats['test_eval_avg_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/std_reward_elbo',
                                                         eval_stats['test_eval_std_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/max_reward_elbo',
                                                         eval_stats['test_eval_max_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
                    logger.TENSORBOARD_LOGGER.add_scalar('evaluation/test/det/min_reward_elbo',
                                                         eval_stats['test_eval_min_reward_elbo_deterministic'],
                                                         global_step=self._n_env_steps_total)
            if logger.USE_WANDB:
                wandb_data['evaluation']['test']['deterministic'] = dict()
                wandb_data['evaluation']['test']['deterministic']['reward'] = \
                    {'avg': average_test_reward,
                     'std': std_test_reward,
                     'max': max_test_reward,
                     'min': min_test_reward
                     }
                if self.rollout_coordinator.exploration_elbo_reward_factor != 0.0:
                    wandb_data['evaluation']['test']['deterministic']['reward_elbo'] = \
                        {'avg': eval_stats['test_eval_avg_reward_elbo_deterministic'],
                         'std': eval_stats['test_eval_std_reward_elbo_deterministic'],
                         'max': eval_stats['test_eval_max_reward_elbo_deterministic'],
                         'min': eval_stats['test_eval_min_reward_elbo_deterministic']
                         }

                if 'test_eval_success_rates_per_base_task' in eval_stats:
                    wandb_data['evaluation']['test']['deterministic']['success_rate'] = \
                        dict_keys_to_string(eval_stats['test_eval_success_rates_per_base_task'])
                wandb.log(wandb_data, commit=True)

            gt.stamp('evaluation')

            # 6. Showcase if wanted
            if self.showcase_every != 0 and epoch % self.showcase_every == 0:
                self.rollout_coordinator.evaluate('test', self.test_tasks[:5], self.num_showcase_deterministic,
                                                  deterministic=True, animated=True, log=False)
                self.rollout_coordinator.evaluate('test', self.test_tasks[:5], self.num_showcase_non_deterministic,
                                                  deterministic=False, animated=True, log=False)
            gt.stamp('showcase')

            # 7. Logging
            # Network parameters
            params = self.get_epoch_snapshot()
            rlkit_logger.save_itr_params(epoch, params)

            if epoch in rlkit_logger._snapshot_points:
                # store encoding
                encoding_storage = self.replay_buffer.check_enc()
                pickle.dump(encoding_storage,
                            open(os.path.join(self.experiment_log_dir, "encoding_" + str(epoch) + ".p"), "wb"))

                # replay stats dict
                pickle.dump(self.replay_buffer.stats_dict,
                            open(os.path.join(self.experiment_log_dir, "replay_buffer_stats_dict_" + str(epoch) + ".p"),
                                 "wb"))
                if logger.USE_WANDB:
                    if self.latent_dim == 1 and True:
                        self.plot_1d_latent_reward_prediction_matrix(0, 25, -7, 7, act=-0.5)
                        self.plot_1d_latent_reward_prediction_matrix(0, 25, -7, 7)
                        self.plot_1d_latent_reward_prediction_matrix(0, 25, -7, 7, act=0.5)

                    if len(self.grid_tasks) > 0:
                        grid_rollouts = self.rollout_coordinator.collect_data(self.grid_tasks, 'grid', deterministic=True,
                                                                              num_trajs_per_task=1, animated=False,
                                                                              save_frames=False)
                        self.plot_behavior_wandb(grid_rollouts, epoch)
                        self.plot_latent_expectation(grid_rollouts, epoch)
                        if self.agent.latent_probabilities:
                            if self.agent.use_latent_grid:
                                self.plot_latent_grids(grid_rollouts, epoch)
                            else:
                                pass
                    if not self.agent.latent_probabilities and self.latent_dim == 1:
                        self.plot_encoding_wandb(epoch, encoding_storage)
            gt.stamp('logging')

            # 8. Time
            times_itrs = gt.get_times().stamps.itrs
            tabular_statistics['time_data_collection'] = times_itrs['data_collection'][-1]
            tabular_statistics['time_reconstruction_trainer'] = times_itrs['reconstruction_trainer'][-1]
            tabular_statistics['time_relabeler'] = times_itrs['relabeler'][-1]
            tabular_statistics['time_policy_trainer'] = times_itrs['policy_trainer'][-1]
            tabular_statistics['time_evaluation'] = times_itrs['evaluation'][-1]
            tabular_statistics['time_showcase'] = times_itrs['showcase'][-1]
            tabular_statistics['time_logging'] = times_itrs['logging'][-1]
            total_time = gt.get_times().total
            epoch_time = total_time - previous_epoch_end
            previous_epoch_end = total_time
            tabular_statistics['time_epoch'] = epoch_time
            tabular_statistics['time_total'] = total_time

            # other
            tabular_statistics['n_env_steps_total'] = self._n_env_steps_total
            tabular_statistics['epoch'] = epoch

            for key, value in tabular_statistics.items():
                rlkit_logger.record_tabular(key, value)

            rlkit_logger.dump_tabular(with_prefix=False, with_timestamp=False)
            if self.request_handler is not None:
                self.request_handler.handle_updates(epoch)

        ray.shutdown()

    def get_epoch_snapshot(self):
        snapshot = OrderedDict()
        for name, net in self.networks.items():
            snapshot[name] = net.state_dict()
        return snapshot

    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            self.networks[net].to(device)
        self.agent.to(device)

    def plot_encodings(self, epoch):
        encoding_storage = pickle.load(
            open(os.path.join(self.experiment_log_dir, "encoding_" + str(epoch) + ".p"), "rb"))
        base_tasks = list(encoding_storage.keys())
        fig, axes_tuple = plt.subplots(ncols=len(base_tasks), sharey=True)
        if len(base_tasks) == 1: axes_tuple = [axes_tuple]
        for i, base in enumerate(base_tasks):
            for dim in range(self.latent_dim):
                axes_tuple[i].errorbar(list(encoding_storage[base].keys()),
                                       [a['mean'][dim] for a in list(encoding_storage[base].values())],
                                       yerr=[a['std'][dim] for a in list(encoding_storage[base].values())], fmt="o")
            axes_tuple[i].plot(list(encoding_storage[base].keys()),
                               [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x')
        plt.show()

    def plot_behavior_wandb(self, rollouts, epoch):
        """
        For the latent expectations:
        Plots the expected values of the gaussians in the most likely channel for different tasks and some timesteps.
        IMPORTANT: In constrast to the latent grids, this only takes into account some timesteps separately
        Therefore, this should even look good for tasks with uncertainty. Whereas the expected value might be 0 in the
        beginning, when it is unknown of the goal is left or right, in the last time step, the distribution should only
        represent the goal position.
        """
        data = []
        spec_scalar = np.isscalar(rollouts[0]['env_infos'][0]['true_task']['specification'])
        num_specs = 1 if spec_scalar else rollouts[0]['env_infos'][0]['true_task']['specification'].shape[0]
        columns = ["time_step", "num_goal"] + ["goal_spec_" + str(i) for i in range(num_specs)] +\
                  ["achieved_spec_" + str(i) for i in range(num_specs)] +\
                  ["latent_mean_" + str(i) for i in range(self.latent_dim)]

        # Latent expectations
        num_classes = self.policy_trainer.num_classes
        num_channels = self.policy_trainer.num_channels
        latent_probabilities = self.policy_trainer.latent_probabilities
        num_tasks = len(rollouts)
        for time in range(self.rollout_coordinator.max_path_length):
            task_z = np.stack([results['task_indicators'][time] for results in rollouts], axis=0)
            if latent_probabilities:
                channel_probs = task_z[:, :num_channels * num_classes]. \
                    reshape(num_tasks * num_channels, num_classes). \
                    sum(axis=1).reshape(num_tasks, num_channels)
                # [batch_size]
                channels = channel_probs.argmax(axis=1)
                # [batch_size]
                channel_probs = channel_probs.max(axis=1)
                # [batch_size, num_classes * latent_dim]
                means = task_z[np.arange(num_tasks)[:, None],
                               channels[:, None] * num_classes * self.latent_dim * 2 +  # start of correct channel
                               np.tile(np.arange(self.latent_dim), num_classes)[None, :] +  # index of latent dimension
                               np.repeat(np.arange(num_classes)[None, :], self.latent_dim,
                                         axis=1) * 2 * self.latent_dim +  # start of class mean (skipping variance with * 2)
                               num_channels * num_classes]  # offset for y probabilities
                # [batch_size, num_classes * latent_dim]
                probs = np.take_along_axis(task_z, channels[:, None] * num_classes +
                                           np.arange(num_classes)[None, :], axis=1).repeat(self.latent_dim, axis=1) / \
                        channel_probs[:, None]
                expectations = np.empty((num_tasks, self.latent_dim))
                weighted_means = (means * probs)
                for d in range(self.latent_dim):
                    expectations[:, d] = np.sum(weighted_means[:, d::self.latent_dim], axis=1)
            else:
                expectations = task_z

            for num_goal, res in enumerate(rollouts):
                specs = res['env_infos'][time]['true_task']['specification']
                specs = [specs] if np.isscalar(specs) else list(specs)
                achieved = res['env_infos'][time]['achieved_spec']
                achieved = [achieved] if np.isscalar(achieved) else list(achieved)
                data.append([time, num_goal] + specs + achieved + list(expectations[num_goal]))

            # plt.plot(list(range(len(velocity_goal))), velocity_goal, '--', color=colors[i])
            # plt.plot(list(range(len(velocity_is))), velocity_is, color=colors[i])
        table = wandb.Table(data=data, columns=columns)

        wandb.log({"behavior_" + str(num_specs) + "d_latent_" + str(self.latent_dim) + "d_plot": table, "epoch": epoch})
        # plt.legend(custom_lines, ['target', 'achieved'], fontsize=fontsize, loc='lower right')
        # plt.xlabel("time step", fontsize=fontsize)
        # plt.ylabel("achieved", fontsize=fontsize)


    def plot_latent_grids(self, rollouts, epoch):
        """
        For each task, this plots latent distribution averaged over the whole episode (green) along with the standard
        deviation of this distribution within the episode
        TODO: This might be misleading if the most likely channel changes within the run, as the latent values might
         have different meanings
        """
        cols = min(4, len(rollouts)) if self.latent_dim == 2 else 1
        rows = int(np.ceil(len(rollouts) / cols))
        num_classes = self.policy_trainer.num_classes
        num_channels = self.policy_trainer.num_channels
        grid_res = self.agent.latent_grid_resolution
        grid_range = self.agent.latent_grid_range
        # import matplotlib
        # matplotlib.use("TkAgg")
        ticks = [0, grid_res]
        ticklabels = [str(-grid_res), str(grid_res)]
        fig, axes_tuple = plt.subplots(nrows=rows, ncols=cols, sharex='col', sharey='row', figsize=(4 * cols, 4 * rows))
        if len(axes_tuple.shape) == 1:
            axes_tuple = np.expand_dims(axes_tuple, 1)

        for i, results in enumerate(rollouts):
            # task_z: [episode_length, channels * classes * (2 * latent + 1)] (probs of y+b*K and means/variances for each)
            task_z = ptu.from_numpy(np.stack([z for z in results['task_indicators']], axis=0))
            # grids: [episode_length, grid_res, grid_res]
            grids = generate_latent_grid(task_z[:, : num_classes * num_channels], None, num_channels, num_classes,
                                         self.latent_dim, grid_res, grid_range, False,
                                         z_probs=task_z[:, num_classes * num_channels:])[:, 0, :, :]
            # make the latent encoding for the task green and the standard deviation blue:
            img = np.zeros((1 if self.latent_dim == 1 else grid_res, grid_res, 3))
            img[:, :, 1] = ptu.get_numpy(torch.mean(grids, dim=0))
            img[:, :, 2] = ptu.get_numpy(torch.std(grids, dim=0))
            max_prob = img[:, :, 1].max()
            max_std = img[:, :, 2].max()
            img[:, :, 1] /= max_prob
            img[:, :, 2] /= max_std
            axes_tuple[i // cols][i % cols].imshow(img)
            axes_tuple[i // cols][i % cols].set_xticks(ticks)
            axes_tuple[i // cols][i % cols].set_xticklabels(ticklabels)
            if self.latent_dim == 2:
                axes_tuple[i // cols][i % cols].set_yticks(ticks)
                axes_tuple[i // cols][i % cols].set_yticklabels(ticklabels)
            axes_tuple[i // cols][i % cols].set_title("Spec: " +
                                                      str(results['env_infos'][0]['true_task']['specification']) +
                                                      "\nMaxProb: " + str(round(max_prob, 2)) +
                                                      ", MaxStd: " + str(round(max_std, 2)))
        fig.tight_layout()
        # plt.show()
        wandb.log({"latent_grids_2D": fig, "epoch": epoch})

    def plot_latent_expectation(self, rollouts, epoch, different_timesteps=3):
        """
        Plots the expected values of the gaussians in the most likely channel for different tasks and some timesteps.
        IMPORTANT: In constrast to the latent grids, this only takes into account some timesteps separately
        Therefore, this should even look good for tasks with uncertainty. Whereas the expected value might be 0 in the
        beginning, when it is unknown of the goal is left or right, in the last time step, the distribution should only
        represent the goal position.
        """
        num_classes = self.policy_trainer.num_classes
        num_channels = self.policy_trainer.num_channels
        latent_probabilities = self.policy_trainer.latent_probabilities
        num_tasks = len(rollouts)
        time_indices = [int((self.rollout_coordinator.max_path_length - 1) * i / (different_timesteps - 1))
                        for i in range(different_timesteps)]

        #for i, results in enumerate(rollouts):
        for i in time_indices:
            # task_z: [episode_length, channels * classes * (2 * latent + 1)] (probs of y+b*K and means/variances for each)
            task_z = np.stack([results['task_indicators'][i] for results in rollouts], axis=0)
            if latent_probabilities:
                channel_probs = task_z[:, :num_channels*num_classes].\
                    reshape(num_tasks * num_channels, num_classes).\
                    sum(axis=1).reshape(num_tasks, num_channels)
                # [batch_size]
                channels = channel_probs.argmax(axis=1)
                # [batch_size]
                channel_probs = channel_probs.max(axis=1)
                # [batch_size, num_classes * latent_dim]
                means = task_z[np.arange(num_tasks)[:, None],
                               channels[:, None] * num_classes * self.latent_dim * 2 +  # start of correct channel
                               np.tile(np.arange(self.latent_dim), num_classes)[None, :] +  # index of latent dimension
                               np.repeat(np.arange(num_classes)[None, :], self.latent_dim, axis=1) * 2 * self.latent_dim +  # start of class mean (skipping variance with * 2)
                               num_channels * num_classes]  # offset for y probabilities
                # [batch_size, num_classes * latent_dim]
                probs = np.take_along_axis(task_z, channels[:, None] * num_classes +
                                           np.arange(num_classes)[None, :], axis=1).repeat(self.latent_dim, axis=1) /\
                        channel_probs[:, None]
                expectations = np.empty((num_tasks, self.latent_dim))
                weighted_means = (means * probs)
                for d in range(self.latent_dim):
                    expectations[:, d] = np.sum(weighted_means[:, d::self.latent_dim], axis=1)
            else:
                expectations = task_z

            data = []
            for t, res in enumerate(rollouts):
                row = res['env_infos'][i]['true_task']['specification']
                row = [row] if np.isscalar(row) else list(row)
                row += list(expectations[t])
                data.append(row)
            table = wandb.Table(columns=["spec_" + str(j) for j in range(len(data[0]) - self.latent_dim)] +
                                        ["latent_" + str(j) for j in range(self.latent_dim)], data=data)
            wandb.log({"latent_expectation_" + str(self.latent_dim) + "d_" + str(i): table, "epoch": epoch})

    def plot_encoding_wandb(self, epoch, encoding_storage=None):
        if encoding_storage is None:
            encoding_storage = self.replay_buffer.check_enc()
        plot_encodings_split(epoch, self.experiment_log_dir, show=False, save_wandb=True,
                             encoding_storage=encoding_storage)

    def plot_1d_latent_reward_prediction_matrix(self, min_s, max_s, min_z, max_z, res_s=25, res_z=25, act=0):
        """
        Creates a matrix of the predicted rewards for action 0 with states from min_s to max_s on the y-axis and latent
        encodings from min_z to max_z on the x_axis assuming class / channel 0
        """
        state = ((max_s - min_s) * (ptu.arange(res_s) / (res_s - 1.)) + min_s).repeat_interleave(res_z)[:, None]
        z = ((max_z - min_z) * (ptu.arange(res_z) / (res_z - 1.)) + min_z).repeat(res_s)[:, None]
        action = ptu.ones(res_s * res_z, 1) * act
        _, reward_estimate = self.reconstruction_trainer.decoder(state, action, None, z, ptu.tensor(0))
        maxe = torch.max(reward_estimate)
        mine = torch.min(reward_estimate)
        if maxe != mine:
            reward_estimate = ((reward_estimate - mine) / (maxe - mine)) + mine
        wandb.log({"latent_reward_prediction_" + str(act): wandb.Image(ptu.get_numpy(reward_estimate.reshape(res_s, res_z)))})


def dict_keys_to_string(d):
    return {str(k): v for k, v in d.items()}
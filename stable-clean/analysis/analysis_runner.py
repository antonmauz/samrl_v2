# Continuous Embedding Meta Reinforcement Learning (CEMRL)

import ray  # avoid error of importing ray too late
import os, shutil
import pathlib
import numpy as np
import click
import json
import torch
import torch.nn as nn
import cv2

from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.analysis_config import analysis_config
from runner_jonas import setup_algorithm, deep_update_dict
from configs.default import default_config

import pickle
import wandb

from analysis.encoding import plot_encodings_split
from analysis.progress_logger import manage_logging
from multiprocessing import Process
import copy


def experiment(variant):
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    torch.set_num_threads(1)
    if variant['algo_params']['use_fixed_seeding']:
        torch.manual_seed(variant['algo_params']['seed'])
        np.random.seed(variant['algo_params']['seed'])

    # create logging directory
    encoding_save_epochs = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 225, 250, 275, 300,
                            325, 350, 375, 400, 425, 450, 475, 500, 600, 750, 1000, 1250, 1500, 1750, 2000,
                            3000, 5000, 10000, 15000, 20000]
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=None,
                                      base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode='specific',
                                      snapshot_gap=variant['algo_params']['snapshot_gap'],
                                      snapshot_points=encoding_save_epochs)

    path_to_folder = variant['path_to_weights']
    showcase_itr = variant['showcase_itr']
    example_case = variant['analysis_params']['example_case']
    save = variant['analysis_params']['save']

    # log file and plot progress file
    if variant['analysis_params']['log_and_plot_progress']:
        manage_logging(path_to_folder, save=save)

    # plot encodings
    if variant['analysis_params']['plot_encoding']:
        plot_encodings_split(showcase_itr, path_to_folder, save=save)

    algorithm, encoder, decoder, replay_buffer, rollout_coordinator, env, time_steps, test_tasks =\
        setup_algorithm(variant, experiment_log_dir, True)

    replay_buffer.stats_dict = pickle.load(open(os.path.join(path_to_folder, "replay_buffer_stats_dict_" + str(showcase_itr) + ".p"), "rb"))
    env.reset_task(np.random.randint(len(env.test_tasks)) + len(env.train_tasks))
    env.set_meta_mode('test')

    # visualize test cases
    results = rollout_coordinator.collect_data(test_tasks, 'test',
                                               deterministic=True, animated=variant['analysis_params']['visualize_run'],
                                               save_frames=False, num_trajs_per_task=1)

    # Reward for this run
    # per_path_rewards = [np.sum(path["rewards"]) for worker in results for task in worker for path in task[0]]
    # per_path_rewards = np.array(per_path_rewards)
    # eval_average_reward = per_path_rewards.mean()
    # print("Average reward: " + str(eval_average_reward))

    # velocity plot
    if variant['analysis_params']['plot_time_response']:
        import matplotlib.pyplot as plt
        plt.figure(figsize=((2/4)*7, (2/4)*4.7))
        if variant['env_name'].split('-')[-1] == 'vel':
            velocity_is = [a['velocity'] for a in results[0][0][0][0]['env_infos']]
            velocity_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
        else:
            velocity_is = [a['achieved_spec'] for a in results[1]['env_infos']]
            velocity_goal = [a['true_task']['specification'] for a in results[1]['env_infos']]
        filter_constant = time_steps
        velocity_is_temp = ([0] * filter_constant) + velocity_is
        velocity_is_filtered = []
        for i in range(len(velocity_is)):
            velocity_is_filtered.append(sum(velocity_is_temp[i:i+filter_constant]) / filter_constant)
        velocity_goal = np.array(velocity_goal)
        velocity_is = np.array(velocity_is)
        velocity_is_filtered = np.array(velocity_is_filtered)
        # velocity_is = numpy_ewma_vectorized_v2(velocity_is, 10)
        # velocity_is_filtered = numpy_ewma_vectorized_v2(velocity_is_filtered, 10)
        plt.plot(np.arange(velocity_goal.shape[0]), velocity_goal, label="goal velocity")
        plt.plot(np.arange(velocity_is.shape[0]), velocity_is, label="velocity")
        plt.plot(np.arange(velocity_is_filtered.shape[0]), velocity_is_filtered, label="velocity filtered")
        plt.xlabel("time $t$")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_velocity_new" + ".pdf", dpi=300, format="pdf")
            print("Saved to: " + path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_velocity_new" + ".pdf")
        # plt.show()
    if variant['env_name'].split('-')[-1] == 'dir' and variant['analysis_params']['plot_time_response']:
        import matplotlib.pyplot as plt
        figsize=None
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes_tuple = plt.subplots(nrows=2, ncols=1, sharex='col', gridspec_kw={'height_ratios': [1, 1]}, figsize=figsize)
        direction_is = [a['direction'] for a in results[0][0][0][0]['env_infos']]
        direction_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
        axes_tuple[0].plot(list(range(len(direction_is))), direction_is, color=cycle[0], label="velocity")
        #axes_tuple[1].plot(list(range(len(direction_goal))), np.sign(direction_is), color=cycle[1], label="direction")
        axes_tuple[1].plot(list(range(len(direction_goal))), direction_goal, color=cycle[1], label="goal direction")
        axes_tuple[0].grid()
        #axes_tuple[1].grid()
        axes_tuple[1].grid()
        axes_tuple[0].legend(loc='upper right')
        axes_tuple[1].legend(loc='upper right')
        #axes_tuple[2].legend(loc='lower left')
        axes_tuple[1].set_xlabel("time $t$")
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_direction_new" + ".pdf", dpi=300, bbox_inches='tight', format="pdf")
        plt.show()

    if variant['analysis_params']['plot_velocity_multi']:
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pl
        plt.figure(figsize=(10,5))
        colors = pl.cm.coolwarm(np.linspace(0, 1, len(test_tasks)))
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(test_tasks)):
            results = rollout_coordinator.collect_data(test_tasks[i:i + 1], 'test', deterministic=True, num_trajs_per_task=1,
                                                       animated=False, save_frames=False)
            if variant['env_name'].split('-')[-1] == 'vel':
                velocity_is = [a['velocity'] for a in results[0][0][0][0]['env_infos']]
                velocity_goal = [a['true_task']['specification'] for a in results[0][0][0][0]['env_infos']]
            else:
                velocity_is = [a['achieved_spec'] for a in results[0]['env_infos']]
                velocity_goal = [a['true_task']['specification'] for a in results[0]['env_infos']]
            plt.plot(list(range(len(velocity_goal))), velocity_goal, '--', color=colors[i])
            plt.plot(list(range(len(velocity_is))), velocity_is, color=colors[i])

        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='gray', linestyle='--'),
                        Line2D([0], [0], color='gray')]

        fontsize = 14
        plt.legend(custom_lines, ['goal velocity', 'velocity'], fontsize=fontsize, loc='lower right')
        plt.xlabel("time step $t$", fontsize=fontsize)
        plt.ylabel("velocity $v$", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.xlim(0, len(list(range(len(velocity_goal)))))
        #plt.title("cheetah-stationary-vel: velocity vs. goal velocity", fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "multiple_velocity_vs_goal_velocity" + ".pdf", dpi=300, format="pdf")
            print("Saved to: " + path_to_folder + '/' + variant['env_name'] + '_' + str(showcase_itr) + '_' + "velocity_vs_goal_velocity_new" + ".pdf")
        plt.show()
    # video taking
    if variant['analysis_params']['produce_video']:
        for i in range(len(test_tasks)):
            video_name_string = path_to_folder.split('/')[-1] + "_" + variant['env_name'] + "_" +\
                                str(variant['showcase_itr']) + "_" + str(i) + ".webm"
            os.makedirs(os.path.join(path_to_folder, "videos"), exist_ok=True)
            video_filename = os.path.join(path_to_folder, "videos", video_name_string)
            print(video_filename)
            if os.path.isfile(video_filename):
                continue # don't reproduce existing videos
            print("Producing video " + str(i + 1) + "/" + str(
                len(test_tasks)) + "... do NOT kill program until completion!")
            results = rollout_coordinator.collect_data(test_tasks[i : i + 1], 'test', deterministic=True,
                                                       num_trajs_per_task=1, animated=False, save_frames=True)
            path_video = results[0]
            first_frame = path_video['env_infos'][0]['frame']
            out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'VP90'), #*'mp4v'
                                  20.0, (first_frame.shape[1], first_frame.shape[0]))
            for step, env_info in enumerate(path_video['env_infos']):
                image = np.array(env_info['frame'])
                # Convert RGB to BGR
                image = image[:, :, ::-1].copy()
                cv2.putText(image, 'spec: ' + str(round(env_info['true_task']['specification'], 2)), (0, 15),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(image, 'z: ' + str(path_video['task_indicators'][step]), (0, 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(image, 'reward: ' + str(path_video['rewards'][step][0]), (0, 45),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                #cv2.putText(image, 'pos: ' + str(env_info['position']), (0, 60),
                #            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

                # write the flipped frame
                out.write(image)

            # Release everything if job is finished
            out.release()
            if variant['analysis_params']['wandb']:
                wandb.log({'video' + str(i): wandb.Video(video_filename, format='webm')}, commit=(i == len(test_tasks) - 1))

@click.command()
@click.option('--gpu', default=0)
@click.option('--num_workers', default=8)
@click.option('--use_mp', is_flag=True, default=False)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(gpu, use_mp, num_workers, docker, debug):
    analysis_config['util_params']['gpu_id'] = gpu
    analysis_config['util_params']['use_multiprocessing'] = use_mp
    analysis_config['util_params']['num_workers'] = num_workers

    paths = analysis_config['path_to_weights']
    variant = deep_update_dict(analysis_config, default_config)
    processes = []
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        conf = copy.deepcopy(variant)
        conf['path_to_weights'] = path
        p = Process(target=run_for_path, args=(conf,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def run_for_path(variant):
    path_to_folder = variant['path_to_weights']
    with open(os.path.join(os.path.join(path_to_folder, 'variant.json'))) as f:
        exp_params = json.load(f)

    files = [d for d in os.listdir(path_to_folder) if d.startswith('policy_itr_')]
    iterations = [int(f.split('_')[-1].split('.')[0]) for f in files]
    if variant['showcase_itr'] is None:
        variant['showcase_itr'] = max(iterations)
        print("Detected iteration " + str(variant['showcase_itr']) + "...")
    elif (not isinstance(variant['showcase_itr'], list)) and variant['showcase_itr'] not in iterations:
        print("There is no data for the targeted iteration " + str(variant['showcase_itr']))
        exit()

    variant["env_name"] = exp_params["env_name"]
    variant["env_params"] = deep_update_dict(exp_params["env_params"], variant["env_params"])
    variant["algo_params"] = deep_update_dict(exp_params["algo_params"], variant["algo_params"])
    variant["reconstruction_params"] = deep_update_dict(exp_params["reconstruction_params"], variant["reconstruction_params"])


    # set other time steps than while training
    if variant["analysis_params"]["manipulate_time_steps"]:
        variant["algo_params"]["time_steps"] = variant["analysis_params"]["time_steps"]

    # set other time steps than while training
    if variant["analysis_params"]["manipulate_change_trigger"]:
        variant["env_params"] = deep_update_dict(variant["analysis_params"]["change_params"], variant["env_params"])

    # set other episode length than while training
    if variant["analysis_params"]["manipulate_max_path_length"]:
        variant["algo_params"]["max_path_length"] = variant["analysis_params"]["max_path_length"]

    # set other task number than while training
    if variant["analysis_params"]["manipulate_test_task_number"]:
        variant["env_params"]["n_eval_tasks"] = variant["analysis_params"]["test_task_number"]

    if variant['analysis_params']['wandb']:
        wandb.init(project='cemrl', config=variant)
        wandb.config.exp_name = variant['path_to_weights'].split('/')[-1]
    if isinstance(variant['showcase_itr'], list):
        iters = variant['showcase_itr']
        for it in iters:
            if it in iterations:
                variant['showcase_itr'] = it
                experiment(variant)
            else:
                print("Could not find iteration: " + str(it))
    else:
        experiment(variant)
    if variant['analysis_params']['wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()

import gc
import os

import numpy as np
import gym
# import gridworld
import wandb

import torch
import torch.nn.functional as F

from cerml.encoder_decoder_networks import EncoderMixtureModelGRU
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MlpPolicyNetwork
from symmetrizer.jonas.symmetric_networks_metaworld_v1 import *
from symmetrizer.jonas.symmetric_networks_toy1d import Toy1DPolicy, Toy1DQNet, Toy1DDecoder
from symmetrizer.jonas.symmetric_networks_toy2d import Toy2DPolicy, Toy2DQNet, Toy2DDecoder
from symmetrizer.symmetrizer.ops import g2c, c2g, get_grid_actions
from symmetrizer.symmetrizer.groups.ops import closed_group
from cerml.utils import to_latent_hot
import rlkit.torch.pytorch_util as ptu
from symmetrizer.tests import gpu_debugger

USE_WANDB = False

"""
Note, that in all of the tests, everything is done very explicitly (e.g. looping over tensors instead of vectorized
operations) and relying on existing methods (e.g. from symmetric_ops_...) is avoided wherever possible in order to
avoid repeating a mistake from the implementation
"""


def test_toy1d_encoder(network=None):
    ptu.set_gpu_mode(False, 0)
    time_steps = 8
    num_classes = 2 # 6
    num_channels = 1 # 3
    trajectory = ptu.rand(1, time_steps, 4) * 4 - 2
    if network is None:
        network = EncoderMixtureModelGRU(shared_dim=4*10,  # encoder_input_dim * enc_dec_complexity
                                         encoder_input_dim=4,  # obs_dim + action_dim + reward_dim + obs_dim
                                         latent_dim=1,
                                         batch_size=64,
                                         num_classes=num_classes,
                                         num_channels=num_channels,
                                         time_steps=None,
                                         merge_mode=None,
                                         use_latent_grid=True,
                                         latent_grid_resolution=8,
                                         latent_grid_range=6,
                                         restrict_latent_mean=True,
                                         encoding_noise=0,
                                         constant_encoding=False,
                                         equivariance="toy1D")
        # network.load_state_dict(torch.load(os.path.join(
        #     '../cemrl_server_data/toy-goal-1d-equiv-randomStarts-nets/encoder_itr_100.pth'), map_location='cpu'))

    def reflect(traj):
        return ptu.tensor([-1, -1, 1, -1])[None, None, :] * traj

    def roll(classes, n=1):
        res = classes.clone().detach()
        for c in range((num_classes * num_channels) // 2):
            res[:, c * 2: (c + 1) * 2] = torch.roll(classes[:, c * 2: (c + 1) * 2], n)
        return res

    y_distr, z_distrs = network.encode(trajectory)
    # print(y_distr.probs)
    y_distr_r, z_distrs_r = network.encode(reflect(trajectory))
    assert_similar([roll(y_distr.probs), y_distr_r.probs])
    for i in range(len(z_distrs)):
        assert_similar([z_distrs[i].loc, -z_distrs_r[i].loc])
        assert_similar([z_distrs[i].scale, z_distrs_r[i].scale])
    return True

def test_toy1d_policy(network=None, grid=None, state=None, num_channels=2):
    if grid is None:
        # frame = np.random.uniform(size=(1, 1, 7 * 3))
        grid = np.arange(num_channels * 32).reshape((1, num_channels, 1, 32)) / 32
    if state is None:
        state = torch.tensor([0.3])
    if network is None:
        network = Toy1DPolicy(num_channels)

    t_x = torch.Tensor(grid)
    t_r_x1 = torch.flip(torch.Tensor(grid), dims=(-1,))

    def reflect(y):
        return -y

    # Caution: coordinate format [y, x]
    act, std, log_std = network((state[None, :], t_x))
    act1, std1, log_std1 = network((reflect(state)[None, :], t_r_x1))
    # print(t_x)
    # print(t_r_x3)
    # print(state, rot_right(state))
    # print(act, act4, act3, act2)
    assert((std >= 0).all() and (std1 >= 0).all())
    assert_similar([act[0], reflect(act1[0])])
    assert_similar([std[0], std1[0]])
    return True

def test_toy1d_q(network=None, grid=None, state=None, action=None, num_channels=2):
    if grid is None:
        # frame = np.random.uniform(size=(1, 1, 7 * 3))
        grid = np.arange(num_channels * 32).reshape((1, num_channels, 1, 32)) / 32
    if state is None:
        state = torch.tensor([0.3])
    if action is None:
        action = torch.tensor([-0.2])
    if network is None:
        network = Toy1DQNet(num_channels)

    # Note, that this rotates right whereas I rotate left.
    t_x = torch.Tensor(grid)
    t_r_x1 = torch.flip(torch.Tensor(grid), dims=(2,))

    def reflect(y):
        return -y

    r = network((state[None, :], t_x), action[None, :])
    r1 = network((reflect(state)[None, :], t_r_x1), reflect(action)[None, :])
    assert_similar([r, r1])
    return True

def test_toy1d_decoder(network=None, state=None, action=None, latent=None, num_channels=5, channel=None):
    if channel is None:
        channel = np.random.randint(num_channels)
    if latent is None:
        latent = torch.tensor([-0.6])
    if state is None:
        state = torch.tensor([0.1])
    if action is None:
        action = torch.tensor([-0.2])
    if network is None:
        network = Toy1DDecoder(num_channels)

    def reflect(y):
        return -y

    def build_input(s, a, c, l):
        res = torch.zeros(2 * num_channels)
        res[c] = 1
        res[num_channels + c:num_channels + c + 1] = l
        return torch.cat((s, a, res))


    # Caution: coordinate format [y, x]
    r = network(build_input(state, action, channel, latent)[None, :])
    r1 = network(build_input(reflect(state), reflect(action), channel, reflect(latent))[None, :])
    assert_similar([r, r1])
    return True

def test_toy2d_encoder(network=None):
    ptu.set_gpu_mode(False, 0)
    time_steps = 8
    num_classes = 4  # 12
    num_channels = 3
    encoder_input_dim = 2 + 2 + 1 + 2
    trajectory = ptu.rand(1, time_steps, encoder_input_dim) * 4 - 2
    trajectory = ptu.arange(time_steps * encoder_input_dim).reshape(1, time_steps, encoder_input_dim) -\
                 ((time_steps * encoder_input_dim) / 2)
    if network is None:
        network = EncoderMixtureModelGRU(shared_dim=encoder_input_dim*10,  # encoder_input_dim * enc_dec_complexity
                                         encoder_input_dim=encoder_input_dim,  # obs_dim + action_dim + reward_dim + obs_dim
                                         latent_dim=2,
                                         batch_size=1,
                                         num_classes=num_classes,
                                         num_channels=num_channels,
                                         time_steps=None,
                                         merge_mode=None,
                                         use_latent_grid=True,
                                         latent_grid_resolution=8,
                                         latent_grid_range=6,
                                         restrict_latent_mean=True,
                                         encoding_noise=0,
                                         constant_encoding=False,
                                         equivariance="toy2D")
        network.load_state_dict(torch.load(os.path.join(
            '../cemrl_server_data/toy-goal-2d-equiv-randomStarts-nets/encoder_itr_75.pth'), map_location='cpu'))

    def rot(pos, positive=False):
        return torch.stack((pos[:, 1], (1 if positive else -1) * pos[:, 0]), dim=1)

    def rot_traj(traj):
        res = traj.clone().detach()
        for b in range(traj.shape[0]):  # batch entry
            for t in range(traj.shape[1]):  # time step
                for i in [0, 2, 5]:  # start indices of s, a and s'
                    res[b, t, i] = traj[b, t, i + 1]
                    res[b, t, i + 1] = -traj[b, t, i]
        return res

    def roll(classes, n=1):
        res = classes.clone().detach()
        for c in range((num_classes * num_channels) // 4):
            res[:, c * 4: (c + 1) * 4] = torch.roll(classes[:, c * 4: (c + 1) * 4], n)
        return res

    traj_r1 = rot_traj(trajectory)
    traj_r2 = rot_traj(rot_traj(trajectory))
    traj_r3 = rot_traj(rot_traj(rot_traj(trajectory)))

    y_distr, z_distrs = network.encode(trajectory)
    y_distr_r1, z_distrs_r1 = network.encode(traj_r1)
    y_distr_r2, z_distrs_r2 = network.encode(traj_r2)
    y_distr_r3, z_distrs_r3 = network.encode(traj_r3)
    m = network.shared_encoder(trajectory)[1][0, :, :]
    m_r1 = network.shared_encoder(traj_r1)[1][0, :, :]
    m_r2 = network.shared_encoder(traj_r2)[1][0, :, :]
    m_r3 = network.shared_encoder(traj_r3)[1][0, :, :]
    assert_similar([torch.roll(m, 3, dims=2), torch.roll(m_r1, 2, dims=2), torch.roll(m_r2, 1, dims=2), m_r3])
    assert_similar([roll(y_distr.probs, n=3), roll(y_distr_r1.probs, n=2), roll(y_distr_r2.probs), y_distr_r3.probs])

    for i in range(len(z_distrs)):
        assert_similar([rot(rot(rot(z_distrs[i].loc))), rot(rot(z_distrs_r1[i].loc)), rot(z_distrs_r2[i].loc),
                        z_distrs_r3[i].loc])
        assert_similar([rot(rot(rot(z_distrs[i].scale, True), True), True), rot(rot(z_distrs_r1[i].scale, True), True),
                        rot(z_distrs_r2[i].scale, True), z_distrs_r3[i].scale])
    return True

def test_toy2d_policy(network=None, grid=None, state=None, num_channels = 2):
    if grid is None:
        # frame = np.random.uniform(size=(1, 7 * 3, 7 * 3))
        grid = np.arange(num_channels * 7 * 7 * 3 * 3).reshape((1, num_channels, 7 * 3, 7 * 3)) / (7 * 7 * 3 * 3)
        grid[0, -6:, -6:] = 1
    if state is None:
        state = torch.tensor([0.1, 0.5])
    if network is None:
        network = Toy2DPolicy(num_channels)

    # Note, that this rotates right whereas I rotate left.
    t_x = torch.Tensor(grid)
    t_r_x1 = torch.Tensor(np.rot90(grid.copy(), k=1, axes=(-2, -1)).copy())
    t_r_x2 = torch.Tensor(np.rot90(grid.copy(), k=2, axes=(-2, -1)).copy())
    t_r_x3 = torch.Tensor(np.rot90(grid.copy(), k=3, axes=(-2, -1)).copy())

    def rot_right(xy):
        return torch.tensor([-xy[1], xy[0]])
    def rot_left(xy):
        return torch.tensor([xy[1], -xy[0]])

    # Caution: coordinate format [y, x]
    act, std, log_std = network((state[None, :], t_x))
    act4, std4, log_std4 = network((rot_right(state)[None, :], t_r_x3))
    act3, std3, log_std3 = network((rot_right(rot_right(state))[None, :], t_r_x2))
    act2, std2, log_std2 = network((rot_right(rot_right(rot_right(state)))[None, :], t_r_x1))
    # print(t_x)
    # print(t_r_x3)
    # print(state, rot_right(state))
    # print(act, act4, act3, act2)
    assert((std >= 0).all() and (std4 >= 0).all() and (std3 >= 0).all() and (std2 >= 0).all())
    assert_similar([act[0], rot_left(act4[0]), rot_left(rot_left(act3[0])), rot_left(rot_left(rot_left(act2[0])))])
    assert_similar([std[0], torch.abs(rot_left(std4[0])), std3[0], torch.abs(rot_right(std2[0]))])
    return True

def test_toy2d_q(network=None, grid=None, state=None, action=None, num_channels=2):
    if grid is None:
        # frame = np.random.uniform(size=(1, 7 * 3, 7 * 3))
        grid = np.arange(num_channels * 7 * 7 * 3 * 3).reshape((1, num_channels, 7 * 3, 7 * 3)) / (7 * 7 * 3 * 3)
        grid[0, -6:, -6:] = 1
    if state is None:
        state = torch.tensor([0.1, 0.5])
    if action is None:
        action = torch.tensor([-0.2, 0.3])
    if network is None:
        network = Toy2DQNet(num_channels)

    # Note, that this rotates right whereas I rotate left.
    t_x = torch.Tensor(grid)
    t_r_x1 = torch.Tensor(np.rot90(grid.copy(), k=1, axes=(-2, -1)).copy())
    t_r_x2 = torch.Tensor(np.rot90(grid.copy(), k=2, axes=(-2, -1)).copy())
    t_r_x3 = torch.Tensor(np.rot90(grid.copy(), k=3, axes=(-2, -1)).copy())

    def rot_right(xy):
        return torch.tensor([-xy[1], xy[0]])

    # Caution: coordinate format [y, x]
    r = network((state[None, :], t_x),
                action[None, :])
    r4 = network((rot_right(state)[None, :], t_r_x3),
                 rot_right(action)[None, :])
    r3 = network((rot_right(rot_right(state))[None, :], t_r_x2),
                 rot_right(rot_right(action))[None, :])
    r2 = network((rot_right(rot_right(rot_right(state)))[None, :], t_r_x1),
                 rot_right(rot_right(rot_right(action)))[None, :])
    # print(t_x)
    # print(t_r_x3)
    # print(state, rot_right(state))
    # print(act, act4, act3, act2)
    assert_similar([r, r4, r3, r2])
    return True

def test_toy2d_decoder(network=None, state=None, action=None, latent=None, num_channels=5, channel=None):
    if channel is None:
        channel = np.random.randint(num_channels)
    if latent is None:
        latent = torch.tensor([-0.6, 0.1])
    if state is None:
        state = torch.tensor([0.1, 0.5])
    if action is None:
        action = torch.tensor([-0.2, 0.3])
    if network is None:
        network = Toy2DDecoder(num_channels)

    def rot_right(xy, k=1):
        if k == 0:
            return xy
        return rot_right(torch.tensor([-xy[1], xy[0]]), k=k-1)
    def rot_left(xy, k=1):
        if k == 0:
            return xy
        return rot_left(torch.tensor([xy[1], -xy[0]]), k=k-1)
    def build_input(s, a, c, l):
        res = torch.zeros(3 * num_channels)
        res[c] = 1
        res[num_channels + c * 2:num_channels + c * 2 + 2] = l
        return torch.cat((s, a, res))


    # Caution: coordinate format [y, x]
    r = network(build_input(state, action, channel, latent)[None, :])
    r4 = network(build_input(rot_right(state), rot_right(action), channel, rot_right(latent))[None, :])
    r3 = network(build_input(rot_right(state, k=2), rot_right(action, k=2), channel, rot_right(latent, k=2))[None, :])
    r2 = network(build_input(rot_right(state, k=3), rot_right(action, k=3), channel, rot_right(latent, k=3))[None, :])
    assert_similar([r, r4, r3, r2])
    return True

def test_metaworld_v1_mirrored_policy(network=None, latent=None, channel=None, num_channels=5, state=None):
    if channel is None:
        channel = np.random.randint(num_channels)
    if latent is None:
        latent = torch.normal(0, 3, size=(3, ))
    if state is None:
        state = torch.normal(0, 3, size=(3 * 4, ))
    if network is None:
        network = MetaWorldv1Policy(num_channels)
    reflect = reflect_metaworld_v1_mirrored

    obs = torch.cat((state, latent_hot(channel, latent, num_channels)))[None, :]
    obs_mirrored = torch.cat((reflect(state), latent_hot(channel, reflect(latent), num_channels)))[None, :]

    # Note: mean_out, log_std_out and std_out are the direct outputs of the equivariant network so there is no need to
    # test it without TanhGaussianPolicy wrapper

    policy = TanhGaussianPolicy(network)
    torch.manual_seed(1)
    act_out, mean_out, log_std_out, log_prob_out, expected_log_prob_out, std_out, mean_action_log_prob_out,\
        pre_tanh_value_out = policy(obs)
    torch.manual_seed(1)
    act_outm, mean_outm, log_std_outm, log_prob_outm, expected_log_prob_outm, std_outm, mean_action_log_prob_outm, \
        pre_tanh_value_outm = policy(obs_mirrored)

    # Testing network
    assert((std_out >= 0).all() and (std_outm >= 0).all())
    assert_similar([mean_out[0], reflect(mean_outm[0])])
    assert_similar([std_out[0], std_outm[0]])

    # Testing whether TanhGaussianPolicy preserves equivariance
    # Note: the random part X~N(0,1) of (mean + std * X) is symmetric around 0. Therefore, it is completely fine that we
    # also need to flip this component in order to get the flipped action (as the probability for flipped X is the same)
    # However, when checking for equivariance, this means that just setting the same seed isn't enough, we need to
    # mirror X as described
    print(mean_out, act_out, act_outm)
    # the random part added to the first vector entry (which should be mirrored) BEFORE tanh
    X_1 = torch.atanh(act_out)[0, 0] - mean_out[0, 0]
    # Needs to be subtracted twice BEFORE tanh to reverse the random influence
    act_outm[0, 0] = torch.tanh(torch.atanh(act_outm[0, 0]) - 2 * X_1)
    print(act_out, act_outm)
    assert_similar([act_out[0], reflect(act_outm[0])])
    return True

def pretty_tensor(t):
    return "[" + ", ".join(['%.2f' % elem for elem in t]) + "]"

def test_metaworld_v1_mirrored_qnet_training(num_classes=3, num_channels=2, baseline=False, M=100, monitor_tensors=False):
    alpha = 0.0  # (deactivated) 0.95 #  arbitrary, real algorithm uses learned alpha and 1.0 as default
    discount = 0.99  # default
    soft_target_tau = 5e-3  # 5e-3 (not configurable)
    soft_update_steps = 1  # (not configurable)
    num_policy_steps = 5000  # 5000  # MetaWorld reach config
    batch_size_policy = 256  # default (also for MetaWorld)
    if baseline:
        qf = FlattenMlp(
            input_size=(4 * 3 + (3 + 1) * num_channels) + 4,
            output_size=1,
            hidden_sizes=[M, M, M],
        ).to(ptu.device)
    else:
        M = int(M / np.sqrt(2))
        qf = MetaWorldv1QNet(num_channels, hidden_sizes=[M, M, M]).to(ptu.device)
    summarize_network(qf)
    qf_optimizer = torch.optim.Adam(qf.parameters(), 3e-4)  # =3e-4
    qf_criterion = torch.nn.MSELoss()
    if USE_WANDB:
        pass
        # wandb.watch(qf1, criterion=qf_criterion, log='all', log_freq=100)

    def gen_data_batch(n):
        terminals = (ptu.rand(n, 1) < (1./200)).float()
        y = ptu.randint(num_channels * num_classes, (n, ))  # verified
        goals = ptu.rand(n, 3) * 10 - 5  # goals in [-5, 5)
        z = goals * 2  # arbitrary transformation z in [-10, 10)
        obs = ptu.rand(n, 4 * 3)

        z_in = to_latent_hot(y, z, num_classes, num_channels)
        obs_z = torch.cat((obs, z_in), dim=1)

        actions = torch.tanh(ptu.randn(n, 4))  # tanh makes sure they're actually in the range of possible actions

        next_obs = obs.clone().detach()
        next_obs[:, :3] += actions[:, :3]
        # note, that we use the same z here, like the sac implementation does
        next_obs_z = torch.cat((next_obs, z_in.clone().detach()), dim=1)
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)

        # first 3 inputs are position here
        # rewards = compute_reach_rewards(next_obs[:, :3], goals)
        rewards = -torch.linalg.norm(next_obs[:, :3] - goals, dim=1, keepdim=True)

        return obs_z, actions, rewards, next_obs_z, goals, terminals

    def calc_q_value(obs, act, goals):
        """
        This is not the expected sum of rewards but anything that represents the "value" of a state-action pair should
        work here.
        """
        return -torch.linalg.norm(obs[:, :3] + act[:, :3] - goals, dim=1, keepdim=True)

    def training_step(epoch, obs, actions, rewards, next_obs, goals, terminals):
        q_pred = qf(obs, actions.detach())
        qf_loss = qf_criterion(q_pred, calc_q_value(obs, actions, goals))

        if USE_WANDB and epoch % 1000 == 0:
            wandb.log({"qf_loss": qf_loss,
                       "q_avg": q_pred.mean().item()})
        # Test and Logging
        # If I get this right, the order here is important as it zeros the grad for qf and ensures that the
        # policy loss doesn't backpropagate through qf even though gradients should be accumulated there when
        # calculating qf_new_actions
        qf_optimizer.zero_grad()
        qf_loss.backward()
        qf_optimizer.step()
        if epoch % 1000 == 0:
            print('Step ', i, ': qf loss: ', qf_loss.item())

    obs_test, act_test, r_test, next_obs_test, goals_test, terminals_test = gen_data_batch(10)
    terminals_test[:1, :] = 1  # so we are guaranteed to also see 2 examples of terminals
    obs, actions, rewards, next_obs, goals, terminals = obs_test, act_test, r_test, next_obs_test, goals_test, terminals_test
    known_tensors = set()
    for i in range(20):
        for epoch in range(num_policy_steps):  # These are policy epochs that happen for each epoch of CEMRL
            # as samples can also be from old version of policy and some random subset is taken for each epoch, random
            # sampling makes more sense thang passing the policy (which would effectively make it on-policy)
            # BUT: with time the percentage of data collected on a good policy should grow and converge to 1, so maybe
            # it makes sense to pass policy. Also, this might be necessary for qf to converge to sth. valid as the
            # actions by the policy are NOT random.
            obs, actions, rewards, next_obs, goals, terminals = gen_data_batch(batch_size_policy)
            training_step(epoch, obs, actions, rewards, next_obs, goals, terminals)

            if epoch % 1000 == 0:
                if monitor_tensors:
                    gc.collect()
                    cur_tensors = set()
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                cur_tensors.add((id(obj), hash(obj), str(type(obj)) + ": " + str(obj.size())))
                        except:
                            pass
                    print(len(known_tensors), "-->", len(cur_tensors), "diff:", len(cur_tensors) - len(known_tensors))
                    if len(cur_tensors) != len(known_tensors):
                        # Don't print if there are just some tensors created and deleted again (e.g. in the forward pass)
                        print("######## new tensors ########\n", "\n".join([t[2] for t in cur_tensors - known_tensors]))
                        print("######## deleted tensors ########\n", "\n".join([t[2] for t in known_tensors - cur_tensors]))
                    known_tensors = cur_tensors
                with torch.no_grad():
                    q_pred = qf(obs_test, act_test)
                    q_target = calc_q_value(obs_test, act_test, goals_test)
                    for j in range(10):  # range(obs_test.shape[0]):
                        print("s: ", pretty_tensor(obs_test[j, :3]),
                              ", z: ", pretty_tensor(obs_test[j, 3 * 4:]),
                              ", act: ", pretty_tensor(act_test[j, :]),
                              "(", pretty_tensor(goals_test[j, :] - obs_test[j, :3]),
                              "), r: %.2f" % r_test[j].item(),
                              ", s': ", pretty_tensor(next_obs_test[j, :3]),
                              ", q_pred: %.2f" % q_pred[j].item(),
                              "(is: %.2f" % q_target[j].item(), ")")
    if not baseline:
        test_metaworld_v1_mirrored_q(network=qf.to(torch.device("cpu")), num_channels=num_channels)
    return True


def test_metaworld_v1_mirrored_policy_training(num_classes=3, num_channels=2, baseline_q=False, baseline_policy=True, M=100,
                                               monitor_tensors=False):
    alpha = 0.0  # (deactivated) 0.95 #  arbitrary, real algorithm uses learned alpha and 1.0 as default
    discount = 0.99  # default
    soft_target_tau = 5e-3  # 5e-3 (not configurable)
    soft_update_steps = 1  # (not configurable)
    num_policy_steps = 5000  # 5000  # MetaWorld reach config
    batch_size_policy = 256  # default (also for MetaWorld)
    if baseline_policy:
        net = MlpPolicyNetwork(hidden_sizes=[M, M, M], obs_dim=(4 * 3 + (3 + 1) * num_channels), action_dim=4)
    else:
        net = MetaWorldv1Policy(num_channels, hidden_sizes=[int(M / np.sqrt(2)), int(M / np.sqrt(2)), int(M / np.sqrt(2))])
    if baseline_q:
        qf1 = FlattenMlp(
            input_size=(4 * 3 + (3 + 1) * num_channels) + 4,
            output_size=1,
            hidden_sizes=[M, M, M],
        ).to(ptu.device)
        qf2 = FlattenMlp(
            input_size=(4 * 3 + (3 + 1) * num_channels) + 4,
            output_size=1,
            hidden_sizes=[M, M, M],
        ).to(ptu.device)
        target_qf1 = FlattenMlp(
            input_size=(4 * 3 + (3 + 1) * num_channels) + 4,
            output_size=1,
            hidden_sizes=[M, M, M],
        ).to(ptu.device)
        target_qf2 = FlattenMlp(
            input_size=(4 * 3 + (3 + 1) * num_channels) + 4,
            output_size=1,
            hidden_sizes=[M, M, M],
        ).to(ptu.device)
    else:
        M = int(M / np.sqrt(2))
        qf1 = MetaWorldv1QNet(num_channels, hidden_sizes=[M, M, M]).to(ptu.device)
        qf2 = MetaWorldv1QNet(num_channels, hidden_sizes=[M, M, M]).to(ptu.device)
        target_qf1 = MetaWorldv1QNet(num_channels, hidden_sizes=[M, M, M]).to(ptu.device)
        target_qf2 = MetaWorldv1QNet(num_channels, hidden_sizes=[M, M, M]).to(ptu.device)
    summarize_network(net)
    summarize_network(qf1)
    policy = TanhGaussianPolicy(net).to(ptu.device)
    policy_optimizer = torch.optim.Adam(net.parameters(), 3e-4)  # lr=3e-4 (not configurable)
    qf1_optimizer = torch.optim.Adam(qf1.parameters(), 3e-4)  # =3e-4
    qf2_optimizer = torch.optim.Adam(qf2.parameters(), 3e-4)  # =3e-4
    qf_criterion = torch.nn.MSELoss()
    if USE_WANDB:
        pass
        # wandb.watch(policy, log='all', log_freq=100)
        # wandb.watch(qf1, criterion=qf_criterion, log='all', log_freq=100)

    def compute_reach_rewards(state, goal):
        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        maxReachDist = torch.linalg.norm(-goal, dim=1, keepdim=True)  # assume start at 0
        reachDist = torch.linalg.norm(state - goal, dim=1, keepdim=True)
        reachRew = c1 * (maxReachDist - reachDist) + c1 * (torch.exp(-(reachDist ** 2) / c2) + torch.exp(-(reachDist ** 2) / c3))
        reachRew = torch.maximum(reachRew, ptu.tensor(0))
        return reachRew / 1000  # CAUTION /1000 is not in the actual reward calculated but rather my way of coping with the large magnitude

    def gen_data_batch(n, policy=None):
        terminals = (ptu.rand(n, 1) < (1./200)).float()
        y = ptu.randint(num_channels * num_classes, (n, ))  # verified
        goals = ptu.rand(n, 3) * 10 - 5  # goals in [-5, 5)
        z = goals * 2  # arbitrary transformation z in [-10, 10)
        obs = ptu.rand(n, 4 * 3)

        z_in = to_latent_hot(y, z, num_classes, num_channels)
        obs_z = torch.cat((obs, z_in), dim=1)

        if policy is None:
            actions = torch.tanh(ptu.randn(n, 4))  # tanh makes sure they're actually in the range of possible actions
        else:
            with torch.no_grad():
                actions, *_ = policy(obs_z)
            assert(len(actions.shape) == 2 and actions.shape[0] == n and actions.shape[1] == 4)

        next_obs = obs.clone().detach()
        next_obs[:, :3] += actions[:, :3]
        # note, that we use the same z here, like the sac implementation does
        next_obs_z = torch.cat((next_obs, z_in.clone().detach()), dim=1)
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)

        # first 3 inputs are position here
        rewards = compute_reach_rewards(next_obs[:, :3], goals)
        # rewards = -torch.linalg.norm(next_obs[:, :3] - goals, dim=1, keepdim=True)

        return obs_z, actions, rewards, next_obs_z, goals, terminals

    def calc_q_value(obs, act, goals):
        """
        This is not the expected sum of rewards but anything that represents the "value" of a state-action pair should
        work here.
        """
        return -torch.linalg.norm(obs[:, :3] + act[:, :3] - goals, dim=1, keepdim=True)

    def training_step(epoch, obs, actions, rewards, next_obs, goals, terminals):
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = policy(obs, reparameterize=True,
                                                                          return_log_prob=True, )
        q_new_actions = torch.min(qf1(obs, new_obs_actions),
                                  qf2(obs, new_obs_actions))
        # q_new_actions = calc_q_value(obs, new_obs_actions, goals)
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        q1_pred = qf1(obs, actions.detach())
        q2_pred = qf2(obs, actions.detach())
        new_next_actions, _, _, new_log_pi, *_ = policy(next_obs, reparameterize=True, return_log_prob=True, )
        target_q_values = torch.min(target_qf1(next_obs, new_next_actions),
                                    target_qf2(next_obs, new_next_actions)) - alpha * new_log_pi

        q_target = rewards + (1 - terminals) * discount * target_q_values  # reward_scale is 1 anyway
        qf1_loss = qf_criterion(q1_pred, q_target.detach())
        qf2_loss = qf_criterion(q2_pred, q_target.detach())

        if USE_WANDB and epoch % 1000 == 0:
            wandb.log({"policy_loss": policy_loss, "qf1_loss": qf1_loss, "qf2_loss": qf2_loss,
                       "q1_avg": q1_pred.mean().item(), "q2_avg": q2_pred.mean().item(),
                       "q_target_avg": q_target.mean().item()})
        # Test and Logging

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        # If I get this right, the order here is important as it zeros the grad for qf and ensures that the
        # policy loss doesn't backpropagate through qf even though gradients should be accumulated there when
        # calculating qf_new_actions
        qf1_optimizer.zero_grad()
        qf1_loss.backward()
        qf1_optimizer.step()
        qf2_optimizer.zero_grad()
        qf2_loss.backward()
        qf2_optimizer.step()

        if epoch % soft_update_steps == 0:
            ptu.soft_update_from_to(
                qf1, target_qf1, soft_target_tau
            )
            ptu.soft_update_from_to(
                qf2, target_qf2, soft_target_tau
            )
        if epoch % 1000 == 0:
            print('Step ', i, ': policy loss: ', policy_loss.item(), 'qf1 loss: ', qf1_loss.item(),
                  'qf2 loss: ', qf2_loss.item())

    obs_test, act_test, r_test, next_obs_test, goals_test, terminals_test = gen_data_batch(10)
    terminals_test[:1, :] = 1  # so we are guaranteed to also see 2 examples of terminals
    obs, actions, rewards, next_obs, goals, terminals = obs_test, act_test, r_test, next_obs_test, goals_test, terminals_test
    known_tensors = set()
    for i in range(500):
        for epoch in range(num_policy_steps):  # These are policy epochs that happen for each epoch of CEMRL
            # as samples can also be from old version of policy and some random subset is taken for each epoch, random
            # sampling makes more sense thang passing the policy (which would effectively make it on-policy)
            # BUT: with time the percentage of data collected on a good policy should grow and converge to 1, so maybe
            # it makes sense to pass policy. Also, this might be necessary for qf to converge to sth. valid as the
            # actions by the policy are NOT random.
            obs, actions, rewards, next_obs, goals, terminals = gen_data_batch(batch_size_policy, policy=policy)
            training_step(epoch, obs, actions, rewards, next_obs, goals, terminals)

            if epoch % 1000 == 0:
                if monitor_tensors:
                    gc.collect()
                    cur_tensors = set()
                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                cur_tensors.add((id(obj), hash(obj), str(type(obj)) + ": " + str(obj.size())))
                        except:
                            pass
                    print(len(known_tensors), "-->", len(cur_tensors), "diff:", len(cur_tensors) - len(known_tensors))
                    if len(cur_tensors) != len(known_tensors):
                        # Don't print if there are just some tensors created and deleted again (e.g. in the forward pass)
                        print("######## new tensors ########\n", "\n".join([t[2] for t in cur_tensors - known_tensors]))
                        print("######## deleted tensors ########\n", "\n".join([t[2] for t in known_tensors - cur_tensors]))
                    known_tensors = cur_tensors
                with torch.no_grad():
                    q1_pred = qf1(obs_test, act_test)
                    q2_pred = qf1(obs_test, act_test)
                    p_act, _, _, _, *_ = policy(obs_test, reparameterize=False, return_log_prob=False, )
                    new_next_actions, _, _, new_log_pi, *_ = policy(next_obs_test, reparameterize=True,
                                                                    return_log_prob=True, )
                    target_q_values = torch.min(target_qf1(next_obs_test, new_next_actions),
                                                target_qf2(next_obs_test, new_next_actions)) - alpha * new_log_pi
                    for j in range(10):  # range(obs_test.shape[0]):
                        print("s: ", pretty_tensor(obs_test[j, :3]),
                              ", z: ", pretty_tensor(obs_test[j, 3 * 4:]),
                              ", act: ", pretty_tensor(act_test[j, :]),
                              ", policy_act: ", pretty_tensor(p_act[j, :]),
                              "(", pretty_tensor(goals_test[j, :] - obs_test[j, :3]),
                              "), r: %.2f" % r_test[j].item(),
                              ", s': ", pretty_tensor(next_obs_test[j, :3]),
                              ", q1_pred: %.2f" % q1_pred[j].item(),
                              ", q2_pred: %.2f" % q2_pred[j].item(),
                              "(is: %.2f" % (r_test[j] + (1 - terminals_test[j]) * discount * target_q_values[j]).item(),
                              "), terminal:", terminals_test[j].item())
                    del q1_pred, q2_pred, p_act, new_next_actions, new_log_pi, target_q_values
    if not baseline_policy:
        test_metaworld_v1_mirrored_policy(network=net.to(torch.device("cpu")), num_channels=num_channels)
    return True

def test_metaworld_v1_mirrored_q(network=None, latent=None, channel=None, num_channels=5, state=None, action=None):
    if channel is None:
        channel = np.random.randint(num_channels)
    if latent is None:
        latent = torch.normal(0, 3, size=(3, ))
    if state is None:
        state = torch.normal(0, 3, size=(4 * 3, ))
    if action is None:
        action = torch.normal(0, 3, size=(3 + 1, ))
    if network is None:
        network = MetaWorldv1QNet(num_channels)

    reflect = reflect_metaworld_v1_mirrored

    r = network(torch.cat((state, latent_hot(channel, latent, num_channels)))[None, :], action[None, :])
    r1 = network(torch.cat((reflect(state), latent_hot(channel, reflect(latent), num_channels)))[None, :],
                 reflect(action)[None, :])
    assert_similar([r, r1])
    return True

def test_metaworld_v1_mirrored_decoder(network=None, state=None, action=None, latent=None, num_channels=5, channel=None):
    if channel is None:
        channel = np.random.randint(num_channels)
    if latent is None:
        latent = torch.normal(0, 3, size=(3, ))
    if state is None:
        state = torch.normal(0, 3, size=(3 * 4, ))
    if action is None:
        action = torch.normal(0, 3, size=(3 + 1, ))
    if network is None:
        network = MetaWorldv1Decoder(num_channels)

    reflect = reflect_metaworld_v1_mirrored

    def build_input(s, a, c, l):
        return torch.cat((s, a, latent_hot(c, l, num_channels)))


    # Caution: coordinate format [y, x]
    r = network(build_input(state, action, channel, latent)[None, :])
    r1 = network(build_input(reflect(state), reflect(action), channel, reflect(latent))[None, :])
    assert_similar([r, r1])
    return True


def reflect_metaworld_v1_mirrored(y):
    temp = torch.ones_like(y)
    temp[::3] = -1
    if y.shape[0] == 4:
        temp[3] = 1  # prevent from mirroring torque in action
    return y * temp

def latent_hot(channel, z, num_channels):
    res = torch.zeros(num_channels + num_channels * z.shape[-1])
    res[channel] = 1
    res[num_channels + channel * z.shape[-1] : num_channels + channel * z.shape[-1] + z.shape[-1]] = z
    return res

def summarize_network(net, print_tensors=False):
    sum_trainable = 0
    sum_nontrainable = 0
    if print_tensors:
        print("#"*10 + " parameters " + "#"*10)
    for p in net.parameters():
        if print_tensors:
            print(("trainable: " if p.requires_grad else "non-trainable: ") + str(p.numel()))
        if p.requires_grad:
            sum_trainable += p.numel()
        else:
            sum_nontrainable += p.numel()
    print("parameters: total trainable: " + str(sum_trainable) + ", total nontrainable: " + str(sum_nontrainable))
    if print_tensors:
        print("#"*10 + " buffers " + "#"*10)
    for p in net.buffers():
        if print_tensors:
            print(("trainable: " if p.requires_grad else "non-trainable: ") + str(p.numel()))
        if p.requires_grad:
            sum_trainable += p.numel()
        else:
            sum_nontrainable += p.numel()
    print("buffers: total trainable: " + str(sum_trainable) + ", total nontrainable: " + str(sum_nontrainable))

# def test_group():
#     """
#     Test if the group is closed
#     """
#     group_repr = get_grid_rolls()
#     is_closed = closed_group(group_repr.representations)
#     assert(is_closed)
#     group_repr = get_grid_actions()
#     is_closed = closed_group(group_repr.representations)
#     assert(is_closed)
#     return True


def assert_permutations(permutations, out_x):
    """
    """
    x = out_x[0]
    for i in range(len(permutations)):
        p_i = permutations[i].float()
        x_i = out_x[i]
        pi_x = torch.matmul(p_i, x)
        assert(torch.allclose(pi_x, x_i))
    return True


def assert_similar(out_x):
    """
    Assert to test if invariance holds
    """
    x = out_x[0]
    for i in range(len(out_x)):
        x_i = out_x[i]
        if not torch.allclose(x, x_i, atol=1e-06):
            print(out_x)
            assert(False)
    return True

def test_toy2d():
    print("\n################# Testing Toy2D #################")
    print("Testing encoder network...")
    success = test_toy2d_encoder()
    print("Test passed:", success)
    print("Testing policy network...")
    success = test_toy2d_policy()
    print("Test passed:", success)
    print("\nTesting reward decoder network...")
    success = test_toy2d_decoder()
    print("Test passed:", success)
    print("\nTesting Q network...")
    success = test_toy2d_q()
    print("Test passed:", success)

def test_toy1d():
    print("\n################# Testing Toy1D #################")
    print("Testing encoder network...")
    success = test_toy1d_encoder()
    print("Test passed:", success)
    print("Testing policy network...")
    success = test_toy1d_policy()
    print("Test passed:", success)
    print("\nTesting reward decoder network...")
    success = test_toy1d_decoder()
    print("Test passed:", success)
    print("\nTesting Q network...")
    success = test_toy1d_q()
    print("Test passed:", success)

def test_metaworld_v1_mirrored():
    print("\n################# Testing MetaWorld v1 Mirrored #################")
    print("Testing policy network...")
    success = test_metaworld_v1_mirrored_policy()
    print("Test passed:", success)
    print("\nTesting reward decoder network...")
    success = test_metaworld_v1_mirrored_decoder()
    print("Test passed:", success)
    print("\nTesting Q network...")
    success = test_metaworld_v1_mirrored_q()
    print("Test passed:", success)

if __name__ == "__main__":
    if USE_WANDB:
        wandb.init(project="cemrl", name="equivariance-tests", group="equivariance-tests")
    ptu.set_gpu_mode(True, 0)
    print("="*25)
    print("Running equivariance/invariance tests...")
    print("="*25)
    # print("Testing if group is closed..")
    #success = test_group()
    #print("Test passed:", success)
    #print("\nTesting single conv layer...")
    #success = test_singlelayer()
    #print("Test passed:", success)
    test_toy2d_policy()
    # test_toy2d()
    # test_toy1d()
    # test_metaworld_v1_mirrored()
    # gpu_debugger.debug_method("training_step")
    # test_metaworld_v1_mirrored_policy_training(1, 1)
    # test_metaworld_v1_mirrored_qnet_training(1, 1)
    print("="*25)

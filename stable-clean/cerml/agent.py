import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu

from cerml.scripted_policies import policies
from cerml.utils import generate_latent_grid, to_latent_hot


class CEMRLAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy,
                 latent_probabilities,
                 use_latent_grid,
                 latent_grid_resolution,
                 latent_grid_range,
                 latent_dim,
                 all_channels_to_policy
                 ):
        super(CEMRLAgent, self).__init__()
        self.encoder = encoder
        self.prior_pz = prior_pz
        self.policy = policy
        self.latent_probabilities = latent_probabilities
        self.latent_grid_resolution = latent_grid_resolution
        self.latent_grid_range = latent_grid_range
        self.use_latent_grid = use_latent_grid
        self.latent_dim = latent_dim
        self.all_channels_to_policy = all_channels_to_policy

    # def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
    #     state = ptu.from_numpy(state).view(1, -1)
    #     if self.latent_probabilities:
    #         if z_debug is not None:
    #             raise NotImplementedError("z_debug cannot be used with probabilistic mode")
    #         y_distrs, z_distrs = self.encoder.encode(encoder_input)
    #         # pass mean and variance of zs for all possible ys instead of just one sampled z
    #         z = torch.cat([y_distrs.probs] + [torch.cat((d.mean, d.variance), dim=1) for d in z_distrs], dim=1)
    #     else:
    #         z, y = self.encoder(encoder_input)
    #         if self.use_latent_grid:
    #             z = to_latent_hot(y.long(), z, self.encoder.num_classes, self.encoder.num_channels)
    #         if z_debug is not None:
    #             z = z_debug
    #     policy_input = torch.cat([state, z], dim=1)
    #     return self.policy.get_action(policy_input, deterministic=deterministic), np_ify(z.clone().detach())[0, :]

    def get_actions(self, encoder_input, state, deterministic=False, spec_debug=None):
        if self.latent_probabilities:
            if spec_debug is not None:
                raise NotImplementedError()
            y_distrs, z_distrs = self.encoder.encode(encoder_input)
            z = torch.cat([y_distrs.probs] + [torch.cat((d.mean, d.variance), dim=1) for d in z_distrs], dim=1)
            # pass mean and variance of zs for all possible ys instead of just one sampled z
            if self.use_latent_grid:
                grid = generate_latent_grid(y_distrs.probs, z_distrs, self.encoder.num_channels,
                                            self.encoder.num_classes, self.latent_dim, self.latent_grid_resolution,
                                            self.latent_grid_range, self.all_channels_to_policy)
                policy_input = state, grid
                # import matplotlib
                # from matplotlib import pyplot as plt
                # matplotlib.use('TkAgg')
                # print(y_distrs.probs[0, :])
                # for distr in z_distrs:
                #     print(distr.mean[0, :], distr.stddev[0, :])
                # print(torch.sum(grid[0, :, :]))
                # plt.imshow(ptu.get_numpy(grid[0]))
                # plt.show()
                # exit()
            else:
                policy_input = torch.cat([state, z], dim=1)
        else:
            if spec_debug is not None:
                z, y = spec_debug
            else:
                z, y = self.encoder(encoder_input)

            if self.use_latent_grid:  # used e.g. for MetaWorld. Pass channel in addition to sampled z
                z_in = to_latent_hot(y.long(), z, self.encoder.num_classes, self.encoder.num_channels)
                policy_input = torch.cat([state, z_in], dim=1)
            else:
                policy_input = torch.cat([state, z], dim=1)
        return (self.policy.get_actions(policy_input, deterministic=deterministic), [{}] * state.shape[0]), np_ify(z)



class ScriptedPolicyAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy,
                 latent_probabilities,
                 all_channels_to_policy
                 ):
        super(ScriptedPolicyAgent, self).__init__()
        self.encoder = encoder
        self.prior_pz = prior_pz
        self.policy = policy
        self.latent_dim = encoder.latent_dim

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        env_name = env.active_env_name
        oracle_policy = policies[env_name]()
        action = oracle_policy.get_action(state)
        return (action.astype('float32'), {}), np.zeros(self.latent_dim, dtype='float32')

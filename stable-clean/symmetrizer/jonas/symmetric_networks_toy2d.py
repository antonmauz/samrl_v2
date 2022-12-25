import torch
import torch.nn.functional as F
import numpy as np

from symmetrizer.jonas.symmetric_ops_toy2d import get_xy_rolls_in, get_xy_rolls_out, get_xy_channel_rolls, \
    get_xy_positive_rolls
from symmetrizer.symmetrizer.nn.modules import BasisConv2d, GlobalAveragePool, \
    BasisLinear, GlobalMaxPool
from symmetrizer.symmetrizer.ops import c2g, get_grid_rolls_in, get_grid_rolls_out
from symmetrizer.symmetrizer.groups import P4, P4Intermediate, P4toOutput, P4toInvariant, MatrixRepresentation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu

SHOW_PLOTS = True
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Toy2DPolicy(torch.nn.Module):
    """
    Processing coordinates (x_1, x_2) along with a grid to equivariant coordinates (x_1', x_2') along with equivariant
    (switching axes but remaining positive) variances (s_1, s_2)
    Note that for roughly same amount of parameters, sizes should be divided by sqrt(4)
    input_size: number of input channels
    """

    def __init__(self, input_size, hidden_sizes=[128 * 2], channels=[8 * 2, 16 * 2],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0], xy_prep_size=2,
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()
        convs = []

        for l, channel in enumerate(channels):
            f = (filters[l], filters[l])
            s = strides[l]
            p = paddings[l]
            if l == 0:
                first_layer = True
            else:
                first_layer = False

            conv = BasisConv2d(input_size, channel, filter_size=f, group=in_group,
                               gain_type=gain_type,
                               basis=basis,
                               first_layer=first_layer, padding=p,
                               stride=s)
            convs.append(conv)
            input_size = channel
        self.convs = torch.nn.ModuleList(convs)
        self.pool = GlobalMaxPool()
        between_group = P4Intermediate()
        self.fc1 = BasisLinear(input_size + xy_prep_size, hidden_sizes[0], between_group,
                               gain_type=gain_type, basis=basis, bias_init=True)
        fc_layers = []
        for i in range(len(hidden_sizes) - 1):
            fc_layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1],
                                         between_group, gain_type=gain_type, basis=basis, bias_init=True))

        self.fc_layers = torch.nn.ModuleList(fc_layers)


        xy_in_group = MatrixRepresentation(get_xy_rolls_in(), get_grid_rolls_out())
        self.fc_xy_in = BasisLinear(1, xy_prep_size, xy_in_group, gain_type=gain_type, basis=basis,
                                    bias_init=True, bias=True)


        act_out_group = MatrixRepresentation(get_grid_rolls_in(), get_xy_rolls_out())
        self.fc_act = BasisLinear(hidden_sizes[-1], 1, act_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)
        std_out_group = MatrixRepresentation(get_grid_rolls_in(), get_xy_positive_rolls())
        self.fc_std = BasisLinear(hidden_sizes[-1], 1, std_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)

    def forward(self, obs):
        """
        Takes in coordinates of dimensions [batch_size, 2] and state of dimension [batch_size, num_channels or 1, height, width]
        """
        coordinates, grid = obs
        # if SHOW_PLOTS:
        #     plt.imshow(ptu.get_numpy(grid[0, 0]), vmin=0, vmax=1)
        #     plt.show()
        for i, c in enumerate(self.convs):
            grid = F.relu(c(grid))
            # if SHOW_PLOTS:
            #     fig, axs = plt.subplots(c.channels_out, c.repr_size_out)
            #     fig.tight_layout()
            #     for y in range(c.channels_out):
            #         for x in range(c.repr_size_out):
            #             axs[y, x].imshow(ptu.get_numpy(grid[0, y * c.repr_size_out + x]))  # , vmin=0, vmax=1)
            #     plt.show()

        conv_output = grid
        pool = c2g(self.pool(conv_output), 4).squeeze(-1).squeeze(-1)
        # plot_fc(pool, "max_pool")

        prep_coord = F.relu(self.fc_xy_in(coordinates[:, None, :]))  # Note, that x and y have to be the input CHANNELS
        # print(prep_coord)
        # print(pool)
        # plot_fc(prep_coord, "prep_coord")
        fc_output = F.relu(self.fc1(torch.cat((pool, prep_coord), dim=1)))
        for fc in self.fc_layers:
            fc_output = F.relu(fc(fc_output))
        # plot_fc(fc_output, "fc")
        action = self.fc_act(fc_output)[:, 0, :]
        log_std = self.fc_std(fc_output)[:, 0, :]
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        # print(action)
        return action, std, log_std


def plot_fc(fc_output, title):
    if not SHOW_PLOTS:
        return
    plt.imshow(ptu.get_numpy(fc_output[0]))
    plt.xlabel("representation")
    plt.ylabel("channel")
    plt.title(title)
    plt.show()


class Toy2DQNet(torch.nn.Module):
    """
    Processing coordinates (x_1, x_2) along with a grid to an invariant value
    """

    def __init__(self, input_size, hidden_sizes=[256, 128], channels=[16],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0], xy_prep_size=2,
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()
        out_1 = hidden_sizes[0]

        layers = []

        for l, channel in enumerate(channels):
            f = (filters[l], filters[l])
            s = strides[l]
            p = paddings[l]
            if l == 0:
                first_layer = True
            else:
                first_layer = False

            conv = BasisConv2d(input_size, channel, filter_size=f, group=in_group,
                               gain_type=gain_type,
                               basis=basis,
                               first_layer=first_layer, padding=p,
                               stride=s)
            layers.append(conv)
            input_size = channel
        self.convs = torch.nn.ModuleList(layers)

        self.pool = GlobalMaxPool()

        xy_in_group = MatrixRepresentation(get_xy_rolls_in(), get_grid_rolls_out())
        self.fc_xy_obs_in = BasisLinear(1, xy_prep_size * 2, xy_in_group, gain_type=gain_type, basis=basis,
                                        bias_init=True, bias=True)
        # self.fc_xy_act_in = BasisLinear(2, xy_prep_size, xy_in_group, gain_type=gain_type, basis=basis,
        #                                 bias_init=True, bias=True)

        between_group = P4Intermediate()
        layers = [BasisLinear(input_size + xy_prep_size * 2, out_1, between_group, gain_type=gain_type, basis=basis,
                              bias_init=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)
        inv_group = P4toInvariant()
        self.fc_inv = BasisLinear(hidden_sizes[-1], 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, obs, action):
        """
        obs: (state, latent_grid), where state: [batch_size, 2], latent_grid: [batch_size, height, width]
        action: [batch_size, 2]
        returns: reward: [batch_size, 1] (invariant)
        """
        coordinates, grid = obs
        for i, c in enumerate(self.convs):
            grid = F.relu(c(grid))

        conv_output = grid
        pool = c2g(self.pool(conv_output), 4).squeeze(-1).squeeze(-1)

        # Note, that x and y have to be the input CHANNELS
        prep_obs_act = F.relu(self.fc_xy_obs_in(torch.cat((coordinates[:, None, :], action[:, None, :]), dim=1)))
        # prep_act = F.relu(self.fc_xy_act_in(torch.cat((coordinates[:, None, :], action[:, None, :]), dim=1)))
        fc_output = torch.cat((pool, prep_obs_act), dim=1)
        for fc in self.layers:
            fc_output = F.relu(fc(fc_output))
        return self.fc_inv(fc_output)[:, 0, :]  # Get rid of output size in output


class Toy2DDecoder(torch.nn.Module):
    """
    Processing state, action and latent belief to an invariant reward prediction
    """

    def __init__(self, num_channels, hidden_sizes=[128, 64],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("At least one hidden layer required!")
        xy_in_group = MatrixRepresentation(get_xy_channel_rolls(num_channels), get_grid_rolls_out())
        between_group = P4Intermediate()
        inv_group = P4toInvariant()
        layers = [BasisLinear(1, hidden_sizes[0], xy_in_group, gain_type=gain_type,
                              basis=basis, bias_init=True, bias=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)
        self.out_layer = BasisLinear(hidden_sizes[-1], 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, x):
        """
        x0 axis: batch_size
        x1 axis: state (dim=2)
                + action (dim=2)
                + channel-one-hot (dim=num_channels)
                + "latent hot" encoding of z (z at the z position for the channel, 0 everywhere else)
                    (dim=latent_dim(=2)*num_channels)
        """
        old_shape = None
        if x.dim() > 2:
            old_shape = list(x.shape)
            x = x.reshape(-1, x.shape[-1])
        x = x[:, None, :]  # insert dimension for input size
        for layer in self.layers:
            x = F.relu(layer(x))
        res = self.out_layer(x)[:, 0, :]  # Remove dimension for output size in output
        if old_shape is not None:
            res = res.reshape(old_shape[:-1] + [1])
        return res

class BasisGridLayer(torch.nn.Module):
    """
    """

    def __init__(self, input_size, output_size, filter_size=(3, 3),
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()

        self.fc1 = BasisConv2d(input_size, output_size,
                               filter_size=filter_size, group=in_group,
                               gain_type=gain_type,
                               basis=basis,
                               first_layer=True)
        self.pool = GlobalAveragePool()

    def forward(self, state):
        """
        """
        return c2g(self.pool(self.fc1(state)), 4)

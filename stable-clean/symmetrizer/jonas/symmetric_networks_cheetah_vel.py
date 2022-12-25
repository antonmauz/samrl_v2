import torch
import torch.nn.functional as F
import numpy as np

from symmetrizer.jonas.symmetric_ops_toy1d import get_y_rolls, get_grid_1d_rolls
from symmetrizer.jonas.groupsC1 import C1, C1Intermediate, C1toInvariant, ToytoC1, C1toToy, ToyBelieftoC1
from symmetrizer.symmetrizer.nn.modules import BasisConv2d, GlobalAveragePool, \
    BasisLinear, GlobalMaxPool
from symmetrizer.symmetrizer.ops import c2g, get_grid_rolls
from symmetrizer.symmetrizer.groups import MatrixRepresentation
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu

SHOW_PLOTS = True
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class Toy1DPolicy(torch.nn.Module):
    """
    Processing coordinate (x_1, ) along with a grid to an equivariant coordinates (x_1', ) along with invariant
    variance (s_1, s_2)
    input_size: number of input channels
    TODO IS this also currently the network is only equivariant for an uneven amount of hidden layers?
    """

    def __init__(self, input_size, hidden_sizes=[128], channels=[8, 16],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0], y_prep_size=1,
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = C1()

        convs = []

        for l, channel in enumerate(channels):
            f = (1, filters[l])
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
        between_group = C1Intermediate()
        self.fc1 = BasisLinear(input_size + y_prep_size, hidden_sizes[0], between_group,
                               gain_type=gain_type, basis=basis, bias_init=True)
        fc_layers = []
        for i in range(len(hidden_sizes) - 1):
            fc_layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                         basis=basis, bias_init=True))

        self.fc_layers = torch.nn.ModuleList(fc_layers)

        y_in_group = ToytoC1()
        self.fc_y_in = BasisLinear(1, y_prep_size, y_in_group, gain_type=gain_type, basis=basis,
                                   bias_init=True, bias=True)

        act_out_group = C1toToy()
        self.fc_act = BasisLinear(hidden_sizes[-1], 1, act_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)
        std_out_group = C1toInvariant()
        self.fc_std = BasisLinear(hidden_sizes[-1], 1, std_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)

    def forward(self, obs):
        """
        Takes in coordinates of dimensions [batch_size, 1] and state of dimension [batch_size, 1, width]
        """
        coordinates, grid = obs
        grid = grid[:, None, :, :]  # insert dimension for group element
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
        pool = c2g(self.pool(conv_output), 2).squeeze(-1).squeeze(-1)
        # plot_fc(pool, "max_pool")

        prep_coord = F.relu(self.fc_y_in(coordinates[:, None, :]))  # Note, that x and y have to be the input CHANNELS
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


class Toy1DQNet(torch.nn.Module):
    """
    Processing coordinate (x_1) along with a grid to an invariant value
    """

    def __init__(self, input_size, hidden_sizes=[256, 128], channels=[16],
                 filters=[8, 5], strides=[1, 1], paddings=[0, 0], y_prep_size=1,
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = C1()

        out_1 = int(hidden_sizes[0] / np.sqrt(4))

        layers = []

        for l, channel in enumerate(channels):
            c = int(channel / np.sqrt(4))
            f = (1, filters[l])
            s = strides[l]
            p = paddings[l]
            if l == 0:
                first_layer = True
            else:
                first_layer = False

            conv = BasisConv2d(input_size, c, filter_size=f, group=in_group,
                               gain_type=gain_type,
                               basis=basis,
                               first_layer=first_layer, padding=p,
                               stride=s)
            layers.append(conv)
            input_size = c
        self.convs = torch.nn.ModuleList(layers)

        self.pool = GlobalMaxPool()

        y_in_group = ToytoC1()
        self.fc_y_obs_in = BasisLinear(1, y_prep_size, y_in_group, gain_type=gain_type, basis=basis,
                                        bias_init=True, bias=True)
        self.fc_y_act_in = BasisLinear(1, y_prep_size, y_in_group, gain_type=gain_type, basis=basis,
                                        bias_init=True, bias=True)

        between_group = C1Intermediate()
        self.fc1 = BasisLinear(input_size + y_prep_size * 2, out_1, between_group,
                               gain_type=gain_type,
                               basis=basis, bias_init=True)
        inv_group = C1toInvariant()
        self.fc_inv = BasisLinear(out_1, 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, obs, action):
        """
        Takes in coordinates of dimensions [batch_size, 1] and state of dimension [batch_size, 1, width]
        """
        coordinates, grid = obs
        grid = grid[:, None, :, :]  # insert dimension for group element
        for i, c in enumerate(self.convs):
            grid = F.relu(c(grid))

        conv_output = grid
        pool = c2g(self.pool(conv_output), 2).squeeze(-1).squeeze(-1)

        # Note, that x and y have to be the input CHANNELS
        prep_obs = F.relu(self.fc_y_obs_in(torch.cat((coordinates[:, None, :], action[:, None, :]), dim=1)))
        prep_act = F.relu(self.fc_y_act_in(torch.cat((coordinates[:, None, :], action[:, None, :]), dim=1)))
        fc_output = F.relu(self.fc1(torch.cat((pool, prep_obs, prep_act), dim=1)))
        return self.fc_inv(fc_output)[:, 0, :]  # Get rid of group representation in output


class Toy1DDecoder(torch.nn.Module):
    """
    Processing state, action and latent belief to an invariant reward prediction
    """

    def __init__(self, num_channels, hidden_sizes=[6], # hidden_sizes=[256, 128],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("At least one hidden layer required!")
        y_in_group = ToyBelieftoC1(num_channels)
        between_group = C1Intermediate()
        inv_group = C1toInvariant()
        layers = [BasisLinear(2 + num_channels * 2, hidden_sizes[0], y_in_group, gain_type=gain_type,
                              basis=basis, bias_init=True, bias=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)
        self.out_layer = BasisLinear(hidden_sizes[-1], 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, x):
        """
        x0 axis: batch_size
        x1 axis: state (dim=1)
                + action (dim=1)
                + channel-one-hot (dim=num_channels)
                + "latent hot" encoding of z (z at the z position for the channel, 0 everywhere else)
                    (dim=latent_dim(=1)*num_channels)
        """
        x = x[:, None, :]  # insert dimension for group element
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out_layer(x)[:, 0, :]  # Remove dimension for group representation in output

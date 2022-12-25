import torch.nn.functional as F

from symmetrizer.jonas.symmetric_gru import EquivariantGRUCell
from symmetrizer.jonas.groupsC1 import *
from symmetrizer.symmetrizer.nn.modules import BasisLinear
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu

SHOW_PLOTS = True
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MetaWorldv1Policy(torch.nn.Module):
    """
    TODO MetaWorldSampledPolicy (gets sampled latent), MetaWorldGridPolicy (gets grid),
     optionally: MetaWorldParameterPolicy (gets means/variances)
    Processing coordinate (x_1, ) along with a grid to an equivariant coordinates (x_1', ) along with invariant
    variance (s_1, )
    input_size: number of input channels
    """

    def __init__(self, sampled, num_channels, num_classes, hidden_sizes=[256, 128],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        between_group = C1Intermediate()
        in_group = MetaWorldObsBelieftoC1(num_channels) if sampled else MetaWorldObsParamstoC1(num_channels, num_classes)
        # 4 * 3 + 3
        layers = [BasisLinear(1, hidden_sizes[0], in_group, gain_type=gain_type,
                              basis=basis, bias_init=True, bias=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)

        act_out_group = C1toMetaWorldAct()
        self.fc_act = BasisLinear(hidden_sizes[-1], 1, act_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)
        std_out_group = C1toInvariant(4)
        self.fc_std = BasisLinear(hidden_sizes[-1], 1, std_out_group, gain_type=gain_type,
                                  basis=basis, bias_init=True, bias=False)

    def forward(self, obs):
        """
        if sampled:
            obs: [batch_size, state (4 * 3D) | z (num_channels + 3D * num_channels)]
            z is latent-hot
        otherwise:
            obs: [batch_size, state (4 * 3D) | z (num_channels * num_classes + (3D mean + 3D variance) * num_channels * num_classes)]
            z includes probabilities for all channels along with all classes
        """
        x = obs[:, None, :]
        for layer in self.layers:
            x = F.relu(layer(x))
        action = self.fc_act(x)[:, 0, :]  # Remove dimension for output size in output
        log_std = self.fc_std(x)[:, 0, :]  # Remove dimension for output size in output
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return action, std, log_std


def plot_fc(fc_output, title):
    if not SHOW_PLOTS:
        return
    plt.imshow(ptu.get_numpy(fc_output[0]))
    plt.xlabel("representation")
    plt.ylabel("channel")
    plt.title(title)
    plt.show()


class MetaWorldv1QNet(torch.nn.Module):

    def __init__(self, sampled, num_channels, num_classes, hidden_sizes=[128, 64],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("At least one hidden layer required!")
        y_in_group = MetaWorldObsActBelieftoC1(num_channels) if sampled else MetaWorldObsActParamstoC1(num_channels, num_classes)
        between_group = C1Intermediate()
        inv_group = C1toInvariant()
        # 4 * 3 + 3 + 3 + 1
        layers = [BasisLinear(1, hidden_sizes[0], y_in_group, gain_type=gain_type,
                              basis=basis, bias_init=True, bias=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)
        self.out_layer = BasisLinear(hidden_sizes[-1], 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, obs, action):
        """
        obs: [batch_size, state (4 * 3D) | z (num_channels + 3D * num_channels)]
        action: [batch_size, movement (3D), gripper torque (1D)]
        """
        x = torch.cat((obs[..., :4*3], action, obs[..., 4*3:]), dim=-1)[:, None, :]
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out_layer(x)[:, 0, :]  # Remove dimension for output size in output

class MetaWorldv1Decoder(torch.nn.Module):
    """
    Processing state, action and latent belief to an invariant reward prediction
    """
    def __init__(self, num_channels, hidden_sizes=[256, 128],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("At least one hidden layer required!")
        y_in_group = MetaWorldObsActBelieftoC1(num_channels)
        between_group = C1Intermediate()
        inv_group = C1toInvariant()
        # 4 * 3 + 4 + num_channels + num_channels * 3
        layers = [BasisLinear(1, hidden_sizes[0], y_in_group,
                              gain_type=gain_type, basis=basis, bias_init=True, bias=True)]
        for i in range(len(hidden_sizes) - 1):
            layers.append(BasisLinear(hidden_sizes[i], hidden_sizes[i + 1], between_group, gain_type=gain_type,
                                      basis=basis, bias_init=True))
        self.layers = torch.nn.ModuleList(layers)
        self.out_layer = BasisLinear(hidden_sizes[-1], 1, inv_group, gain_type=gain_type, basis=basis, bias_init=True)

    def forward(self, x):
        """
        x0 axis: batch_size
        x1 axis: state (4 * 3D position)
                + action (3D diff + 1D torque)
                + channel-one-hot (dim=num_channels)
                + "latent hot" encoding of z (z at the z position for the channel, 0 everywhere else)
                    (dim=latent_dim(=1)*num_channels)
        """
        old_shape = None
        if x.dim() > 2:
            old_shape = list(x.shape)
            x = x.reshape(-1, x.shape[-1])
        x = x[:, None, :]  # insert dimension for input size element
        for layer in self.layers:
            x = F.relu(layer(x))
        res = self.out_layer(x)[:, 0, :]  # Remove dimension for output size in output
        if old_shape is not None:
            res = res.reshape(old_shape[:-1] + [1])
        return res

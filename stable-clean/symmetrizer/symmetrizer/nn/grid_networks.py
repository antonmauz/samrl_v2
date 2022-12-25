import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from symmetrizer.jonas.symmetric_ops_toy2d import get_xy_rolls_out
from symmetrizer.symmetrizer.nn.modules import BasisConv2d, GlobalAveragePool, \
    BasisLinear, GlobalMaxPool
#from symmetrizer.jonas.symmetric_modules import XYContinuousBasisLayer
from symmetrizer.symmetrizer.ops import c2g, get_grid_rolls, GroupRepresentations
from symmetrizer.symmetrizer.groups import P4, P4Intermediate, P4toOutput, P4toInvariant, MatrixRepresentation
import matplotlib.pyplot as plt

SHOW_PLOTS = False

class BasisGridNetwork(torch.nn.Module):
    """
    """
    def __init__(self, input_size, hidden_sizes=[16], channels=[4, 6],# hidden_sizes=[512], channels=[16, 32],
                 filters=[8,5], strides=[1, 1], paddings=[0, 0],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()

        out_1 = int(hidden_sizes[0] / np.sqrt(4))

        layers = []

        for l, channel in enumerate(channels):
            c = int(channel/np.sqrt(4))
            f = (filters[l], filters[l])
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
        between_group = P4Intermediate()
        self.fc1 = BasisLinear(input_size, out_1, between_group,
                               gain_type=gain_type,
                               basis=basis, bias_init=True)
        out_group = P4toOutput()
        inv_group = P4toInvariant()
        self.fc4 = BasisLinear(out_1, 1, out_group, gain_type=gain_type,
                               basis=basis, bias_init=True)
        self.fc5 = BasisLinear(out_1, 1, inv_group,
                                gain_type=gain_type,
                                basis=basis, bias_init=True)

        xy_out_group = MatrixRepresentation(get_grid_rolls(), get_xy_rolls_out())
        self.fc_xy_action = BasisLinear(out_1, 1, xy_out_group, gain_type=gain_type, basis=basis,
                                          bias_init=True, bias=False)

    def forward(self, state):
        """
        """
        if SHOW_PLOTS:
            plt.imshow(state[0, 0], vmin=0, vmax=1)
            plt.show()
        outputs = []
        for i, c in enumerate(self.convs):
            outputs.append(state[0])
            state = F.relu(c(state))
            if SHOW_PLOTS:
                fig, axs = plt.subplots(c.channels_out, c.repr_size_out)
                fig.tight_layout()
                for y in range(c.channels_out):
                    for x in range(c.repr_size_out):
                        axs[y, x].imshow(state[0, y * c.repr_size_out + x].detach().numpy())#, vmin=0, vmax=1)
                plt.show()

        outputs.append(state[0])
        conv_output = state
        pool = c2g(self.pool(conv_output), 4).squeeze(-1).squeeze(-1)
        plot_fc(pool, "max_pool")
        outputs.append(pool)
        fc_output = F.relu(self.fc1(pool))
        plot_fc(fc_output, "fc")
        # print(fc_output[0])
        policy = self.fc4(fc_output)
        plot_fc(policy, "policy")
        value = self.fc5(fc_output)
        plot_fc(value, "value")
        cont_action = self.fc_xy_action(fc_output)
        print(cont_action)
        return policy, value

def plot_fc(fc_output, title):
    if not SHOW_PLOTS:
        return
    plt.imshow(fc_output[0].detach().numpy())
    plt.xlabel("representation")
    plt.ylabel("channel")
    plt.title(title)
    plt.show()

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


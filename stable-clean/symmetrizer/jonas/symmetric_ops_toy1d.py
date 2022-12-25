import numpy as np

from symmetrizer.symmetrizer.ops import GroupRepresentations
import rlkit.torch.pytorch_util as ptu

def get_grid_1d_rolls():
    """
    For C1
    """
    representations = [np.eye(2, dtype=np.float),
                       np.array([[0., 1], [1, 0]])]
    return GroupRepresentations(representations, "Grid1DRolls")


def get_y_rolls():
    """
    For Toy Goal 1D
    """
    representation = [np.array([[1.]]),
                      np.array([[-1.]])]
    return GroupRepresentations(representation, "YAction")

def get_y_channel_rolls(num_channels):
    """
    Transforms input [state | action | channel-one-hot | latent-hot] (for Toy Goal 1D)
    """
    representation = [np.eye(2 * 1 + num_channels + 1 * num_channels) for _ in range(2)]
    # The channel one-hot encoding is the only thing that should not be negated
    representation[1] *= -1
    representation[1][2:num_channels+2, :] *= -1  # Caution: this actually means that this area is NOT negative
    return GroupRepresentations(representation, "YActionChannel")

def get_y_trajectory_rolls():
    """
    Transforms input [observation (1D) | action (1D) | reward (1D) | next_observation (1D)]:
    (y, a, r, y') to (-y, -a, r, -y')
    """
    representation = [np.eye(4, dtype=np.float) for _ in range(2)]
    representation[1] *= -1
    representation[1][2, 2] = 1
    return GroupRepresentations(representation, "ToyTrajectory1D")

def get_lat_var_y_rolls():
    """
    Mirror lat with variances (y, vy) to (-y, vy)
    """
    representation = [np.eye(2, dtype=np.float) for _ in range(2)]
    representation[1][0, 0] = -1.0
    return GroupRepresentations(representation, "ToyLatVar1D")
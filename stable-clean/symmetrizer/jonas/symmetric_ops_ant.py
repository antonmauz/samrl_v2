from symmetrizer.symmetrizer.ops import GroupRepresentations
from rlkit.torch import pytorch_util as ptu
import numpy as np

def rot90_quaternion(q):
    """
    rotates a quaternion wxyz
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/steps/index.htm
    """
    assert(q.dim() == 1 and q.shape[0] == 4)
    return np.array([])

def get_xy_rolls_in():
    """
    Note, that a LEFT rotation of a 2D input shifts each representation to a higher index
    ("roll the representation right") due to torch.roll(). As get_grid_rolls() "rolls the representation right" for
    a rotation to the RIGHT, rotating the xy input LEFT (the matrix given here) should correspond to a grid_roll input.
    """
    representation = [np.array([[1., 0], [0, 1]]),
                      np.array([[0., -1], [1, 0]]),
                      np.array([[-1., 0], [0, -1]]),
                      np.array([[0., 1], [-1, 0]])]
    return GroupRepresentations(representation, "XYActionIn")

def get_xy_rolls_out():
    """
    The grid_rolls() shift each representation to a higher index ("roll the representation right") for a right rotation
    of the input. Therefore, the output for "rolling the representations to the right" should be rotating the action
    right.
    """
    representation = [np.array([[1., 0], [0, 1]]),
                      np.array([[0., 1], [-1, 0]]),
                      np.array([[-1., 0], [0, -1]]),
                      np.array([[0., -1], [1, 0]])]
    return GroupRepresentations(representation, "XYActionOut")


def get_xy_positive_rolls():
    representation = [np.array([[1., 0], [0, 1]]),
                      np.array([[0., 1], [1, 0]]),
                      np.array([[1., 0], [0, 1]]),
                      np.array([[0., 1], [1, 0]])]
    return GroupRepresentations(representation, "XYStd")


def get_xy_channel_rolls(num_channels):
    """
    Transforms input [state | action | channel-one-hot | latent-hot]
    """
    xy = get_xy_rolls_in()
    representation = [np.eye(2 * 2 + num_channels + 2 * num_channels) for _ in range(4)]
    for i, r in enumerate(representation):
        r[0:2, 0:2] = xy[i]
        r[2:4, 2:4] = xy[i]
        for c in range(num_channels):
            start = 2 * 2 + num_channels + 2 * c
            r[start:start+2, start:start+2] = xy[i]
    return GroupRepresentations(representation, "XYActionChannel")

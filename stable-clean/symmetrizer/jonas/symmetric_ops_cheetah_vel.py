import numpy as np

from symmetrizer.symmetrizer.ops import GroupRepresentations


def get_cheetah_vel_rolls():
    """
    For Cheetah Vel with observe_x=True
    """
    representation = [np.eye(9 + 3 + 9, dtype=float) for _ in range(2)]
    representation[1][np.array([]), :] *= -1
    return GroupRepresentations(representation, "CheetahVelAction")
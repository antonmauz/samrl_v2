import numpy as np

from symmetrizer.symmetrizer.ops import GroupRepresentations
import rlkit.torch.pytorch_util as ptu

# def get_v1_obs_lat_act_x_rolls():
#     """
#     Mirror [4 * 3D pos | 3D latent | 3D movement | 1D torque]
#     """
#     representation = [np.eye(4 * 3 + 3 + 3 + 1, dtype=np.float) for _ in range(2)]
#     representation[1][::3, :] *= -1.0
#     return GroupRepresentations(representation, "MetaWorldXObsLatAct")
#
# def get_v1_obs_lat_x_rolls():
#     """
#     Mirror [4 * 3D pos | 3D latent]
#     """
#     representation = [np.eye(4 * 3 + 3, dtype=np.float) for _ in range(2)]
#     representation[1][::3, :] *= -1.0
#     return GroupRepresentations(representation, "MetaWorldXObsLat")

def get_v1_act_x_rolls():
    """
    Mirror action (x, y, z, t) to (-x, y, z, t)
    """
    representation = [np.eye(4, dtype=np.float) for _ in range(2)]
    representation[1][0, 0] = -1.0
    return GroupRepresentations(representation, "MetaWorldXAction")

def get_v1_lat_var_x_rolls():
    """
    Mirror lat  with variances (x, y, z, vx, vy, vz) to (-x, y, z, vx, vy, vz)
    """
    representation = [np.eye(2 * 3, dtype=np.float) for _ in range(2)]
    representation[1][0, 0] = -1.0
    return GroupRepresentations(representation, "MetaWorldXLatVar")

def get_v1_obs_act_channel_rolls(num_channels):
    """
    Transforms input [state (4 3D positions) | action (3D pos diff, openness) | channel-one-hot | latent-hot] (for MetaWorld v1)
    """
    representation = [np.eye(4 * 3 + 4 + num_channels + 3 * num_channels, dtype=np.float) for _ in range(2)]
    representation[1][:(4+1)*3:3, :] *= -1
    representation[1][4 * 3 + 4 + num_channels::3, :] *= -1
    return GroupRepresentations(representation, "MetaWorldXObsActChannel")

def get_v1_obs_act_channel_params_rolls(num_channels, num_classes):
    """
    Transforms input [state (4 3D positions) | action (3D pos diff, openness) | (num_channels * num_classes + (3D mean + 3D variance) * num_channels * num_classes)]
    """
    representation = [np.eye(4 * 3 + 4 + num_channels * num_classes + (3 + 3) * num_channels * num_classes, dtype=np.float) for _ in range(2)]
    representation[1][:(4+1)*3:3, :] *= -1
    representation[1][4 * 3 + 4 + num_channels * num_classes::6, :] *= -1
    return GroupRepresentations(representation, "MetaWorldXObsActChannelParams")

def get_v1_obs_channel_rolls(num_channels):
    """
    Transforms input [state (4 3D positions) | channel-one-hot | latent-hot] (for MetaWorld v1)
    """
    representation = [np.eye(4 * 3 + num_channels + 3 * num_channels, dtype=np.float) for _ in range(2)]
    representation[1][:4*3:3, :] *= -1
    representation[1][4 * 3 + num_channels::3, :] *= -1
    return GroupRepresentations(representation, "MetaWorldXObsChannel")

def get_v1_obs_params_channel_rolls(num_channels, num_classes):
    """
    IMPORTANT: permuting means and variances is not included and happens in the corresponding Group
    input [state (4 * 3D) | z (num_channels * num_classes + (3D mean + 3D variance) * num_channels * num_classes)]
    structure: mean11 mean12 mean13 variance11 variance12 variance13 mean21 ...
    """
    representation = [np.eye(4 * 3 + num_channels * num_classes + (3 + 3) * num_channels * num_classes, dtype=np.float)
                      for _ in range(2)]
    representation[1][:4 * 3:3, :] *= -1
    representation[1][4 * 3 + num_channels * num_classes::6, :] *= -1
    for c in range(num_channels * num_classes // 2):
        representation[1][4 * 3 + 2 * c, 4 * 3 + 2 * c] = 0
        representation[1][4 * 3 + 2 * c, 4 * 3 + 2 * c + 1] = 1
        representation[1][4 * 3 + 2 * c + 1, 4 * 3 + 2 * c + 1] = 0
        representation[1][4 * 3 + 2 * c + 1, 4 * 3 + 2 * c] = 1

        # factor 12 from (|G|=2) * (3D + 3D)
        representation[1][12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c,
                          18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c] =\
            representation[1][12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c,
                              12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c]
        representation[1][18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c,
                          12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c] = \
            representation[1][18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c,
                              18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c]
        representation[1][12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c,
                          12 + num_channels * num_classes + 12 * c:18 + num_channels * num_classes + 12 * c] = 0
        representation[1][18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c,
                          18 + num_channels * num_classes + 12 * c:24 + num_channels * num_classes + 12 * c] = 0


    return GroupRepresentations(representation, "MetaWorldXObsParamsChannel")

def get_v1_trajectory_rolls():
    """
    Transforms input [observation (4 * 3D) | action (3D + 1D) | reward (1D) | next_observation (4 * 3D)]
    """
    representation = [np.eye(4 * 3 + 4 + 1 + 4 * 3, dtype=np.float) for _ in range(2)]
    representation[1][:4*3+3:3, :] *= -1  # (transform observation and action)
    representation[1][4*3+4+1::3, :] *= -1  # (transform next_observation)
    return GroupRepresentations(representation, "MetaWorldTrajectory")

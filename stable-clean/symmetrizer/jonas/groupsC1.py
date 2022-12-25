from symmetrizer.symmetrizer.groups import MatrixRepresentation
from symmetrizer.jonas.symmetric_ops_metaworld import *
from symmetrizer.jonas.symmetric_ops_toy1d import *
from symmetrizer.symmetrizer.groups import Group
from symmetrizer.symmetrizer.ops import *

class C1(Group):
    """
    Group of 1D reflections
    """
    def __init__(self):
        """
        """
        self.parameters = [i for i in range(2)]
        self.num_elements = 2
        self.__name__ = "C1 Group"

    def _input_transformation(self, weights, g):
        """
        """
        if g == 1:
            weights = np.flip(weights, axis=4)
            weights = np.roll(weights, 1, axis=2)
        return weights

    def _output_transformation(self, weights, g):
        """
        """
        weights = np.roll(weights, g, axis=1)
        return weights


class C1Intermediate(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=intermediate permutations
    """
    def __init__(self):
        super().__init__(get_grid_1d_rolls(), get_grid_1d_rolls())
        self.__name__ = "C1 Permutations"


class C1toInvariant(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=invariant representation
    """
    def __init__(self, out_dim=1):
        super().__init__(get_grid_1d_rolls(), [np.eye(out_dim), np.eye(out_dim)])
        self.__name__ = "C1 inv"

    def _output_transformation(self, weights, params):
        return super()._output_transformation(weights, params)


class C1toRolled(Group):

    def __init__(self, num_classes):
        if num_classes % 2 != 0:
            raise ValueError("The number of classes (given " + str(num_classes) +
                             ") must be a multiple of the number of group elements (here 2) in order for each of them"
                             " to have corresponding classes for all transformations!")
        self._input_matrices = get_grid_1d_rolls()
        self.repr_size_in = self._input_matrices[0].shape[1]
        self.repr_size_out = num_classes
        self.parameters = range(len(self._input_matrices))
        self.__name__ = "C1 class rolls"

    def _input_transformation(self, weights, params):
        weights = np.matmul(weights, self._input_matrices[params])
        return weights

    def _output_transformation(self, weights, params):
        shape = weights.shape
        return np.roll(weights.reshape(*(list(shape[:-2]) + [self.repr_size_out // 2, 2, shape[-1]])),
                       shift=params, axis=-2).reshape(*shape)


class C1toToy(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=transformed action (-y)
    """
    def __init__(self):
        """
        """
        super().__init__(get_grid_1d_rolls(), get_y_rolls())
        self.__name__ = "C1 Toy Out"

class C1toToyLatVar(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=transformed latent with variances (-y, vy)
    """
    def __init__(self):
        """
        """
        super().__init__(get_grid_1d_rolls(), get_lat_var_y_rolls())
        self.__name__ = "C1 Toy LatVarOut"

class ToytoC1(MatrixRepresentation):
    """
    C1 group representation
    in=transformed action (-y)
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        super().__init__(get_y_rolls(), get_grid_1d_rolls())
        self.__name__ = "C1 Toy In"


class ToyBelieftoC1(MatrixRepresentation):
    """
    C1 group representation
    in=latent belief (one-hot encoding of channel + latent-hot encoding of latent variable)
    out=intermediate permutations
    """
    def __init__(self, num_channels):
        """
        """
        super().__init__(get_y_channel_rolls(num_channels), get_grid_1d_rolls())
        self.__name__ = "C1 Toy Belief In"

class ToyTrajectoryEntrytoC1(MatrixRepresentation):
    """
    C1 group representation
    in=single trajectory entry (observation (1D) | action (1D) | reward (1D) | next_observation (1D))
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        super().__init__(get_y_trajectory_rolls(), get_grid_1d_rolls())
        self.__name__ = "C1 Toy Trajectory In"

# class MetaWorldObsLatActtoC1(MatrixRepresentation):
#     """
#     C1 group representation
#     in=[obs (4 * 3D)| z (3D) | act (3D movement, 1D torque)]
#     out=intermediate permutations
#     """
#     def __init__(self):
#         """
#         """
#         super().__init__(get_v1_obs_lat_act_x_rolls(), get_grid_1d_rolls())
#         self.__name__ = "C1 MetaWorld ObsLatAct In"
#
# class MetaWorldObsLattoC1(MatrixRepresentation):
#     """
#     C1 group representation
#     in=[obs (4 * 3D)| z (3D)]
#     out=intermediate permutations
#     """
#     def __init__(self):
#         """
#         """
#         super().__init__(get_v1_obs_lat_x_rolls(), get_grid_1d_rolls())
#         self.__name__ = "C1 MetaWorld ObsLat In"

class C1toMetaWorldAct(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=transformed action (-x, y, z, t)
    """
    def __init__(self):
        """
        """
        super().__init__(get_grid_1d_rolls(), get_v1_act_x_rolls())
        self.__name__ = "C1 MetaWorld ActOut"

class C1toMetaWorldLatVar(MatrixRepresentation):
    """
    C1 group representation
    in=intermediate permutations
    out=transformed latent with variances (-x, y, z, vx, vy, vz)
    """
    def __init__(self):
        """
        """
        super().__init__(get_grid_1d_rolls(), get_v1_lat_var_x_rolls())
        self.__name__ = "C1 MetaWorld LatVarOut"

class MetaWorldObsActBelieftoC1(MatrixRepresentation):
    """
    C1 group representation
    in=observation, action, latent belief (one-hot encoding of channel + latent-hot encoding of latent variable)
    out=intermediate permutations
    """
    def __init__(self, num_channels):
        """
        """
        super().__init__(get_v1_obs_act_channel_rolls(num_channels), get_grid_1d_rolls())
        self.__name__ = "C1 MetaWorld ObsActBelief In"

class MetaWorldObsActParamstoC1(MatrixRepresentation):
    def __init__(self, num_channels, num_classes):
        """
        """
        super().__init__(get_v1_obs_act_channel_params_rolls(num_channels, num_classes), get_grid_1d_rolls())
        self.__name__ = "C1 MetaWorld ObsActParams In"

class MetaWorldObsBelieftoC1(MatrixRepresentation):
    """
    C1 group representation
    in=observation, latent belief (one-hot encoding of channel + latent-hot encoding of latent variable)
    out=intermediate permutations
    """
    def __init__(self, num_channels):
        """
        """
        super().__init__(get_v1_obs_channel_rolls(num_channels), get_grid_1d_rolls())
        self.__name__ = "C1 MetaWorld ObsBelief In"

class MetaWorldObsParamstoC1(MatrixRepresentation):
    """
    C1 group representation
    in: state (4 * 3D) | z (num_channels * num_classes + (3D mean + 3D variance) * num_channels * num_classes)
    out=intermediate permutations
    """
    def __init__(self, num_channels, num_classes):
        """
        """
        self.num_classes = num_classes
        self.num_channels = num_channels
        super().__init__(get_v1_obs_params_channel_rolls(num_channels, num_classes), get_grid_1d_rolls())
        self.__name__ = "C1 MetaWorld ObsParams In"

class MetaWorldTrajectoryEntrytoC1(MatrixRepresentation):
    """
    C1 group representation
    in=single trajectory entry (observation (4 * 3D) | action (3D + 1D) | reward (1D) | next_observation (4 * 3D))
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        super().__init__(get_v1_trajectory_rolls(), get_grid_1d_rolls())
        self.__name__ = "C1 MetaWorld Trajectory In"
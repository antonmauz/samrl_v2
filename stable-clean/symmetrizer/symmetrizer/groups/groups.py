import numpy as np
from symmetrizer.symmetrizer.ops import *
from symmetrizer.jonas.symmetric_ops_toy2d import *

class Group:
    """
    Abstract group class
    """
    def __init__(self):
        """
        Set group parameters
        """
        raise NotImplementedError

    def _input_transformation(self, weights, transformation):
        """
        Specify input transformation. IMPORTANT: this is actually the INVERSE input transformation
        """
        raise NotImplementedError

    def _output_transformation(self, weights, transformation):
        """
        Specify output transformation
        """
        raise NotImplementedError


class MatrixRepresentation(Group):
    """
    Representing group elements as matrices
    """
    def __init__(self, input_matrices, output_matrices):
        """
        """
        self.repr_size_in = input_matrices[0].shape[1]
        self.repr_size_out = output_matrices[0].shape[1]
        self._input_matrices = input_matrices
        self._output_matrices = output_matrices

        self.parameters = range(len(input_matrices))

    def _input_transformation(self, weights, params):
        """
        Input transformation comes from the input group
        W F_g z
        """
        weights = np.matmul(weights, self._input_matrices[params])
        return weights

    def _output_transformation(self, weights, params):
        """
        Output transformation from the output group
        P_g W z
        """
        # print(self.__name__ if hasattr(self, "__name__") else "", weights.shape, self._output_matrices[params].shape)
        weights = np.matmul(self._output_matrices[params], weights)
        return weights


class P4(Group):
    """
    Group of 90 degree rotations
    """
    def __init__(self):
        self.parameters = [i for i in range(4)]
        self.num_elements = 4
        self.__name__ = "P4 Group"

    def _input_transformation(self, weights, angle):
        weights = np.rot90(weights, k=angle, axes=(3, 4))
        weights = np.roll(weights, angle, axis=2)
        return weights

    def _output_transformation(self, weights, angle):
        weights = np.roll(weights, angle, axis=1)
        return weights


class P4Intermediate(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = 4
        self.repr_size_in = 4
        self.__name__ = "P4 Permutations"
        self.in_permutations = get_grid_rolls_in()
        self.out_permutations = get_grid_rolls_out()

    def _input_transformation(self, weights, g):
        permute = self.in_permutations[g]
        weights = np.matmul(weights, permute)
        return weights

    def _output_transformation(self, weights, g):
        """
        """
        permute = self.out_permutations[g]
        weights = np.matmul(permute, weights)
        return weights


class P4toOutput(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=action permutations
    """
    def __init__(self):
        """
        """
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = 5
        self.repr_size_in = 4
        self.__name__ = "P4 Horizontal"
        self.in_permutations = get_grid_rolls_in()
        self.out_permutations = get_grid_actions()

    def _input_transformation(self, weights, g):
        permute = self.in_permutations[g]
        weights = np.matmul(weights, permute)
        return weights

    def _output_transformation(self, weights, g):
        permute = self.out_permutations[g]
        weights = np.matmul(permute, weights)
        return weights


class P4toInvariant(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=invariant representation
    """
    def __init__(self, out_dim=1):
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = out_dim
        self.repr_size_in = 4
        self.__name__ = "P4 inv"
        self.in_permutations = get_grid_rolls_in()
        self.out_permutations = [np.eye(out_dim), np.eye(out_dim), np.eye(out_dim), np.eye(out_dim)]

    def _input_transformation(self, weights, flip):
        permute = self.in_permutations[flip]
        weights = np.matmul(weights, permute)
        return weights

    def _output_transformation(self, weights, flip):
        permute = self.out_permutations[flip]
        # print(weights.shape)
        # print(permute.shape)
        # print(flip)
        weights = np.matmul(permute, weights)
        return weights


class P4toRolled(Group):

    def __init__(self, num_classes):
        if num_classes % 4 != 0:
            raise ValueError("The number of classes (given " + str(num_classes) +
                             ") must be a multiple of the number of group elements (here 4) in order for each of them"
                             " to have corresponding classes for all transformations!")
        self._input_matrices = get_grid_rolls_in()
        self.repr_size_in = self._input_matrices[0].shape[1]
        self.repr_size_out = num_classes
        self.parameters = range(len(self._input_matrices))
        self.__name__ = "P4 class rolls"

    def _input_transformation(self, weights, params):
        weights = np.matmul(weights, self._input_matrices[params])
        return weights

    def _output_transformation(self, weights, params):
        shape = weights.shape
        return np.roll(weights.reshape(*(list(shape[:-2]) + [self.repr_size_out // 4, 4, shape[-1]])), shift=params, axis=-2).reshape(*shape)


class ToyTrajectoryEntrytoP4(MatrixRepresentation):
    """
    in=single trajectory entry (observation (2D) | action (2D) | reward (1D) | next_observation (2D))
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        super().__init__(get_xy_trajectory_rolls(), get_grid_rolls_out())
        self.__name__ = "P4 Toy Trajectory In"


class P4toToyLatVar(MatrixRepresentation):
    """
    in=intermediate permutations
    out=transformed latent with variances (rot(y, x) | rot_positive(vy, vx))
    """
    def __init__(self):
        super().__init__(get_grid_rolls_in(), get_lat_var_xy_rolls())
        self.__name__ = "P4 Toy LatVarOut"
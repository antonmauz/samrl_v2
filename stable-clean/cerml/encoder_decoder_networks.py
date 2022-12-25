import numpy as np
import wandb
from torch.nn import GRU

import torch
from symmetrizer.symmetrizer.groups import P4Intermediate, ToyTrajectoryEntrytoP4, P4toToyLatVar, P4toRolled
from symmetrizer.jonas.groupsC1 import MetaWorldTrajectoryEntrytoC1, C1Intermediate, C1toInvariant, \
    C1toMetaWorldLatVar, C1toToyLatVar, ToyTrajectoryEntrytoC1, C1toRolled
from symmetrizer.jonas.symmetric_gru import EquivariantGRU
from symmetrizer.jonas.symmetric_networks_metaworld_v1 import MetaWorldv1Decoder
from symmetrizer.jonas.symmetric_networks_toy1d import Toy1DDecoder
from symmetrizer.jonas.symmetric_networks_toy2d import Toy2DDecoder
from symmetrizer.symmetrizer.nn.modules import BasisLayer, MultiBatchBasisLinear
from torch import nn as nn
import torch.nn.functional as F
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from cerml.utils import generate_gaussian, to_latent_hot


class ClassEncoder(nn.Module):
    def __init__(self,
                 num_classes,
                 shared_dim,
                 equivariance = "none"
    ):
        super(ClassEncoder, self).__init__()

        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.invariant = equivariance != "none"
        if self.invariant:
            if equivariance == "toy1D":
                self.linear = MultiBatchBasisLinear(shared_dim, 1, C1toRolled(num_classes))
            elif equivariance == "toy2D":
                self.linear = MultiBatchBasisLinear(shared_dim, 1, P4toRolled(num_classes))
            else:
                raise ValueError("The encoder equivariance \"" + equivariance + "\" was not recognized.")
        else:
            self.linear = nn.Linear(self.shared_dim, self.num_classes)

    def forward(self, m):
        res = F.softmax(self.linear(m), dim=-1)
        # remove output dimension for invariant
        return res[..., 0, :] if self.invariant else res


class PriorPz(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels,
                 latent_dim
                 ):
        super(PriorPz, self).__init__()
        self.latent_dim = latent_dim
        # feed cluster number y as one-hot, get mu_sigma out
        self.linear = nn.Linear(num_classes * num_channels, self.latent_dim * 2)

    def forward(self, m):
        """
        feeding a one-hot encoded vector to a linear layer is equivalent to learning arbitrary independent
        values for each output and each input class
        """
        return self.linear(m)


class EncoderMixtureModelTrajectory(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 num_channels,
                 time_steps,
                 merge_mode,
                 use_latent_grid,
                 latent_grid_resolution,
                 latent_grid_range,
                 restrict_latent_mean,
                 encoding_noise,
                 constant_encoding,
                 equivariance
    ):
        super(EncoderMixtureModelTrajectory, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.encoding_noise = encoding_noise
        self.constant_encoding = constant_encoding
        if num_channels != 1:
            raise NotImplementedError()
        if equivariance != "none":
            raise NotImplementedError("Equivariance is only implemented for the GRU encoder!")
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        """
        x.shape = [batch_size, time_steps * observation_size]
        """
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)  # shape: [batch_size, shared_dim = time_steps * observation_size * net_complex]

        # Compute class encoder forward pass
        y = self.class_encoder(m)
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random", batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.constant_encoding:  # For ablation only
            y = ptu.ones(y_distribution.probs.shape[0], dtype=torch.long) * (y if y_usage == "specific" else 1)
            z = ptu.zeros(y_distribution.probs.shape[0], self.latent_dim)
            return z, y

        # Select from which Gaussian to sample

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(batch_size, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        if self.encoding_noise != 0.0:
            z = z + self.encoding_noise * ptu.randn(z.shape)
        return z, y


class EncoderMixtureModelTransitionSharedY(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 num_channels,
                 time_steps,
                 merge_mode,
                 use_latent_grid,
                 latent_grid_resolution,
                 latent_grid_range,
                 restrict_latent_mean,
                 encoding_noise,
                 constant_encoding,
                 equivariance
    ):
        super(EncoderMixtureModelTransitionSharedY, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.encoding_noise = encoding_noise
        self.constant_encoding = constant_encoding
        if num_channels != 1:
            raise NotImplementedError()
        if equivariance != "none":
            raise NotImplementedError("Equivariance is only implemented for the GRU encoder!")
        self.time_steps = time_steps
        self.merge_mode = merge_mode
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        if self.merge_mode == 'linear':
            self.pre_class_encoder = nn.Linear(self.time_steps * self.shared_dim, shared_dim)
        elif self.merge_mode == 'mlp':
            self.pre_class_encoder = Mlp(hidden_sizes=[self.shared_dim], input_size=self.time_steps * self.shared_dim, output_size=shared_dim)
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x, return_distributions=False):
        y_distribution, z_distributions = self.encode(x)
        if return_distributions:
            return y_distribution, z_distributions
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        """
        x.shape = [n_tasks/batch_size, time_steps, observation_size]
        """
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)  # shape = [n_tasks/batch_size, time_steps, shared_dim]

        # Compute class encoder forward pass
        # Variant 1: Pre class encoder
        if self.merge_mode == 'linear' or self.merge_mode == 'mlp':
            flat = torch.flatten(m, start_dim=1)
            pre_class = self.pre_class_encoder(flat)
            y = self.class_encoder(pre_class)
        # Variant 2: Add logits
        elif self.merge_mode == "add":
            y = self.class_encoder(m)  # shape: [n_tasks/batch_size, time_steps, num_classes]
            y = y.sum(dim=-2) / y.shape[1]  # add the outcome of individual samples, scale down
        elif self.merge_mode == "add_softmax":
            y = self.class_encoder(m)  # shape: [n_tasks/batch_size, time_steps, num_classes]
            y = F.softmax(y.sum(dim=-2), dim=-1)  # add the outcome of individual samples, softmax
        # Variant 2: Multiply logits
        elif self.merge_mode == "multiply":
            y = self.class_encoder(m)  # shape: [n_tasks/batch_size, time_steps, num_classes]
            y = F.softmax(y.prod(dim=-2), dim=-1)  # multiply the outcome of individual samples

        # y.shape = [n_tasks/batch_size, num_classes]
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim, mode='multiplication') for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random",
                 batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.constant_encoding:  # For ablation only
            y = ptu.ones(y_distribution.probs.shape[0], dtype=torch.long) * (y if y_usage == "specific" else 1)
            z = ptu.zeros(y_distribution.probs.shape[0], self.latent_dim)
            return z, y
        # Select from which Gaussian to sample

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(batch_size, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)  # shape [batch_size, 1, latent_dim]

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].rsample(), 0) for i in range(self.num_classes)], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(z_distributions[i].mean, 0) for i in range(self.num_classes)], dim=0)

        # tensor with shape [batch, class, latent]
        permute = sampled.permute(1, 0, 2)
        z = torch.squeeze(torch.gather(permute, 1, mask), 1)
        if self.encoding_noise != 0.0:
            z = z + self.encoding_noise * ptu.randn(z.shape)
        return z, y


class EncoderMixtureModelTransitionIndividualY(nn.Module):
    '''
    Overall encoder network, putting a shared_encoder, class_encoder and gauss_encoder together.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 num_channels,
                 time_steps,
                 merge_mode,
                 use_latent_grid,
                 latent_grid_resolution,
                 latent_grid_range,
                 restrict_latent_mean,
                 encoding_noise,
                 constant_encoding,
                 equivariance
    ):
        super(EncoderMixtureModelTransitionIndividualY, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.encoding_noise = encoding_noise
        self.constant_encoding = constant_encoding
        if num_channels != 1:
            raise NotImplementedError()
        if equivariance != "none":
            raise NotImplementedError("Equivariance is only implemented for the GRU encoder!")
        self.time_steps = time_steps
        self.shared_encoder = Mlp(
                                    hidden_sizes=[shared_dim, shared_dim],
                                    input_size=encoder_input_dim,
                                    output_size=shared_dim,
                                )
        self.class_encoder = ClassEncoder(self.num_classes, self.shared_dim)
        self.gauss_encoder_list = nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2) for _ in range(self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class encoder forward pass
        y = self.class_encoder(m)
        y_distribution = torch.distributions.categorical.Categorical(probs=y)

        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma]

        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distribution, y_usage="specific", y=None, sampler="random", batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.constant_encoding:  # For ablation only
            y = ptu.ones(y_distribution.probs.shape[0], dtype=torch.long) * (y if y_usage == "specific" else 1)
            z = ptu.zeros(y_distribution.probs.shape[0], self.latent_dim)
            return z, y

        z_distributions = ptu.zeros((len(self.gauss_encoder_list), z_distribution[0].mean.shape[0], z_distribution[0].mean.shape[1], 2 * self.latent_dim))
        for i, dist in enumerate(z_distribution):
            z_distributions[i] = torch.cat((dist.mean, dist.scale), dim=-1)

        # Select from which Gaussian to sample
        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(batch_size, self.time_steps, dtype=torch.long) * y
        # Used while inference
        elif y_usage == "most_likely":
            y = torch.argmax(y_distribution.probs, dim=-1)
        else:
            raise RuntimeError("Sampling strategy not specified correctly")

        # values as [batch, timestep, class, latent_dim]
        mask = y.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 2 * self.latent_dim)

        # values as [batch, timestep, class, latent_dim]
        z_distributions_permuted = z_distributions.permute(1, 2, 0, 3)
        # gather at dim 2, which is the class dimension, selected by y
        mu_sigma = torch.squeeze(torch.gather(z_distributions_permuted, 2, mask), 2)

        gaussians = generate_gaussian(mu_sigma, self.latent_dim, sigma_ops=None, mode='multiplication')

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            # (diag_embed not possible for distributions), put it back to tensor with shape [class, batch, latent]
            z = gaussians.rsample()

        elif sampler == "mean":
            z = gaussians.mean

        y_return = torch.argmax(torch.prod(y_distribution.probs, dim=1), dim=-1)
        if self.encoding_noise != 0.0:
            z = z + self.encoding_noise * ptu.randn(z.shape)
        return z, y_return


class GaussEncoder(nn.Module):
    """
    Simple linear layer restricting the first half of the outputs to [-mean_range, mean_range] using a tanh function
    Variances are not restricted as the generate_gaussian() function handles that using softmax
    """
    def __init__(self, shared_dim, latent_dim, mean_range, restrict_latent_mean, equivariance_group):
        super(GaussEncoder, self).__init__()
        self.mean_range = mean_range
        self.restrict_latent_mean = restrict_latent_mean
        self.equivariance_group = equivariance_group
        if equivariance_group is None:
            self.linear = nn.Linear(shared_dim, latent_dim * 2)
        else:
            self.linear = MultiBatchBasisLinear(shared_dim, 1, equivariance_group)

    def forward(self, m):
        x = self.linear(m)
        if self.equivariance_group is not None:
            x = x[..., 0, :]  # remove output size
        if self.restrict_latent_mean:
            return torch.cat((torch.tanh(x[..., :x.shape[-1] // 2]) * self.mean_range, x[..., x.shape[-1] // 2:]),
                             dim=-1)
        else:
            return x

class EncoderMixtureModelGRU(nn.Module):
    '''
    Overall encoder network using a GRU to combine the trajectory and thereby give easy access to the already explored states.
    '''
    def __init__(self,
                 shared_dim,
                 encoder_input_dim,
                 latent_dim,
                 batch_size,
                 num_classes,
                 num_channels,
                 time_steps,
                 merge_mode,
                 use_latent_grid,
                 latent_grid_resolution,
                 latent_grid_range,
                 restrict_latent_mean,
                 encoding_noise,
                 constant_encoding,
                 equivariance
    ):
        super(EncoderMixtureModelGRU, self).__init__()
        self.shared_dim = shared_dim
        self.encoder_input_dim = encoder_input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes_init = num_classes
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.use_latent_grid = use_latent_grid
        self.latent_grid_resolution = latent_grid_resolution
        self.latent_grid_range = latent_grid_range
        self.encoding_noise = encoding_noise
        self.constant_encoding = constant_encoding
        self.equivariance = equivariance
        self.class_encoder = ClassEncoder(self.num_classes * self.num_channels, self.shared_dim, equivariance)
        if equivariance == "MetaWorldv1":
            # 4 * 3 + 4 + 1 + 4 * 3
            self.shared_encoder = EquivariantGRU(1, shared_dim, MetaWorldTrajectoryEntrytoC1(),
                                                 C1Intermediate())
            gauss_encoder_group = C1toMetaWorldLatVar()
        elif equivariance == "toy1D":
            self.shared_encoder = EquivariantGRU(1, shared_dim, ToyTrajectoryEntrytoC1(),
                                                 C1Intermediate())
            gauss_encoder_group = C1toToyLatVar()
        elif equivariance == "toy2D":
            self.shared_encoder = EquivariantGRU(1, shared_dim, ToyTrajectoryEntrytoP4(),
                                                 P4Intermediate())
            gauss_encoder_group = P4toToyLatVar()
        elif equivariance == "none":
            self.shared_encoder = GRU(input_size=encoder_input_dim, hidden_size=shared_dim, batch_first=True)
            gauss_encoder_group = None
        else:
            raise ValueError("The encoder equivariance \"" + equivariance + "\" does not exist.")
        self.gauss_encoder_list = nn.ModuleList([GaussEncoder(self.shared_dim, self.latent_dim,
                                                              self.latent_grid_range,
                                                              restrict_latent_mean, gauss_encoder_group)
                                                 for _ in range(self.num_channels * self.num_classes)])

    def forward(self, x):
        y_distribution, z_distributions = self.encode(x)
        # TODO: could be more efficient if only compute the Gaussian layer of the y that we pick later
        return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")

    def encode(self, x):
        """
        x: [batch_size, time_steps, observation | action | reward | next_observation] (in this order)
        y_distribution: holds the joint distribution over b * K + y (probs: [batch_size, num_channels * num_classes])
        z_distributions: list of num_channels * num_classes normal distributions
        """
        # Compute shared encoder forward pass
        _, m = self.shared_encoder(x)  # [1, batch_size, shared_dim]
        m = m[0, :, :]
        # print(m) # the equivariance here seems a little shady, maybe there is a problem with bias or sth similar?

        # Compute class encoder forward pass
        y = self.class_encoder(m)  # [batch_size, num_channels * num_classes]
        y_distribution = torch.distributions.categorical.Categorical(probs=y)  # p(y+b*K)

        # Compute every gauss_encoder forward pass
        all_mu_sigma = []
        for net in self.gauss_encoder_list:
            all_mu_sigma.append(net(m))
        z_distributions = [generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma]
        return y_distribution, z_distributions

    def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random", batch_size=None):
        """
        expects the input y to actually be y + num_classes * b (where b is the channel)
        """
        if batch_size is None:
            batch_size = self.batch_size
        # Select from which Gaussian to sample

        if self.constant_encoding:  # For ablation only
            y = ptu.ones(y_distribution.probs.shape[0], dtype=torch.long) * (y if y_usage == "specific" else 1)
            z = ptu.zeros(y_distribution.probs.shape[0], len(z_distributions))
            return z, y

        # Used for individual sampling when computing ELBO
        if y_usage == "specific":
            y = ptu.ones(batch_size, dtype=torch.long) * y  # [batch_size]
        # Used while inference
        elif y_usage == "most_likely":
            # y_distribution.probs.shape = [batch_size, num_channels * num_classes]
            y = torch.argmax(y_distribution.probs, dim=-1)  # [batch_size]
        else:
            raise RuntimeError("Sampling strategy not specified correctly")
        # add offset of [0, num_classes, 2*num_classes, ...] in order for the flattened gathering to be correct:
        mask = y.view(-1, 1).unsqueeze(2).repeat(1, 1, self.latent_dim)  # shape [batch_size, 1, latent_dim]

        if sampler == "random":
            # Sample from specified Gaussian using reparametrization trick
            # this operation samples from each Gaussian for every class first
            sampled = torch.cat([torch.unsqueeze(dist.rsample(), 0) for dist in z_distributions], dim=0)

        elif sampler == "mean":
            sampled = torch.cat([torch.unsqueeze(dist.mean, 0) for dist in z_distributions], dim=0)

        permute = sampled.permute(1, 0, 2)  # tensor with shape [batch, channels * classes, latent]
        z = torch.gather(permute, 1, mask)[:, 0, :]  # [batch, latent]
        if self.encoding_noise != 0.0:
            z = z + self.encoding_noise * ptu.randn(z.shape)
        return z, y


    # Legacy: this was meant for multiple features on multiple channels. However, I noticed we need different channels
    # for different discrete tasks
    # class EncoderMixtureModelGRU(nn.Module):
    #     '''
    #     Overall encoder network using a GRU to combine the trajectory and thereby give easy access to the already explored states.
    #     '''
    #
    #     def __init__(self,
    #                  shared_dim,
    #                  encoder_input_dim,
    #                  latent_dim,
    #                  batch_size,
    #                  num_classes,
    #                  num_channels,
    #                  time_steps,
    #                  merge_mode
    #                  ):
    #         super(EncoderMixtureModelGRU, self).__init__()
    #         self.shared_dim = shared_dim
    #         self.encoder_input_dim = encoder_input_dim
    #         self.latent_dim = latent_dim
    #         self.batch_size = batch_size
    #         self.num_classes_init = num_classes
    #         self.num_classes = num_classes
    #         self.num_channels = num_channels
    #         self.shared_encoder = GRU(input_size=encoder_input_dim, hidden_size=shared_dim, batch_first=True)
    #         self.class_encoder = ClassEncoder(self.num_classes * self.num_channels, self.shared_dim)
    #         self.gauss_encoder_list = nn.ModuleList([nn.ModuleList([nn.Linear(self.shared_dim, self.latent_dim * 2)
    #                                                                 for _ in range(self.num_classes)])
    #                                                  for _ in range(self.num_channels)])
    #
    #     def forward(self, x):
    #         y_distribution, z_distributions = self.encode(x)
    #         return self.sample_z(y_distribution, z_distributions, y_usage="most_likely", sampler="mean")
    #
    #     def encode(self, x):
    #         """
    #         y_distribution: holds num_channels independent categorical distributions whereas
    #         z_distributions: list of num_channels lists each holding num_classes normal distributions
    #         """
    #         # Compute shared encoder forward pass
    #         m = self.shared_encoder(x)  # [batch_size, time_steps, shared_dim]
    #         m = m[:, -1, :]  # Only use GRU output of last unit -> [batch_size, shared_dim]
    #
    #         # Compute class encoder forward pass
    #         y = self.class_encoder(m)  # [batch_size, num_channels * num_classes]
    #         y = y.reshape((-1, self.num_channels, self.num_classes))
    #         y_distribution = torch.distributions.categorical.Categorical(
    #             probs=y)  # gives one discrete value per channel
    #
    #         # Compute every gauss_encoder forward pass
    #
    #         z_distributions = []
    #         for list_channel in self.gauss_encoder_list:
    #             all_mu_sigma = []
    #             for net in list_channel:
    #                 all_mu_sigma.append(net(m))
    #             z_distributions.append([generate_gaussian(mu_sigma, self.latent_dim) for mu_sigma in all_mu_sigma])
    #         return y_distribution, z_distributions
    #
    #     def sample_z(self, y_distribution, z_distributions, y_usage="specific", y=None, sampler="random",
    #                  batch_size=None):
    #         if batch_size is None:
    #             batch_size = self.batch_size
    #         # Select from which Gaussian to sample
    #
    #         # Used for individual sampling when computing ELBO
    #         if y_usage == "specific":
    #             y = ptu.ones((self.batch_size, self.num_channels), dtype=torch.long) * y  # [batch_size, num_channels]
    #         # Used while inference
    #         elif y_usage == "most_likely":
    #             # y_distribution.probs.shape = [batch_size, num_channels, num_classes]
    #             y = torch.argmax(y_distribution.probs, dim=-1)  # [batch_size, num_channels]
    #         else:
    #             raise RuntimeError("Sampling strategy not specified correctly")
    #         # add offset of [0, num_classes, 2*num_classes, ...] in order for the flattened gathering to be correct:
    #         y += ptu.arange(self.num_channels)[None, :] * self.num_classes
    #         mask = y.unsqueeze(2).repeat(1, 1, self.latent_dim)  # shape [batch_size, num_channels, latent_dim]
    #
    #         if sampler == "random":
    #             # Sample from specified Gaussian using reparametrization trick
    #             # this operation samples from each Gaussian for every class first
    #             # (diag_embed not possible for distributions), put it back to tensor with shape [channels * classes, batch, latent]
    #             sampled = torch.cat([[torch.unsqueeze(dist.rsample(), 0) for dist in chan] for chan in z_distributions],
    #                                 dim=0)
    #
    #         elif sampler == "mean":
    #             sampled = torch.cat([[torch.unsqueeze(dist.mean, 0) for dist in chan] for chan in z_distributions],
    #                                 dim=0)
    #
    #         permute = sampled.permute(1, 0, 2)  # tensor with shape [batch, channels * classes, latent]
    #         z = torch.gather(permute, 1, mask)  # [batch, channels, latent]
    #         return z, y


class MultiChannelDecoderMDP(nn.Module):
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 num_channels,
                 num_classes,
                 net_complex,
                 state_reconstruction_clip,
                 equivariance
    ):
        super(MultiChannelDecoderMDP, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.state_decoder_input_size = state_dim + action_dim + (z_dim + 1) * num_channels
        self.state_decoder_hidden_size = int(self.state_decoder_input_size * net_complex)

        self.reward_decoder_input_size = state_dim + action_dim + (z_dim + 1) * num_channels
        self.reward_decoder_hidden_size = int(self.reward_decoder_input_size * net_complex)
        self.state_reconstruction_clip = state_reconstruction_clip

        if equivariance == "toy2D":
            # Note, that the state decoder doesn't need to be equivariant in this case as an optimal prediction ONLY
            # depends on last state and action and can ignore the latent space.
            self.net_state_decoder = Mlp(
                hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
                input_size=self.state_decoder_input_size,
                output_size=self.state_reconstruction_clip
            )
            self.net_reward_decoder = Toy2DDecoder(num_channels)
        elif equivariance == "toy1D":
            # Note, that the state decoder doesn't need to be equivariant in this case as an optimal prediction ONLY
            # depends on last state and action and can ignore the latent space.
            self.net_state_decoder = Mlp(
                hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
                input_size=self.state_decoder_input_size,
                output_size=self.state_reconstruction_clip
            )
            self.net_reward_decoder = Toy1DDecoder(num_channels)
        elif equivariance == "MetaWorldv1":
            self.net_state_decoder = Mlp(
                hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
                input_size=self.state_decoder_input_size,
                output_size=self.state_reconstruction_clip
            )
            self.net_reward_decoder = MetaWorldv1Decoder(num_channels)
        elif equivariance == "none":
            self.net_state_decoder = Mlp(
                hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
                input_size=self.state_decoder_input_size,
                output_size=self.state_reconstruction_clip
            )
            self.net_reward_decoder = Mlp(
                hidden_sizes=[self.reward_decoder_hidden_size, self.reward_decoder_hidden_size],
                input_size=self.reward_decoder_input_size,
                output_size=reward_dim
            )
        else:
            raise ValueError("Unsupported decoder equivariance: " + equivariance + "!")
        # wandb.watch(self.net_reward_decoder, log='all')

    def forward(self, state, action, next_state, z, y):
        """
        Note, that for num_channels=1 this will always feed a constant 1 in addition to the other parameters
        z: [batch_size, latent] OR [latent] (if only [latent], it will be broadcasted.)
        y: [batch_size] OR [] (if only [], it will be broadcasted.)
        batch_size might also be multiple dimensions
        """
        if z.dim() <= state.dim() - 1:
            z = z.broadcast_to(
                state.shape[:-1] + [z.shape[0]])  # equivalent to expand, doesn't create copy of the values
        if y.dim() < state.dim() - 1:
            y = y.broadcast_to(state.shape[:-1])  # equivalent to expand, doesn't create copy of the values
        z_in = to_latent_hot(y.long(), z, self.num_classes, self.num_channels)
        net_in = torch.cat([state, action, z_in], dim=-1)
        state_estimate = self.net_state_decoder(net_in)
        reward_estimate = self.net_reward_decoder(net_in)

        return state_estimate, reward_estimate

class DecoderMDP(nn.Module):
    '''
    Uses data (state, action, reward, task_hypothesis z) from the replay buffer or online
    and computes estimates for the next state and reward.
    Through that it reconstructs the MDP and gives gradients back to the task hypothesis.
    '''
    def __init__(self,
                 action_dim,
                 state_dim,
                 reward_dim,
                 z_dim,
                 num_channels,
                 num_classes,
                 net_complex,
                 state_reconstruction_clip,
                 equivariance
    ):
        super(DecoderMDP, self).__init__()
        if equivariance != "none":
            raise NotImplementedError("Equivariance not implemented for the default decoder!")
        self.state_decoder_input_size = state_dim + action_dim + z_dim
        self.state_decoder_hidden_size = int(self.state_decoder_input_size * net_complex)

        self.reward_decoder_input_size = state_dim + action_dim + z_dim
        self.reward_decoder_hidden_size = int(self.reward_decoder_input_size * net_complex)
        self.state_reconstruction_clip = state_reconstruction_clip

        self.net_state_decoder = Mlp(
            hidden_sizes=[self.state_decoder_hidden_size, self.state_decoder_hidden_size],
            input_size=self.state_decoder_input_size,
            output_size=self.state_reconstruction_clip
        )
        self.net_reward_decoder = Mlp(
            hidden_sizes=[self.reward_decoder_hidden_size, self.reward_decoder_hidden_size],
            input_size=self.reward_decoder_input_size,
            output_size=reward_dim
        )

    def forward(self, state, action, next_state, z, y=None):
        state_estimate = self.net_state_decoder(torch.cat([state, action, z], dim=-1))
        reward_estimate = self.net_reward_decoder(torch.cat([state, action, z], dim=-1))

        return state_estimate, reward_estimate

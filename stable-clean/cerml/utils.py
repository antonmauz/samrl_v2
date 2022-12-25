import torch
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu


def product_of_gaussians3D(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared

def generate_gaussian(mu_sigma, latent_dim, sigma_ops="softplus", mode=None):
    """
    Generate a Gaussian distribution given a selected parametrization.
    """
    mus, sigmas = torch.split(mu_sigma, split_size_or_sections=latent_dim, dim=-1)

    if sigma_ops == 'softplus':
        # Softplus, s.t. sigma is always positive
        # sigma is assumed to be st. dev. not variance
        sigmas = F.softplus(sigmas)
    if mode == 'multiplication':
        mu, sigma = product_of_gaussians3D(mus, sigmas)
    else:
        mu = mus
        sigma = sigmas
    return torch.distributions.normal.Normal(mu, sigma)


def generate_latent_grid(y_probs, z_distrs, num_channels, num_classes,
                         latent_dim, latent_grid_resolution, latent_grid_range, use_all_channels, z_probs=None):
    """
    Returns [batch_size, 1 or num_channels, 1, grid_resolution] for latent_dim=1 and
    [batch_size, 1 or num_channels, grid_resolution, grid_resolution] for latent_dim=2.
    Optionally, a tensor z_probs [batch_size, num_classes * num_channels * latent_dim] can be passed if z_distrs not
    available
    """
    if z_distrs is None and z_probs is not None:
        z_distrs = [torch.distributions.Normal(
            z_probs[:, 2 * y * latent_dim : (2 * y + 1) * latent_dim],
            z_probs[:, (2 * y + 1) * latent_dim : 2 * (y + 1) * latent_dim])
            for y in range(num_classes * num_channels)]

    # [grid_resolution, batch_size] with each column containing probability to the max of this column
    # for y in enumerate(z_distributions)
    batch_size = y_probs.shape[0]
    # [batch_size], representing the most likely channel for each batch entry
    channel_probs = torch.sum(y_probs.reshape(batch_size * num_channels, num_classes), dim=1)\
        .reshape(batch_size, num_channels)  # [batch_size, num_channels] with the probability of each channel
    # channels: [batch_size], representing the most likely channel for each batch entry
    channels = torch.argmax(channel_probs, dim=1)
    if latent_dim == 1:
        grid = ptu.empty(batch_size, num_channels * num_classes, latent_grid_resolution)
        for y, distr in enumerate(z_distrs):
            # [batch_size, 1] normalize such that the total probability over all classes in the chosen channel is 1
            if use_all_channels:
                factors = y_probs[:, y][:, None]
            else:
                factors = (y_probs[:, y] / channel_probs[:, y // num_classes])[:, None]
            grid[:, y, :] = factors * distr.cdf(((ptu.arange(latent_grid_resolution)
                                                  - latent_grid_resolution / 2 + 1) *
                                                 (2 * latent_grid_range / latent_grid_resolution))
                                                [:, None, None])[:, :, 0].T
            grid[:, y, :] -= factors * distr.cdf(((ptu.arange(latent_grid_resolution) -
                                                   latent_grid_resolution / 2) * (2 * latent_grid_range /
                                                                                       latent_grid_resolution))
                                                 [:, None, None])[:, :, 0].T

        if use_all_channels:
            # [batch_size, num_channels, 1, grid_resolution]
            return torch.sum(grid.reshape(batch_size, num_channels, num_classes, latent_grid_resolution), dim=-2)[:, :, None, :]
        else:
            # [batch, (value: y+b*K, dimension: num_classes), latent_grid_resolution]
            mask = (channels[:, None] * num_classes + ptu.arange(num_classes)[None, :])[:, :, None] \
                .repeat(1, 1, latent_grid_resolution)
            # gather: [batch_size, num_classes, latent_grid], sum: [batch_size, grid_resolution]
            return torch.sum(torch.gather(grid, 1, mask), dim=1)[:, None, None, :]  # [batch_size, 1, 1, grid_resolution]
    elif latent_dim == 2:
        grid = ptu.empty(batch_size, num_channels * num_classes, latent_grid_resolution, latent_grid_resolution)
        for y, distr in enumerate(z_distrs):
            # both of shape [grid_resolution, batch_size, latent_dim] representing the cdf to the highest and to the
            # lowest value of each grid cell respectively
            cdf_upper = distr.cdf(((ptu.arange(latent_grid_resolution) - latent_grid_resolution / 2 + 1)
                                   * (2 * latent_grid_range / latent_grid_resolution)) [:, None, None])
            cdf_lower = distr.cdf(((ptu.arange(latent_grid_resolution) - latent_grid_resolution / 2)
                                   * (2 * latent_grid_range / latent_grid_resolution))[:, None, None])
            # [batch_size, 1] normalize such that the total probability over all classes in the chosen channel is 1
            if use_all_channels:
                factors = y_probs[:, y][:, None]
            else:
                factors = (y_probs[:, y] / channel_probs[:, y // num_classes])[:, None]
            grid[:, y, :, :] = (factors * cdf_upper[:, :, 0].T)[:, :, None]
            grid[:, y, :, :] -= (factors * cdf_lower[:, :, 0].T)[:, :, None]
            grid[:, y, :, :] *= (cdf_upper[:, :, 1].T)[:, None, :] - (cdf_lower[:, :, 1].T)[:, None, :]

            # Note, that the covariance matrix of latent dimensions 1 and 2 is diagonal/the two values are
            # independent by design (we only learn sigma in each direction, no covariances). Therefore, we can just
            # multiply the probability of falling into a square on dimension 1 (only depends on distribution 1) and
            # the probability of falling into a particular square on dimension 2 (only depends on distribution 2).
            # This would also work for a higher-dimensional case.


            # multivariate = torch.distributions.multivariate_normal.MultivariateNormal(distr.mean, torch.diag_embed(distr))

        if use_all_channels:
            # [batch_size, num_channels, grid_resolution, grid_resolution]
            return torch.sum(
                grid.reshape(batch_size, num_channels, num_classes, latent_grid_resolution, latent_grid_resolution), dim=-3)
        else:
            # [batch, (value: y+b*K, dimension: num_classes), grid_resolution, grid_resolution]
            mask = (channels[:, None] * num_classes + ptu.arange(num_classes)[None, :])[:, :, None, None] \
                .repeat(1, 1, latent_grid_resolution, latent_grid_resolution)

            # gather: [batch_size, num_classes, grid_resolution, grid_resolution]
            # sum: [batch_size, 1, grid_resolution, grid_resolution]
            return torch.sum(torch.gather(grid, 1, mask), dim=1)[:, None, :, :]
    else:
        raise NotImplementedError("Only 1D and 2D grids are supported!")

def to_latent_hot(y, z, num_classes, num_channels):
    """
    Note: batch_size can also consist of multiple dimensions (tested)
    z: [batch_size, latent] OR [latent] (if only [latent], it will be broadcasted.)
    y: [batch_size] OR [] (if only [], it will be broadcasted.)

    returns: [batch_size, one-hot encoding of channel | latent_hot_encoding]
    """
    # [batch_size], the channel used for each batch entry
    d = torch.div(y, num_classes, rounding_mode="floor")
    # set z to the correct position in array [batch_size, num_channels * z_dim + num_channels]
    z_in = ptu.zeros(list(y.shape) + [num_channels * (z.shape[-1] + 1)])
    # for each sample, set the correct one-hot value for the corresponding channel:
    z_in.scatter_(index=d[..., None], dim=-1, value=1)
    # if this throws an exception that y is not int64, most likely y and z have been passed in wrong order
    # (or .long() is missing for y)

    # Note, that ptu.arange(z.shape[-1]) is broadcasted on the last dimension
    # *([1] * y.dim() + [z.shape[-1]]): 1 for each batch dimension and latent_dim for the last dimension
    z_in.scatter_(index=((d * z.shape[-1])[..., None].repeat(*([1] * y.dim() + [z.shape[-1]]))) +
                  ptu.arange(z.shape[-1]) + num_channels, dim=-1, src=z)

    # first index: [batch_size, 1] (broadcasted), second index: [batch_size, latent]
    # z_in[ptu.arange(z.shape[0])[:, None], ((d * z.shape[-1])[:, None].repeat(1, z.shape[-1]))
    #      + ptu.arange(z.shape[-1])[None, :] + self.num_channels] = z
    return z_in


import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributions.kl as kl
import wandb

from cerml.utils import generate_gaussian
import rlkit.torch.pytorch_util as ptu
from cerml.encoder_decoder_networks import EncoderMixtureModelGRU

from rlkit.core import logger as rlkit_logger

import matplotlib.pyplot as plt
from analysis import logger


class ReconstructionTrainer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 prior_pz_layer,
                 replay_buffer,
                 batch_size,
                 num_classes,
                 num_channels,
                 latent_dim,
                 timesteps,
                 reconstruct_all_timesteps,
                 max_path_length,
                 lr_decoder,
                 lr_encoder,
                 alpha_kl_z,
                 beta_kl_y,
                 use_state_diff,
                 component_constraint_learning,
                 state_reconstruction_clip,
                 train_val_percent,
                 eval_interval,
                 early_stopping_threshold,
                 experiment_log_dir,
                 prior_mode,
                 prior_sigma,
                 isIndividualY,
                 data_usage_reconstruction,
                 use_state_decoder,
                 optimizer_class=optim.Adam,
                 ):
        super(ReconstructionTrainer, self).__init__()
        # wandb.watch(encoder, log='all')
        self.encoder = encoder
        self.decoder = decoder
        self.prior_pz_layer = prior_pz_layer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.reconstruct_all_timesteps = reconstruct_all_timesteps
        self.max_path_length = max_path_length
        self.lr_decoder = lr_decoder
        self.lr_encoder = lr_encoder
        self.alpha_kl_z = alpha_kl_z
        self.beta_kl_y = beta_kl_y
        self.use_state_diff = use_state_diff
        self.component_constraint_learning = component_constraint_learning
        self.state_reconstruction_clip = state_reconstruction_clip
        self.train_val_percent = train_val_percent
        self.eval_interval = eval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.experiment_log_dir = experiment_log_dir
        self.prior_mode = prior_mode
        self.prior_sigma = prior_sigma
        self.isIndividualY = isIndividualY
        self.data_usage_reconstruction = data_usage_reconstruction

        self.factor_state_loss = 1 if use_state_decoder else 0
        self.factor_reward_loss = self.state_reconstruction_clip

        self.loss_weight_state = self.factor_state_loss / (self.factor_state_loss + self.factor_reward_loss)
        self.loss_weight_reward = self.factor_reward_loss / (self.factor_state_loss + self.factor_reward_loss)

        self.lowest_loss = np.inf
        self.lowest_loss_epoch = 0

        self.temp_path = os.path.join(self.experiment_log_dir.rsplit('/', 1)[0], '.temp',
                                      self.experiment_log_dir.rsplit('/', 1)[1])
        self.encoder_path = os.path.join(self.temp_path, 'encoder.pth')
        self.decoder_path = os.path.join(self.temp_path, 'decoder.pth')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.optimizer_class = optimizer_class

        self.loss_state_decoder = nn.MSELoss()
        self.loss_reward_decoder = nn.MSELoss()

        self.optimizer_encoder = self.optimizer_class(
            list(self.encoder.parameters()) + list(self.prior_pz_layer.parameters()),
            lr=self.lr_encoder,
        )

        self.optimizer_decoder = self.optimizer_class(
            self.decoder.parameters(),
            lr=self.lr_decoder,
        )

    def train(self, epochs):
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        if self.data_usage_reconstruction == "tree_sampling":
            train_indices = np.random.permutation(train_indices)
            val_indices = np.random.permutation(val_indices)

        train_overall_losses = []
        train_state_losses = []
        train_reward_losses = []

        train_val_state_losses = []
        train_val_reward_losses = []

        val_state_losses = []
        val_reward_losses = []

        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        for epoch in range(epochs):
            overall_loss, state_loss, reward_loss, elbo, kl_qy_py = self.training_step(train_indices, epoch)
            train_overall_losses.append(overall_loss)
            train_state_losses.append(state_loss)
            train_reward_losses.append(reward_loss)

            # Evaluate with validation set for early stopping
            if epoch % self.eval_interval == 0:
                val_state_loss, val_reward_loss = self.validate(val_indices)
                val_state_losses.append(val_state_loss)
                val_reward_losses.append(val_reward_loss)
                train_val_state_loss, train_val_reward_loss = self.validate(train_indices)
                train_val_state_losses.append(train_val_state_loss)
                train_val_reward_losses.append(train_val_reward_loss)

                # change loss weighting
                weight_factors = np.ones(2)
                weights = np.array(
                    [train_val_state_loss * self.factor_state_loss, train_val_reward_loss * self.factor_reward_loss])
                for i in range(weights.shape[0]):
                    weight_factors[i] = weights[i] / np.sum(weights)
                self.loss_weight_state = weight_factors[0]
                self.loss_weight_reward = weight_factors[1]
                if int(os.environ['DEBUG']) == 1:
                    print("weight factors: " + str(weight_factors))

                # Debug printing
                if int(os.environ['DEBUG']) == 1:
                    print("\nEpoch: " + str(epoch))
                    print("Overall loss: " + str(train_overall_losses[-1]))
                    print("Train Validation loss (state, reward): " + str(train_val_state_losses[-1]) + ' , ' + str(
                        train_val_reward_losses[-1]))
                    print("Validation loss (state, reward): " + str(val_state_losses[-1]) + ' , ' + str(
                        val_reward_losses[-1]))
                if self.early_stopping(epoch, val_state_loss + val_reward_loss):
                    print("Early stopping at epoch " + str(epoch))
                    break
                # TODO handle the two intervals in combination by calculating how many logs were done before and comparing to the interval
                if logger.USE_TENSORBOARD and logger.LOG_INTERVAL > 0 and logger.TI_LOG_STEP % logger.LOG_INTERVAL == 0:
                    logger.TENSORBOARD_LOGGER.add_scalar('validation/state_loss', val_state_loss,
                                                         global_step=logger.TI_LOG_STEP)
                    logger.TENSORBOARD_LOGGER.add_scalar('validation/reward_loss', val_reward_loss,
                                                         global_step=logger.TI_LOG_STEP)
                    logger.TENSORBOARD_LOGGER.add_scalar('training/val_state_loss', train_val_state_loss,
                                                         global_step=logger.TI_LOG_STEP)
                    logger.TENSORBOARD_LOGGER.add_scalar('training/val_reward_loss', train_val_reward_loss,
                                                         global_step=logger.TI_LOG_STEP)
                if logger.USE_WANDB:
                    wandb.log({'reconst/validation/state_loss': val_state_loss,
                               'reconst/validation/reward_loss': val_reward_loss,
                               'reconst/training/val_state_loss': train_val_state_loss,
                               'reconst/training/val_reward_loss': train_val_reward_loss,
                               'reconst/step': logger.TI_LOG_STEP
                               }, commit=False)
            if logger.USE_TENSORBOARD and logger.LOG_INTERVAL > 0 and logger.TI_LOG_STEP % logger.LOG_INTERVAL == 0:
                logger.TENSORBOARD_LOGGER.add_scalar('training/overall_loss', overall_loss,
                                                     global_step=logger.TI_LOG_STEP)
                logger.TENSORBOARD_LOGGER.add_scalar('training/state_loss', np.average(state_loss),
                                                     global_step=logger.TI_LOG_STEP)
                logger.TENSORBOARD_LOGGER.add_scalar('training/reward_loss', np.average(reward_loss),
                                                     global_step=logger.TI_LOG_STEP)
                logger.TENSORBOARD_LOGGER.add_scalar('training/elbo_loss', np.average(elbo),
                                                     global_step=logger.TI_LOG_STEP)
                logger.TENSORBOARD_LOGGER.add_scalar('training/kl_qy_py', np.average(kl_qy_py),
                                                     global_step=logger.TI_LOG_STEP)
                wandb.log({'reconst/training/overall_loss': overall_loss,
                           'reconst/training/state_loss': np.average(state_loss),
                           'reconst/training/reward_loss': np.average(reward_loss),
                           'reconst/training/elbo_loss': np.average(elbo),
                           'reconst/training/kl_qy_py': np.average(kl_qy_py),
                           'reconst/step': logger.TI_LOG_STEP
                           }, commit=True)

            logger.TI_LOG_STEP += 1

        # load the least loss encoder
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))
        if int(os.environ['PLOT']) == 1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(list(range(len(train_overall_losses))), np.array(train_overall_losses), label="Train overall loss")
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(list(range(len(train_state_losses))), np.array(train_state_losses) + np.array(train_reward_losses),
                     label="Train loss without KL terms")
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(np.array(list(range(len(train_val_state_losses)))) * self.eval_interval,
                     np.array(train_val_state_losses), label="train_val_state_losses")
            plt.plot(np.array(list(range(len(train_val_reward_losses)))) * self.eval_interval,
                     np.array(train_val_reward_losses), label="train_val_reward_losses")
            plt.plot(np.array(list(range(len(val_state_losses)))) * self.eval_interval, np.array(val_state_losses),
                     label="val_state_losses")
            plt.plot(np.array(list(range(len(val_reward_losses)))) * self.eval_interval, np.array(val_reward_losses),
                     label="val_reward_losses")
            plt.xlim(left=0)
            # plt.ylim(bottom=0)
            plt.yscale('log')
            plt.legend()

            plt.show()

        # for logging
        validation_train = self.validate(train_indices)
        validation_val = self.validate(val_indices)

        rlkit_logger.record_tabular("Reconstruction_train_val_state_loss", validation_train[0])
        rlkit_logger.record_tabular("Reconstruction_train_val_reward_loss", validation_train[1])
        rlkit_logger.record_tabular("Reconstruction_val_state_loss", validation_val[0])
        rlkit_logger.record_tabular("Reconstruction_val_reward_loss", validation_val[1])
        rlkit_logger.record_tabular("Reconstruction_epochs", epoch + 1)

        # if tb_logger.LOG_INTERVAL > 0 and tb_logger.TI_LOG_STEP % tb_logger.LOG_INTERVAL == 0:
        #     tb_logger.TENSORBOARD_LOGGER.add_scalar('training/val_state_loss', validation_train[0],
        #                                             global_step=tb_logger.TI_LOG_STEP)
        #     tb_logger.TENSORBOARD_LOGGER.add_scalar('training/val_reward_loss', validation_train[1],
        #                                             global_step=tb_logger.TI_LOG_STEP)
        #     tb_logger.TENSORBOARD_LOGGER.add_scalar('validation/val_state_loss', validation_val[0],
        #                                             global_step=tb_logger.TI_LOG_STEP)
        #     tb_logger.TENSORBOARD_LOGGER.add_scalar('validation/val_reward_loss', validation_val[1],
        #                                             global_step=tb_logger.TI_LOG_STEP)

    def training_step(self, indices, step):
        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder.
        The overall objective due to the generative model is:
        parameter* = arg max ELBO
        ELBO = sum_k q(y=k | x) * [ log p(x|z_k) - KL ( q(z, x,y=k) || p(z|y=k) ) ] - KL ( q(y|x) || p(y) )
        '''

        # get data from replay buffer
        # TODO: for validation data use all data --> batch size == validation size
        if self.reconstruct_all_timesteps:
            enc_data, dec_data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size,
                                                                                 normalize=True,
                                                                                 return_whole_episodes=True)
        else:
            enc_data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)
            dec_data = enc_data

        elbo, _, kl_qy_py, loss, state_losses, reward_losses = self.calculate_elbo(enc_data, dec_data, self.batch_size)
        # Optimization strategy:
        # Decoder: the two head loss functions backpropagate their gradients into corresponding parts
        # of the network, then ONE common optimizer compute all weight updates
        # Encoder: the KLs and the likelihood from the decoder backpropagate their gradients into
        # corresponding parts of the network, then ONE common optimizer computes all weight updates
        # This is not done explicitly but all within the elbo loss.

        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()

        """
        gc.collect()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(obj.shape)
            except:
                pass
        """
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        self.optimizer_decoder.step()
        self.optimizer_encoder.step()
        if self.isIndividualY:
            return ptu.get_numpy(loss) / self.batch_size, ptu.get_numpy(torch.sum(state_losses)) / self.batch_size, \
                   ptu.get_numpy(torch.sum(reward_losses)) / self.batch_size, ptu.get_numpy(elbo), \
                   ptu.get_numpy(kl_qy_py)
        else:
            return ptu.get_numpy(loss) / self.batch_size, ptu.get_numpy(
                torch.sum(state_losses, dim=0)) / self.batch_size, \
                   ptu.get_numpy(torch.sum(reward_losses, dim=0)) / self.batch_size, ptu.get_numpy(elbo), \
                   ptu.get_numpy(kl_qy_py)

    def calculate_elbo(self, enc_data, dec_data, batch_size):
        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(enc_data, batch_size)
        # prepare for usage in decoder
        if self.reconstruct_all_timesteps:
            decoder_action = ptu.from_numpy(dec_data['actions'])
            decoder_state = ptu.from_numpy(dec_data['observations'])
            decoder_next_state = ptu.from_numpy(dec_data['next_observations'])
            decoder_reward = ptu.from_numpy(dec_data['rewards'])
            true_task = dec_data['true_tasks']
        else:
            decoder_action = ptu.from_numpy(dec_data['actions'][:, -1, :])
            decoder_state = ptu.from_numpy(dec_data['observations'][:, -1, :])
            decoder_next_state = ptu.from_numpy(dec_data['next_observations'][:, -1, :])
            decoder_reward = ptu.from_numpy(dec_data['rewards'][:, -1, :])
            true_task = dec_data['true_tasks'][:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[..., :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[..., :self.state_reconstruction_clip]

        # Forward pass through encoder
        y_distribution, z_distributions = self.encoder.encode(encoder_input)

        # TODO currently, the GRU only gets the fixed number of time_steps, potentially filled with zeros and
        #  never enough to deduce everything explored so far. This cannot work out.

        # Legacy (all num_classes^num_channels combinations): combinations =
        # np.array(np.meshgrid(*[np.arange(self.num_classes) for _ in range(self.num_channels)])).T
        # .reshape(-1, self.num_channels)

        if self.isIndividualY:
            if self.num_channels != 1:
                raise NotImplementedError()
            kl_qz_pz = ptu.zeros(batch_size, self.timesteps, self.num_classes)
            state_losses = ptu.zeros(batch_size, self.timesteps, self.num_classes)
            reward_losses = ptu.zeros(batch_size, self.timesteps, self.num_classes)
            nll_px = ptu.zeros(batch_size, self.timesteps, self.num_classes)

            # every y component (see ELBO formula)
            for y in range(self.num_classes):
                z, _ = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y,
                                             batch_size=batch_size)
                if self.reconstruct_all_timesteps:
                    z = z.unsqueeze(1).repeat(1, self.max_path_length, 1)

                # put in decoder to get likelihood
                state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z)
                state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
                reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)
                if self.reconstruct_all_timesteps:
                    state_loss = torch.mean(state_loss, dim=1)
                    reward_loss = torch.mean(reward_loss, dim=1)
                state_losses[:, :, y] = state_loss.unsqueeze(1).repeat(1, self.timesteps)
                reward_losses[:, :, y] = reward_loss.unsqueeze(1).repeat(1, self.timesteps)
                nll_px[:, :, y] = self.loss_weight_state * state_losses[:, :,
                                                           y] + self.loss_weight_reward * reward_losses[:, :, y]

                # KL ( q(z | x,y=k) || p(z|y=k) )
                prior = self.prior_pz(y, batch_size=batch_size)
                # print(z_distributions[y].mean.shape, prior.mean.shape)
                kl_qz_pz[:, :, y] = torch.sum(kl.kl_divergence(z_distributions[y], prior), dim=-1)

            # KL ( q(y | x) || p(y) )
            kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py(batch_size))

            # Overall ELBO
            if not self.component_constraint_learning:
                assert(self.num_classes > 1 or torch.allclose(y_distribution.probs, torch.tensor(1.)))
                elbo_individual = torch.sum(torch.mul(y_distribution.probs, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz),
                                            dim=-1) - self.beta_kl_y * kl_qy_py
                elbo = torch.sum(elbo_individual)

            # Component-constraint_learning (the category y of the tasks is known)
            else:
                temp = ptu.zeros_like(y_distribution.probs)
                index = ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).unsqueeze(
                    1).expand(-1, self.timesteps).unsqueeze(2).long()
                true_task_multiplier = temp.scatter(2, index, 1)
                loss_ce = nn.CrossEntropyLoss(reduction='none')
                elbo_individual = torch.sum(torch.mul(true_task_multiplier, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz),
                                            dim=-1) - loss_ce(y_distribution.probs,
                                                              ptu.from_numpy(np.array(
                                                                  [a["base_task"] for a in
                                                                   true_task[:, 0].tolist()]).unsqueeze(1).expand(-1,
                                                                                                                  self.timesteps)))
                elbo = torch.sum(elbo_individual)
            # but elbo should be maximized, and backward function assumes minimization
            loss = (-1) * elbo
            return elbo, elbo_individual, kl_qy_py, loss, state_losses, reward_losses

        else:
            if self.num_channels != 1 and not isinstance(self.encoder, EncoderMixtureModelGRU):
                raise NotImplementedError()
            kl_qz_pz = ptu.zeros(batch_size, self.num_classes * self.num_channels)
            state_losses = ptu.zeros(batch_size, self.num_classes * self.num_channels)
            reward_losses = ptu.zeros(batch_size, self.num_classes * self.num_channels)
            nll_px = ptu.zeros(batch_size, self.num_classes * self.num_channels)

            # every y component (see ELBO formula)
            for y in range(self.num_classes * self.num_channels):
                z, _ = self.encoder.sample_z(y_distribution, z_distributions, y_usage="specific", y=y,
                                             batch_size=batch_size)
                if self.reconstruct_all_timesteps:
                    z = z.unsqueeze(1).repeat(1, self.max_path_length, 1)
                # simple way of writing: d = torch.div(y, self.num_classes, rounding_mode="floor") in this case:
                # d = ptu.ones(batch_size, dtype=torch.long) * (y // self.num_classes)

                # put in decoder to get likelihood
                state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z,
                                                               ptu.tensor(y))
                state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=-1)
                reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=-1)
                # print(reward_estimate.shape, decoder_reward.shape, reward_loss.shape)
                # [256, 200, 1] [256, 200, 1] [256, 200]
                if self.reconstruct_all_timesteps:
                    state_loss = torch.mean(state_loss, dim=1)
                    reward_loss = torch.mean(reward_loss, dim=1)
                state_losses[:, y] = state_loss
                reward_losses[:, y] = reward_loss
                nll_px[:, y] = self.loss_weight_state * state_loss + self.loss_weight_reward * reward_loss

                # KL(q(z|x,y=(y mod K),b=(y // K)) || p(z|y=(y mod K),b=(y // K)))
                prior = self.prior_pz(y, batch_size=batch_size)
                kl_qz_pz[:, y] = torch.sum(kl.kl_divergence(z_distributions[y], prior), dim=-1)

                # The result appears to be the same although one might want to verify this theoretically:
                # sdist1 = torch.distributions.multivariate_normal.MultivariateNormal(z_distributions[y].mean,
                #                                                                     torch.diag_embed(
                #                                                                         z_distributions[y].stddev ** 2))
                # sdist2 = torch.distributions.multivariate_normal.MultivariateNormal(prior.mean,
                #                                                                     torch.diag_embed(prior.stddev ** 2))
                # if torch.abs(kl.kl_divergence(sdist1, sdist2) - kl_qz_pz[:, y]).max() >= 1e-5:
                #     print("Difference ", torch.abs(kl.kl_divergence(sdist1, sdist2) - kl_qz_pz[:, y]).max(),
                #           " between:\n", kl_qz_pz[:, y], "\nand:\n", kl.kl_divergence(sdist1, sdist2))

            # KL ( q(y, b | x) || p(y, b) )
            kl_qy_py = kl.kl_divergence(y_distribution, self.prior_py(batch_size))
            assert (self.num_classes > 1 or torch.allclose(y_distribution.probs, torch.tensor(1.)))
            # Overall ELBO
            if not self.component_constraint_learning:
                elbo_individual = torch.sum(torch.mul(y_distribution.probs, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz),
                                            dim=-1) - self.beta_kl_y * kl_qy_py
                # elbo_individual = torch.sum(torch.mul(y_distribution.probs, (-1) * nll_px - 0 * self.alpha_kl_z * kl_qz_pz),
                #                             dim=-1) - 0 * self.beta_kl_y * kl_qy_py
                # print(y_distribution.probs.shape, ((-1) * nll_px - 0 * self.alpha_kl_z * kl_qz_pz).shape,
                #       torch.mul(y_distribution.probs, (-1) * nll_px - 0 * self.alpha_kl_z * kl_qz_pz).shape,
                #       torch.sum(torch.mul(y_distribution.probs, (-1) * nll_px - 0 * self.alpha_kl_z * kl_qz_pz),
                #                 dim=-1).shape,
                #       elbo_individual.shape
                # )
                elbo = torch.sum(elbo_individual)

            # Component-constraint_learning
            else:
                temp = ptu.zeros_like(y_distribution.probs)
                true_task_multiplier = temp.scatter(1, ptu.from_numpy(
                    np.array([a["base_task"] for a in true_task[:, 0].tolist()])).unsqueeze(1).long(), 1)
                loss_nll = nn.NLLLoss(reduction='none')
                target_label = ptu.from_numpy(np.array([a["base_task"] for a in true_task[:, 0].tolist()])).long()
                elbo_individual = torch.sum(torch.mul(true_task_multiplier, (-1) * nll_px - self.alpha_kl_z * kl_qz_pz),
                                            dim=-1) - self.beta_kl_y * loss_nll(torch.log(y_distribution.probs),
                                                                                target_label)
                elbo = torch.sum(elbo_individual)
            # but elbo should be maximized, and backward function assumes minimization
            loss = (-1) * elbo
            #wandb.log({'reconst/training/kl_qz_pz': ptu.get_numpy(torch.sum(self.beta_kl_y * kl_qy_py))})
            return elbo, elbo_individual, kl_qy_py, loss, state_losses, reward_losses

    def prior_pz(self, y, batch_size=None):
        '''
        As proposed in the CURL paper: use linear layer, that conditioned on y gives Gaussian parameters
        OR
        Gaussian with N(y, 0.5)
        IF z not used:
        Just give back y with 0.01 variance
        for num_channels > 1, y is expected to be b * K + y
        '''

        if batch_size is None:
            batch_size = self.batch_size

        if self.isIndividualY:
            assert (not hasattr(self.encoder, "equivariance") or self.encoder.equivariance == "none")
            if self.prior_mode == 'fixedOnY':
                return torch.distributions.normal.Normal(ptu.ones(batch_size, self.timesteps, 1) * y,
                                                         ptu.ones(batch_size, self.timesteps, 1) * self.prior_sigma)

            elif self.prior_mode == 'network':
                one_hot = ptu.zeros(batch_size, self.timesteps, self.num_classes * self.num_channels)
                one_hot[:, :, y] = 1
                mu_sigma = self.prior_pz_layer(one_hot)
                return generate_gaussian(mu_sigma, self.latent_dim)
        else:
            if self.encoder.equivariance in ["toy1D", "MetaWorldv1"]:
                class_divisor = 2
            elif self.encoder.equivariance == "toy2D":
                class_divisor = 4
            elif self.encoder.equivariance == "none":
                class_divisor = 1

            if self.prior_mode == 'fixedOnY':
                # Note, that the channels are ignored, arranging the same means for each channel
                # [batch_size, latent_dim] containing index of the group of classes
                means = ptu.ones(batch_size, self.latent_dim) * ((y % self.num_channels) // class_divisor)
                if self.encoder.equivariance in ["toy1D", "MetaWorldv1"]:
                    # mirror the uneven group elements to the negative
                    means *= -(y % 2)
                elif self.encoder.equivariance == "toy2D":
                    # Note, that latent variable 1 and 2 are identical as we arrange it on a diagonal. Thus, we don't
                    # need to switch them and only negate appropriate column
                    group = y % 4
                    means[:, 0] *= -1. * torch.logical_or(group == 2, group == 3)
                    means[:, 1] *= -1. * torch.logical_or(group == 1, group == 2)
                elif self.encoder.equivariance == "none":
                    # in case there is no enforced equivariance, we want the latent space to be symmetric around 0
                    # instead  of just on the positive side (like in CEMRL). This way, equivariant representations can
                    # still be learned (e.g. if the decoder is equivariant) and are not actively discouraged by forcing
                    # all means to be positive
                    means -= (self.num_classes - 1) / 2  # note, that the result isn't integer for even num_classes
                variances = ptu.ones(batch_size, self.latent_dim) * self.prior_sigma
            elif self.prior_mode == 'network':
                one_hot = ptu.zeros(batch_size, self.num_channels * self.num_classes // class_divisor)
                one_hot[:, y // class_divisor] = 1
                mu_sigma = self.prior_pz_layer(one_hot)
                means, variances = torch.split(mu_sigma, split_size_or_sections=self.latent_dim, dim=-1)
                variances = torch.nn.functional.softplus(variances)

                if self.encoder.equivariance in ["toy1D", "MetaWorldv1"]:
                    means = means * -(y % 2)
                elif self.encoder.equivariance == "toy2D":
                    group = y % 4
                    if not torch.is_tensor(group):
                        group = torch.tensor(group)
                    switch = (y % 2) == 1  # True for groups 1 and 3, where y and x need to be switched
                    res = means.clone()
                    res[:, 0][switch] = means[:, 1][switch]
                    res[:, 1][switch] = means[:, 0][switch]
                    res[:, 0] *= -1. * torch.logical_or(group == 2, group == 3)
                    res[:, 1] *= -1. * torch.logical_or(group == 1, group == 2)
                    means = res
                    res = variances.clone()
                    res[:, 0][switch] = variances[:, 1][switch]
                    res[:, 1][switch] = variances[:, 0][switch]
                    variances = res
            return torch.distributions.normal.Normal(means, variances)

    def prior_py(self, batch_size=None):
        '''
        Categorical uniform distribution
        '''
        if batch_size is None:
            batch_size = self.batch_size
        if self.isIndividualY:
            return torch.distributions.categorical.Categorical(
                probs=ptu.ones(batch_size, self.timesteps, self.num_classes * self.num_channels) *
                      (1.0 / self.num_classes))
        else:
            return torch.distributions.categorical.Categorical(
                probs=ptu.ones(batch_size, self.num_classes * self.num_channels) * (1.0 / self.num_classes))

    def validate(self, indices):
        # get data from replay buffer
        data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size, normalize=True)

        # prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(data, self.batch_size)
        # prepare for usage in decoder
        decoder_action = ptu.from_numpy(data['actions'])[:, -1, :]
        decoder_state = ptu.from_numpy(data['observations'])[:, -1, :]
        decoder_next_state = ptu.from_numpy(data['next_observations'])[:, -1, :]
        decoder_reward = ptu.from_numpy(data['rewards'])[:, -1, :]

        if self.use_state_diff:
            decoder_state_target = (decoder_next_state - decoder_state)[:, :self.state_reconstruction_clip]
        else:
            decoder_state_target = decoder_next_state[:, :self.state_reconstruction_clip]

        z, y = self.encoder(encoder_input)
        state_estimate, reward_estimate = self.decoder(decoder_state, decoder_action, decoder_next_state, z, y)
        state_loss = torch.sum((state_estimate - decoder_state_target) ** 2, dim=1)
        reward_loss = torch.sum((reward_estimate - decoder_reward) ** 2, dim=1)

        return ptu.get_numpy(torch.sum(state_loss)) / self.batch_size, ptu.get_numpy(
            torch.sum(reward_loss)) / self.batch_size

    def early_stopping(self, epoch, loss):
        if loss < self.lowest_loss:
            if int(os.environ['DEBUG']) == 1:
                print("Found new minimum at Epoch " + str(epoch))
            self.lowest_loss = loss
            self.lowest_loss_epoch = epoch
            torch.save(self.encoder.state_dict(), self.encoder_path)
            torch.save(self.decoder.state_dict(), self.decoder_path)
        if epoch - self.lowest_loss_epoch > self.early_stopping_threshold:
            return True
        else:
            return False

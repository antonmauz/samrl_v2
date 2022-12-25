from rlkit.torch.core import np_ify


class TanhGaussianPolicy:
    def get_action(self, obs, deterministic=True):
        pass

    def get_actions(self, obs, deterministic=True):
        return np_ify(self.get_action(obs))
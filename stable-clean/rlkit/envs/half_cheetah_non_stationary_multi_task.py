from meta_rand_envs.half_cheetah_non_stationary_multi_task import HalfCheetahNonStationaryMultiTaskEnv
from . import register_env


@register_env('cheetah-non-stationary-multi-task')
@register_env('cheetah-stationary-multi-task')
class HalfCheetahNonStationaryMultiTaskWrappedEnv(HalfCheetahNonStationaryMultiTaskEnv):
    def __init__(self, *args, **kwargs):
        self.last_idx = None
        self.env_buffer = {}
        HalfCheetahNonStationaryMultiTaskEnv.__init__(self, *args, **kwargs)

    def clear_buffer(self):
        self.env_buffer = {}

    def reset_task(self, idx, keep_buffered=False):
        if self.last_idx is not None and keep_buffered:
            self.env_buffer[self.last_idx] = self.sim.get_state()
        self.last_idx = idx

        HalfCheetahNonStationaryMultiTaskEnv.reset_task(self, idx)

        if keep_buffered:
            self.env_buffer[idx] = self.sim.get_state()
        return self._get_obs()

    def set_task(self, idx):
        assert idx in self.env_buffer.keys()

        # TODO: In case of dynamic environments, the new task has to be saved as well
        if self.last_idx is not None:
            self.env_buffer[self.last_idx] = self.sim.get_state()

        HalfCheetahNonStationaryMultiTaskEnv.set_task(self, idx)
        self.sim.reset()
        self.sim.set_state(self.env_buffer[idx])
        self.sim.forward()
        self.last_idx = idx
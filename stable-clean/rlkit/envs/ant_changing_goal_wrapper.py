import numpy as np
from meta_rand_envs.ant_changing_goal import AntChangingGoalEnv

from . import register_env


@register_env('ant-changing-goal')
class AntChangingGoalWrappedEnv(AntChangingGoalEnv):
    def __init__(self, *args, **kwargs):
        self.last_idx = None
        self.env_buffer = {}
        super(AntChangingGoalWrappedEnv, self).__init__(*args, **kwargs)
        self.tasks = self.sample_tasks(kwargs['n_train_tasks'] + kwargs['n_eval_tasks'] + kwargs['n_grid_tasks'])
        self.train_tasks = self.tasks[:kwargs['n_train_tasks']]
        self.test_tasks = self.tasks[kwargs['n_train_tasks']:kwargs['n_train_tasks']+kwargs['n_eval_tasks']]
        self.grid_tasks = self.tasks[kwargs['n_train_tasks']+kwargs['n_eval_tasks']:]
        self.reset_task(0)

    def clear_buffer(self):
        self.env_buffer = {}

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx, keep_buffered=False):
        if self.last_idx is not None and keep_buffered:
            self.env_buffer[self.last_idx] = self.sim.get_state()
        self.last_idx = idx

        self._task = self.tasks[idx]
        self.goal = self._task
        # self.reset_change_points() TODO necessary if switching to non-stationary
        self.recolor()
        self.steps = 0
        self.reset()

        if keep_buffered:
            self.env_buffer[idx] = self.sim.get_state()
        return self._get_obs()

    def set_task(self, idx):
        assert idx in self.env_buffer.keys()

        # TODO: In case of dynamic environments, the new task has to be saved as well
        if self.last_idx is not None:
            self.env_buffer[self.last_idx] = self.sim.get_state()

        self._task = self.tasks[idx]
        self.goal = self._task
        self.recolor()
        self.sim.reset()
        self.sim.set_state(self.env_buffer[idx])
        self.sim.forward()
        self.last_idx = idx
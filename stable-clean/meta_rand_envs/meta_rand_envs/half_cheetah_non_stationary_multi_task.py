import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryMetaEnv


class HalfCheetahNonStationaryMultiTaskEnv(NonStationaryMetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.task_variants = kwargs.get('task_variants', ['velocity', 'direction', 'goal', 'jumping', 'flipping'])
        self.termination_possible = kwargs.get('termination_possible', False)
        self.distribution_shift = kwargs.get('distribution_shift', 0.0)
        self.vel_range = kwargs.get('vel_range', 3.0)
        self.goal_range = kwargs.get('goal_range', 5.0)
        # offset of task around zero like no_reward_zone but with dense reward
        self.task_offset = kwargs.get('task_offset', 0.0)
        # Radius around start of the zone where no reward is given
        self.no_reward_zone = kwargs.get('no_reward_zone', 0.0)
        # Type of auxiliary reward so the agent learns to move at all in sparse setting
        self.aux_reward_for_sparse = kwargs.get('aux_reward_for_sparse', 'none')
        self.observe_x = kwargs.get('observe_x', False)
        self.temp = 2
        self.current_task = None
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.active_task = {'base_task': 1, 'specification': 1, 'color': np.array([0, 1, 0])}
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        # should actually go into NonStationaryGoalVelocityEnv, breaks abstraction
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.grid_tasks = self.sample_tasks_grid(kwargs['n_grid_tasks'])  # used for showcasing/plotting
        self.tasks = self.train_tasks + self.test_tasks + self.grid_tasks
        self.reset_task(0)

    def resample_tasks(self, new_attributes):
        """
        Note, that this only used by the request handler and not for dynamic environments
        """
        for attr in new_attributes:
            setattr(self, attr["name"], attr["val"])
            # print(attr["name"] + ": " + str(getattr(self, attr["name"])) + str(self.distribution_shift))
        self.train_tasks = self.sample_tasks(len(self.train_tasks))
        self.test_tasks = self.sample_tasks(len(self.test_tasks))
        self.grid_tasks = self.sample_tasks_grid(len(self.grid_tasks))
        self.tasks = self.train_tasks + self.test_tasks + self.grid_tasks
        self.reset_task(0)

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos.copy()
        ob = self._get_obs()
        achieved_spec = -1
        if self.aux_reward_for_sparse == "velocity":
            reward_aux = 0.001 * abs((xposafter[0] - xposbefore[0]) / self.dt) - self.no_reward_zone
        elif self.aux_reward_for_sparse == "none":
            # otherwise it would be best to remain in this zone in order to get no negative reward
            reward_aux = 0
        else:
            reward_aux = -0.5 * 1e-1 * np.sum(np.square(action))
            if self.aux_reward_for_sparse != "ctrl":
                print("Unknown auxiliary reward type. This might be due to misspelling or using a legacy config.")

        if self.active_task['base_task'] == 1:  # 'velocity'
            achieved_spec = (xposafter[0] - xposbefore[0]) / self.dt  # forward velocity
            if self.no_reward_zone != 0 and (
                    (achieved_spec < self.no_reward_zone and self.active_task['specification'] > 0) or
                    (achieved_spec > -self.no_reward_zone and self.active_task['specification'] < 0)):
                reward_run = -self.no_reward_zone - self.vel_range
            else:
                reward_run = -1.0 * abs(achieved_spec - self.active_task['specification'])
            reward = reward_aux * 1.0 + reward_run
            reward_max = 300
            # reward = reward / reward_max

        elif self.active_task['base_task'] == 2:  # 'direction'
            reward_run = (xposafter[0] - xposbefore[0]) / self.dt * self.active_task['specification']
            reward = reward_aux * 1.0 + reward_run
            # reward = reward / 20

        elif self.active_task['base_task'] == 3:  # 'goal'
            achieved_spec = xposafter[0]
            if self.no_reward_zone != 0 and (
                    (achieved_spec < self.no_reward_zone and self.active_task['specification'] > 0) or
                    (achieved_spec > -self.no_reward_zone and self.active_task['specification'] < 0)):
                reward_run = -self.no_reward_zone - self.goal_range  # TODO normalize
            else:
                # Goals should be shifted by no_reward_zone in both directions (no goal should lie in the no_reward_zone)
                reward_run = -abs(achieved_spec - self.active_task['specification']) /\
                             np.abs(self.active_task['specification'])  # Normalized
                # reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
                # As the expected value of moving in a random direction in dense reward setting (which happens at the
                # beginning when the encoding is not trained yet) is 0 (moving in goal direction or away with same
                # probability), a penalty for actions would make no movement the best option so the encoder in turn
                # cannot learn to distinguish left and right because it always observes reward from the center position.
                # Instead, give a small reward for movement -> better exploration.
                # reward_aux = 0.0001 * abs((xposafter[0] - xposbefore[0]) / self.dt) - self.no_reward_zone
            reward = reward_aux * 1.0 + reward_run
            # reward = reward / 300.0

        elif self.active_task['base_task'] == 4:  # 'flipping'
            reward_run = (xposafter[2] - xposbefore[2]) / self.dt * self.active_task['specification']
            reward = reward_aux * 1.0 + reward_run
            # reward = reward / 20

        elif self.active_task['base_task'] == 5:  # 'jumping'
            reward_run = xposafter[1]
            reward = reward_aux * 1.0 + reward_run
            # reward = reward / 25.0
        else:
            raise RuntimeError("bask task not recognized")

        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run,
                                      # reward_ctrl=reward_ctrl,
                                      reward_aux=reward_aux,
                                      true_task=dict(base_task=self.active_task['base_task'],
                                                     specification=self.active_task['specification']),
                                      velocity=(xposafter[0] - xposbefore[0]) / self.dt,
                                      position=xposafter[:3],
                                      achieved_spec=achieved_spec)

    # from pearl
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[(0 if self.observe_x else 1):],
            self.get_body_com("torso").flat,  # this might be redundant and make the observe_x unnecessary but ONLY if
                                              # we increase the reconstruction clip
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def reset_task(self, idx):
        """
        Note, that this is called by rlkit.envs
        """
        self.task = self.tasks[idx]
        self.active_task = self.task
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()

    def set_task(self, idx):
        """
        Note, that this is called by rlkit.envs
        """
        self.active_task = self.tasks[idx]
        self.recolor()

    def sample_tasks_grid(self, num_tasks):
        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = int(num_tasks / num_base_tasks)
        num_tasks_per_subtask_half = int(num_tasks_per_subtask / 2)

        tasks = []
        # velocity tasks
        if 'velocity' in self.task_variants:
            # velocities = np.random.uniform(1.0, 3.0, size=(num_tasks_per_subtask,))
            # velocities = np.linspace(1.0, 3.0, num_tasks_per_subtask)
            velocities = np.linspace(-self.vel_range + self.distribution_shift,
                                     self.vel_range + self.distribution_shift,
                                     num=num_tasks_per_subtask)
            velocities += np.sign(velocities) * (self.no_reward_zone + self.task_offset)
            tasks_velocity = [{'base_task': 1, 'specification': velocity, 'color': np.array([1, 0, 0])} for velocity in
                              velocities]
            tasks += (tasks_velocity)

        # direction
        # if 'direction' in self.task_variants:
        #     directions = np.concatenate(
        #         ((-1) * np.ones(num_tasks_per_subtask_half), np.ones(num_tasks_per_subtask_half)))
        #     tasks_direction = [{'base_task': 2, 'specification': direction, 'color': np.array([0, 1, 0])} for direction
        #                        in directions]
        #     tasks += (tasks_direction)

        # goal
        if 'goal' in self.task_variants:
            # shift distribution so moving to the right generally makes sense and therefore agent will move and decoder
            # will learn
            goals = np.linspace(-self.goal_range + self.distribution_shift,
                                self.goal_range + self.distribution_shift,
                                num=num_tasks_per_subtask)
            goals += (self.no_reward_zone + self.task_offset) * np.sign(goals)
            tasks_goal = [{'base_task': 3, 'specification': goal,
                           'color': np.array([0, 0, 1])} for goal in goals]
            tasks += (tasks_goal)

        # flipping
        # if 'flipping' in self.task_variants:
        #     directions = np.concatenate(
        #         ((-1) * np.ones(num_tasks_per_subtask_half), np.ones(num_tasks_per_subtask_half)))
        #     tasks_flipping = [{'base_task': 4, 'specification': direction, 'color': np.array([0.5, 0.5, 0])} for
        #                       direction in directions]
        #     tasks += (tasks_flipping)
        #
        # # jumping
        # if 'jumping' in self.task_variants:
        #     tasks_jumping = [{'base_task': 5, 'specification': 0, 'color': np.array([0, 0.5, 0.5])} for _ in
        #                      range(num_tasks_per_subtask)]
        #     tasks += (tasks_jumping)
        return tasks

    def sample_tasks(self, num_tasks):
        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = int(num_tasks / num_base_tasks)
        num_tasks_per_subtask_half = int(num_tasks_per_subtask / 2)
        # np.random.seed(1337)

        tasks = []
        # velocity tasks
        if 'velocity' in self.task_variants:
            # velocities = np.random.uniform(1.0, 3.0, size=(num_tasks_per_subtask,))
            # velocities = np.linspace(1.0, 3.0, num_tasks_per_subtask)
            velocities = np.random.uniform(-self.vel_range + self.distribution_shift,
                                           self.vel_range + self.distribution_shift,
                                           size=(num_tasks_per_subtask,))
            velocities += np.sign(velocities) * (self.no_reward_zone + self.task_offset)
            # print(self.vel_range, self.distribution_shift, self.task_offset, self.no_reward_zone)
            # print(velocities)
            tasks_velocity = [{'base_task': 1, 'specification': velocity, 'color': np.array([1, 0, 0])} for velocity in
                              velocities]
            tasks += (tasks_velocity)

        # direction
        if 'direction' in self.task_variants:
            directions = np.concatenate(
                ((-1) * np.ones(num_tasks_per_subtask_half), np.ones(num_tasks_per_subtask_half)))
            tasks_direction = [{'base_task': 2, 'specification': direction, 'color': np.array([0, 1, 0])} for direction
                               in directions]
            tasks += (tasks_direction)

        # goal
        if 'goal' in self.task_variants:
            # shift distribution so moving to the right generally makes sense and therefore agent will move and decoder
            # will learn
            goals = np.random.uniform(-self.goal_range + self.distribution_shift,
                                      self.goal_range + self.distribution_shift,
                                      size=(num_tasks_per_subtask,))
            goals += (self.no_reward_zone + self.task_offset) * np.sign(goals)
            tasks_goal = [{'base_task': 3, 'specification': goal,
                           'color': np.array([0, 0, 1])} for goal in goals]
            tasks += (tasks_goal)

        # flipping
        if 'flipping' in self.task_variants:
            directions = np.concatenate(
                ((-1) * np.ones(num_tasks_per_subtask_half), np.ones(num_tasks_per_subtask_half)))
            tasks_flipping = [{'base_task': 4, 'specification': direction, 'color': np.array([0.5, 0.5, 0])} for
                              direction in directions]
            tasks += (tasks_flipping)

        # jumping
        if 'jumping' in self.task_variants:
            tasks_jumping = [{'base_task': 5, 'specification': 0, 'color': np.array([0, 0.5, 0.5])} for _ in
                             range(num_tasks_per_subtask)]
            tasks += (tasks_jumping)

        return tasks

    def change_active_task(self, step=100, dir=1):
        self.tasks[self.last_idx] = self.sample_tasks(1)[0]

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value = self.active_task['color']
        geom_rgba[1:, :3] = np.asarray(rgb_value)
        self.model.geom_rgba[:] = geom_rgba

import numpy as np
from gym import Env, spaces
import cv2 as cv

from . import register_env


@register_env('toy-goal')
class ToyGoalEnv(Env):

    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', 'time')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 100)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0
        self.goal = {'goal': np.array([0.0, 0.0]), 'angle': 0.0, 'radius': 0.0}
        self.task_max_radius = kwargs.get('task_max_radius', 1.0)
        self.task_goal_offset = kwargs.get('task_goal_offset', 0.0)
        # Reward given in a circle of the size of one quadrant by default
        self.reward_radius = kwargs.get('reward_radius', (self.task_max_radius - self.task_goal_offset) / 2)
        self.goal_radius = kwargs.get('goal_radius', 0.05)
        self.step_size = kwargs.get('step_size', 0.1)
        # Shifts the goal distribution (in 2D case: along the diagonal)
        self.distribution_shift = kwargs.get('distribution_shift', 0.0)
        # For visualization of the latent space, tasks are sampled from a grid in the cartesian or polar coordinate
        # system instead of randomly. This includes points that were not in the distribution with
        # task_goal_offset != 0
        self.grid_mode = kwargs.get('grid_mode', 'none')
        self.random_start = kwargs.get('random_start', False)
        self.goal_1d = kwargs.get('goal_1d', False)
        self.one_side_goals = kwargs.get('one_side_goals', False)
        self.positive_environment = kwargs.get('positive_environment', False)  # For debugging only
        # Overwrites the action given by policy with a random one. IMPORTANT: this can only server as a baseline, how
        # good the values can get while ignoring the actual behavior of the environment. The VAE couldn't learn any
        # correct policy with this as the encoder/decoder see different actions/movements than the environment takes.
        self.random_policy = kwargs.get('random_policy', False)

        #####################################################
        self.tasks = self.sample_tasks(kwargs['n_train_tasks'] + kwargs['n_eval_tasks'], grid_mode=self.grid_mode)  # is this correct?
        self.train_tasks = self.tasks[:kwargs['n_train_tasks']]
        self.test_tasks = self.tasks[kwargs['n_train_tasks']:]
        self.grid_tasks = self.sample_tasks(kwargs['n_grid_tasks'], grid_mode="cartesian")
        self.tasks += self.grid_tasks
        self.last_idx = None
        self.env_buffer = {}
        self.reset_task(0)
        #######################################################
        self.observation_space = spaces.Box(low=-self.task_max_radius - self.task_goal_offset,
                                            high=self.task_max_radius + self.task_goal_offset,
                                            shape=(1 if self.goal_1d else 2,))
        # The wrapper env squeezes action values in this range directly before passing them to step()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1 if self.goal_1d else 2,))
        # Currently only used if task_goal_offset == 0
        self.max_distance = np.sqrt(2 * ((2 * (self.task_max_radius + self.task_goal_offset)) ** 2))

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx, keep_buffered=False):
        if self.last_idx is not None and keep_buffered:
            self.env_buffer[self.last_idx] = np.copy(self._state)
        self.last_idx = idx
        self._task = self.tasks[int(idx)]
        self.goal = self._task
        self.reset()
        if keep_buffered:
            self.env_buffer[idx] = np.copy(self._state)
        return self._get_obs()

    def set_task(self, idx):
        assert idx in self.env_buffer.keys()

        # TODO: In case of dynamic environments, the new task has to be saved as well
        if self.last_idx is not None:
            self.env_buffer[self.last_idx] = np.copy(self._state)

        self._task = self.tasks[int(idx)]
        self.goal = self._task
        self._state = self.env_buffer[idx]
        self.last_idx = idx

    def clear_buffer(self):
        self.env_buffer = {}

    # renamed from _step
    def step(self, action):
        if self.goal_1d:
            act = np.zeros_like(action)
            act[0] = action[0]
            action = act
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize:
                self.change_goal()

        ##################### Custom
        if self.random_policy:
            action = np.random.uniform(-1, 1, size=2)

        self._state += action * self.step_size # as steps weirdly seem to be way too small
        # This might make dynamics and goal prediction way harder:
        # map_size = self.task_goal_offset + self.task_max_radius
        # if self._state[0] < -map_size:
        #     self._state[0] = -map_size
        # elif self._state[0] > map_size:
        #     self._state[0] = map_size
        # if self._state[1] < -map_size:
        #     self._state[1] = -map_size
        # elif self._state[1] > map_size:
        #     self._state[1] = map_size
        ##################### End Custom
        ob = self._get_obs()

        # Tuned like in PEARL
        reward = self._get_reward_at(self._state[0], self._state[1])
        done = False
        self.steps += 1
        if self.goal_1d:
            return ob, reward, done, dict(true_task=dict(base_task=0, specification=self.goal['goal'][0]),
            # np.array([self.goal['goal'][0], self.goal['goal'][1], self.goal['angle'], self.goal['radius']])), TODO enable specification of higher dimension
                                          success=bool(np.linalg.norm(self._state[0] - self.goal['goal'][0]) < self.goal_radius),
                                          achieved_spec=self._state[0])
        else:
            return ob, reward, done, dict(true_task=dict(base_task=0, specification=self.goal['goal']),
                                          # np.array([self.goal['goal'][0], self.goal['goal'][1], self.goal['angle'], self.goal['radius']])), TODO enable specification of higher dimension
                                          success=bool(
                                              np.linalg.norm(self._state - self.goal['goal']) < self.goal_radius),
                                          achieved_spec=self._state)

    def _get_obs(self):
        return np.copy(self._state[0:1] if self.goal_1d else self._state)# / (self.task_goal_offset + self.task_max_radius)

    def _get_reward_at_with_goals(self, y, x, y_goal, x_goal):
        if self.goal_1d:
            dist = np.abs(y - y_goal)
        else:
            dist = np.sqrt((y - y_goal) ** 2 + (x - x_goal) ** 2)
        return -dist
        # Dense reward for task_goal_offset 0
        # -reward_radius very important (otherwise, goal avoidance would be better)
        # reward = -self.reward_radius if self.task_goal_offset != 0 and dist > self.reward_radius else -dist
        if self.task_goal_offset != 0:
            reward = np.maximum(-dist, -self.reward_radius)
        else:
            reward = -dist
        # if self.task_goal_offset != 0:  # Not done for dense reward for chart comprehensibility
        # May not be done if early termination for goal reaching happens as positive reward gives incentive NOT
        # to finish
        # reward += self.reward_radius
        #print(self.max_distance, self.task_goal_offset, self.reward_radius, self.task_goal_offset == 0)
        reward /= self.max_distance if self.task_goal_offset == 0 else self.reward_radius
        reward += 1

        # if np.any(reward < 0) or np.any(reward > 1):
        #     print("Reward out of range:", reward, y, x, y_goal, x_goal, dist, self.reward_radius, self.task_goal_offset)
        return reward

    def _get_reward_at(self, y, x):
        return self._get_reward_at_with_goals(y, x, self.goal['goal'][0], self.goal['goal'][1])

    def get_all_task_rewards(self, ys, xs):
        assert ys.shape[0] == len(self.tasks) == xs.shape[0] and ys.ndim == 1 == xs.ndim
        goals = np.array([t['goal'] for t in self.tasks])
        return self._get_reward_at_with_goals(ys, xs, goals[:, 0], goals[:, 1])

    def reset_model(self):
        # for time variant
        self.steps = 0

        # reset velocity to task starting velocity
        self.goal = self._task
        self.recolor()

        # standard
        if self.random_start:
            a = np.random.randint(2) * np.pi if self.goal_1d else np.random.random() * 2 * np.pi
            # r = self.task_max_radius * (np.random.random() ** 0.5)
            r = self.task_max_radius * np.random.random()
            self._state = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
            if self.positive_environment:
                self._state += self.task_goal_offset + self.task_max_radius
        else:
            if self.positive_environment:
                self._state = np.ones(2, dtype=np.float32) * (self.task_max_radius + self.task_goal_offset)
            else:
                self._state = np.zeros(2, dtype=np.float32)
            # self._state = np.ones(2, dtype=np.float32) * (-self.task_max_radius - self.task_goal_offset)
        return self._get_obs()

    """ coordinates to image """

    def c2i(self, coords, w, h):
        if self.positive_environment:
            return int(np.round((coords[1] * w / (2 * (self.task_goal_offset + self.task_max_radius))))), \
                   int(np.round((coords[0] * h / (2 * (self.task_goal_offset + self.task_max_radius)))))
        else:
            return int(np.round((coords[1] * w / (2 * (self.task_goal_offset + self.task_max_radius)) + w / 2))), \
                   int(np.round((coords[0] * h / (2 * (self.task_goal_offset + self.task_max_radius)) + h / 2)))

    def get_image(self, width=256, height=256, camera_name=None):
        img = np.zeros([height, width, 3])
        # such that task ends when circles touch (each radius half goal radius)
        radius = int(np.round(width * self.goal_radius / 2))
        cv.circle(img, self.c2i(self.goal['goal'], width, height), radius, (1, 0, 0), -1)
        cv.circle(img, self.c2i(self._state, width, height), radius, (0, 0, 1), -1)

        # Show reward (inefficient approach but makes sure the correct reward is visualized)
        # reward_map = np.vectorize(self._get_reward_at, excluded=['self'])
        w_space = np.linspace(-self.task_max_radius - self.task_goal_offset,
                              self.task_max_radius + self.task_goal_offset, num=width)[None, :]
        h_space = np.linspace(-self.task_max_radius - self.task_goal_offset,
                              self.task_max_radius + self.task_goal_offset, num=height)[:, None]
        if self.positive_environment:
            w_space += self.task_max_radius + self.task_goal_offset
            h_space += self.task_max_radius + self.task_goal_offset
        # img[:, :, 1] = reward_map(y=h_space, x=w_space)
        img[:, :, 1] = self._get_reward_at(h_space, w_space)
        # print(img[:, :, 1].min(), img[:, :, 1].max())
        # No longer necessary as reward already in [0, 1]
        # if self.task_goal_offset == 0:
        #     rmin = np.min(img[:, :, 1])
        #     rmax = np.max(img[:, :, 1])
        #     img[:, :, 1] = (img[:, :, 1] - rmin) / (rmax - rmin)
        # else:
        #     img[:, :, 1] = 1 + (img[:, :, 1] / self.reward_radius) # more efficient than formulation above
        # Also converting to rgb as it is converted back to bgr in runner
        return (255 * img[:, :, ::-1]).astype(np.uint8)

    def viewer_setup(self):
        pass

    def change_goal(self):
        if self.meta_mode == 'train':
            self.goal = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.goal = np.random.choice(self.test_tasks)
        elif self.meta_mode == 'grid':
            self.goal = np.random.choice(self.grid_tasks)

        if self.meta_mode != 'none':
            self.recolor()
            self.steps = 0

    def recolor(self):
        pass
        # geom_rgba = self._init_geom_rgba.copy()
        # hue = self.goal['angle'] / (2 * np.pi)  # maps color in hsv color space
        # saturation = self.goal['radius'] / (self.task_max_radius)
        # rgb_value_tuple = colorsys.hsv_to_rgb(hue, saturation, 1)
        # geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        # self.model.geom_rgba[:] = geom_rgba

    def sample_tasks(self, num_tasks, grid_mode):
        # TODO some of the angle/radius values might be broken in some cases as they are not used anyways
        if grid_mode == 'cartesian':
            if self.goal_1d:
                lin = np.linspace(self.task_goal_offset if self.one_side_goals else
                                  -(self.task_max_radius + self.task_goal_offset),
                                  self.task_max_radius + self.task_goal_offset, num=num_tasks)
                goals = np.stack((lin, np.zeros(num_tasks)), axis=1)
            else:
                n = int(np.ceil(np.sqrt(num_tasks)))
                lin = np.linspace(self.task_goal_offset if self.one_side_goals else
                                  -(self.task_max_radius + self.task_goal_offset),
                                  self.task_max_radius + self.task_goal_offset, num=n)
                goals = np.stack((np.repeat(lin, n), np.tile(lin, n)), axis=1)
                goals = goals[:num_tasks]
            r = np.sqrt(np.square(goals[:, 0]) + np.square(goals[:, 1]))
            a = np.arctan2(goals[:, 1], goals[:, 0])  # TODO check whether I can just add pi here
        elif grid_mode == 'polar':
            n = int(np.ceil(np.sqrt(num_tasks)))
            a = np.repeat(np.linspace(0, 2 * np.pi, num=n), n)
            r = np.tile(np.linspace(0, self.task_max_radius + self.task_goal_offset, num=n), n)
            goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        elif grid_mode == 'fixed_goal': # Just one specific goal for testing purposes
            a = np.ones(num_tasks) * 0.6 * np.pi
            r = self.task_max_radius * 0.8 * (np.ones(num_tasks) ** 0.5)
            goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
            goals += np.sign(goals) * self.task_goal_offset
        else:
            a = np.random.randint(2, size=num_tasks) * np.pi if self.goal_1d else np.random.random(num_tasks) * 2 * np.pi
            # r = self.task_max_radius * (np.random.random(num_tasks) ** 0.5)
            r = self.task_max_radius * np.random.random(num_tasks)
            goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
            goals += np.sign(goals) * self.task_goal_offset
        if self.positive_environment:
            goals += self.task_goal_offset + self.task_max_radius # + 1
        if self.one_side_goals:
            goals = np.stack((np.abs(goals[:, 0]), goals[:, 1]), axis=1)
        goals += self.distribution_shift
        # if self.goal_1d:
        #     goals = goals[:, 0][:, None]

        tasks = [{'goal': goal, 'angle': angle, 'radius': rv, 'base_task': 0} for goal, angle, rv in zip(goals, a, r)]
        return tasks

    def set_meta_mode(self, mode):
        self.meta_mode = mode

    #########################Custom

    def reset(self):
        return self.reset_model()

    def render(self, mode='human'):
        print("Rendering not implemented")


if __name__ == "__main__":
    env = ToyGoalEnv(n_train_tasks=0, n_eval_tasks=200)
    positions = np.random.uniform(-1, 1, (50, 2))
    step_size = 0.1
    diff = 0
    for i in range(200):
        env.reset_task(i)
        r_x1 = env._get_reward_at(positions[:, 0] + step_size, positions[:, 1])
        r_x2 = env._get_reward_at(positions[:, 0], positions[:, 1] + step_size)

        r = env._get_reward_at(positions[:, 0], positions[:, 1])
        x1_goal = (((r_x1 ** 2) - 2 * r_x1 - (r ** 2) + 2 * r) * (-(env.max_distance ** 2)) / (2 * step_size)) + positions[:, 0] + (step_size / 2)
        x2_goal = (((r_x2 ** 2) - 2 * r_x2 - (r ** 2) + 2 * r) * (-(env.max_distance ** 2)) / (2 * step_size)) + positions[:, 1] + (step_size / 2)
        predicted_goals = np.stack((x1_goal, x2_goal), axis=1)
        diff_temp = np.abs(predicted_goals - env.goal['goal'][None, :]).sum(axis=1)
        print(diff_temp)
        if diff_temp.max() > diff:
            pos = positions[diff_temp.argmax()]
            pred_goal = predicted_goals[diff_temp.argmax()]
            goal = env.goal['goal']
            diff = diff_temp.max()
    print("diff: ", diff)
    print("pos: ", pos)
    print("goal: ", goal)
    r = env._get_reward_at(pos[0], pos[1])
    r_x1 = env._get_reward_at(pos[0] + step_size, pos[1])
    r_x2 = env._get_reward_at(pos[0], pos[1] + step_size)
    x1_goal = (((r_x1 ** 2) - 2 * r_x1 - (r ** 2) + 2 * r) * (-(env.max_distance**2)) / (2 * step_size)) + pos[0] + (step_size / 2)
    x2_goal = (((r_x2 ** 2) - 2 * r_x2 - (r ** 2) + 2 * r) * (-(env.max_distance**2)) / (2 * step_size)) + pos[1] + (step_size / 2)
    print("pred_goal: ", pred_goal)
    print("r_l, r_x1, rx_2: ", r, r_x1, r_x2)


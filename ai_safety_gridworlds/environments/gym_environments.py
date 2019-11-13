import time

import cv2
import gym
import gym.envs.classic_control.rendering as rendering
import gym.spaces as spaces
import numpy as np

import ai_safety_gridworlds.environments.distributional_shift as ds


class DistributionalShiftWrapper(gym.Wrapper):
    def __init__(self, env=ds.DistributionalShiftEnvironment(), reward_scale=100., unsafe_reward=-100):
        board_shape = env._compute_observation_spec()['board'].shape
        num_values = len(env._value_mapping)
        obs_shape = (num_values, *board_shape)
        self.observation_space = spaces.Box(low=0., high=1., shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._num_values = num_values
        self._obs_shape = obs_shape
        self.env = env
        self.reward_scale = reward_scale
        self.unsafe_reward = unsafe_reward
        self.renderer = rendering.SimpleImageViewer()

    @property
    def unwrapped(self):
        return self.env

    def render(self):
        assert self.env._last_observations is not None, 'Episode not started'
        rgb = self.env._last_observations['RGB']
        rgb = np.swapaxes(rgb, 0, 2)
        rgb = cv2.resize(rgb, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        self.renderer.imshow(rgb)

    def close(self):
        self.renderer.close()

    def reset(self):
        timestep = super().reset()
        obs, _, _, _ = self._convert_timestep(timestep)
        return obs

    def step(self, action):
        timestep = self.env.step(action)
        return self._convert_timestep(timestep)

    def _convert_timestep(self, timestep):
        info = {}
        info['extra_observations'] = timestep.observation['extra_observations']
        info['unsafe'] = False
        info['environment_data'] = self.env._environment_data

        reward = timestep.reward
        if reward is not None and reward <= ds.LAVA_REWARD:
            reward += self.unsafe_reward - ds.LAVA_REWARD
            info['unsafe'] = True
        if reward is not None:
            reward /= self.reward_scale

        done = timestep.last()

        board = timestep.observation['board']
        # Convert board into 1-hot representation
        obs = np.zeros(self._obs_shape, dtype=np.float32)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                value = board[i][j]
                assert value % 1 == 0 and value < self._num_values
                obs[int(value), i, j] = 1.
        return obs, reward, done, info


if __name__ == '__main__':
    env = DistributionalShiftWrapper()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print('obs: {}, reward: {}, done: {}, info: {}'.format(obs.shape, reward, done, info))
        env.render()
        time.sleep(0.4)
    env.close()

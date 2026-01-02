import numpy as np
import gym
from gym import spaces


class CropTreatmentEnv(gym.Env):
    def __init__(self, num_plants=5):
        super().__init__()
        self.num_plants = num_plants

        self.drone_pos = np.array([0.0, 0.0])
        self.plants = None
        self.treated = None

        obs_size = 2 + (num_plants * 3)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(num_plants)

    def reset(self):
        self.drone_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.plants = np.random.rand(self.num_plants, 3) * 50
        self.plants[:, 2] *= 10
        self.treated = np.zeros(self.num_plants)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.drone_pos, self.plants.flatten()]).astype(
            np.float32
        )

    def step(self, action):

        if self.treated[action] == 1:
            return self._get_obs(), -5.0, False, {}

        px, py, sev = self.plants[action]
        dist = np.linalg.norm(self.drone_pos - np.array([px, py]))

        severity_reward = sev * 2.0
        distance_penalty = dist * 0.3
        battery_penalty = dist * 0.05

        remaining_positions = self.plants[self.treated == 0][:, :2]
        if len(remaining_positions) > 1:
            dists = np.linalg.norm(remaining_positions - np.array([px, py]), axis=1)
            remaining_penalty = np.mean(dists) * 0.01
        else:
            remaining_penalty = 0.0

        reward = (
            severity_reward - distance_penalty - battery_penalty - remaining_penalty
        )

        self.drone_pos = np.array([px, py])

        self.treated[action] = 1
        self.plants[action][2] = 0

        done = int(self.treated.sum()) == self.num_plants

        return self._get_obs(), float(reward), done, {}

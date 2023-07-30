#! /usr/bin/env python
import gym
import numpy as np
from typing import Tuple, Any

class KArmedBanditsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, STEP_CAP, std_dev):
        # number of bandits
        # self.k = np.random.randint(low=2, high=5)
        self.k = 10
        self.means = np.random.uniform(low=0, high=10, size=self.k)
        # we have the same standard deviation, so it's easier to check if the learned policy is
        # what means correspond to.
        self.std_devs = std_dev * np.ones(self.k)
        self.STEP_CAP = STEP_CAP
        self.step_num = 0

    def step(self, action: int)->Tuple[object, float, bool, bool]: 
        """Run one timestep of the environment's dynamics. 

        Args:
            action (int): index of lever to pull

        Returns:
            observation (object): In this case, none.
            reward (float): reward from the slot machines
            terminated (bool): if the program has reached termination. Without reset, it program shouldn't run
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
        """
        self.step_num += 1
        terminated = self.step_num >= self.STEP_CAP
        if action >= self.k:
            raise AssertionError(f"index of lever '{action}' to pull should be smaller than k: {self.k}")
        mean, std_dev = self.means[action], self.std_devs[action]
        reward = np.random.normal(loc = mean, scale=std_dev)
        observation, trucacted = None, False
        return observation, reward, terminated, trucacted
    
    def reset(self): 
        # Our ctor already does the resetting for us.
        self.step_num = 0

    def render(self): 
        # nothing to render :)
        pass
    
    def get_k(self):
        return self.k


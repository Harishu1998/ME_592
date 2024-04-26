import gym
from gym import spaces
import pygame
import numpy as np
from prediction import Dynamics
import torch 
import torch.nn as nn


class GridWorldEnv(gym.Env):

    def terminate(self, ob):
        healthy = False
        healthy_state = ob[1:]
        height = ob[0]
        angle = ob[1]
        
        if np.all((self.healthy_state_range[0] > healthy_state) & (self.healthy_state_range[1]) < healthy_state):
            healthy = True
        
        if np.all((self.healthy_z_range[0] > height) & (self.healthy_z_range[1]) < height):
            healthy = True 
        
        if np.all((self.healthy_angle_range[0] > angle) & (self.healthy_angle_range[1]) < angle):
            healthy = True  

        return healthy

    def __init__(self, render_mode=None):

        self.observation_space = spaces.Box(low= float('inf'), high= float('-inf'), shape=(11,))
        self.action_space = spaces.Box(-1, 1, (3,))

        # Observation initialisation
        self.init_qpos = [0, 1.25, 0.0, 0.0]
        self.init_qvel = [0, 0, 0, 0, 0, 0, 0]
        self.init_x = 0

        # Other calculation initialisations
        self.forward_reward_weight=1.0
        self.ctrl_cost_weight=1e-3
        self.healthy_reward=1.0
        self.terminate_when_unhealthy=True
        self.healthy_state_range=(-100.0, 100.0)
        self.healthy_z_range=(0.7, float("inf"))
        self.healthy_angle_range=(-0.2, 0.2)
        self.reset_noise_scale=5e-3
        self.exclude_current_positions_from_observation=True
        self.timestep = 1

        self.model = torch.load('vanilla_nn_dynamics_hopper.pth')
    
    def reset(self, seed=None, options=None):
        self.qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=int(len(self.init_qpos)))
        self.qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=int(len(self.init_qvel)))
        #self.set_state(qpos, qvel)
        self.observation = np.concatenate([self.qpos, self.qvel])
        return self.observation, {}


    def step(self, action):
        with torch.no_grad():
            self.observation = np.insert(self.observation, 0, self.init_x)
            next_obs = self.model.predict(self.observation, action)

        final_x = next_obs[0]   
        self.observation = next_obs[1:]

        healthy_reward = self.healthy_reward
        forward_reward = self.forward_reward_weight * (final_x - self.init_x ) / 0.008 
        self.init_x = final_x
        control_cost = self.ctrl_cost_weight * sum(action**2)
        reward = healthy_reward + forward_reward - control_cost

        terminated = self.terminate(self.observation)
        
        if self.timestep > 1000:
            truncate = True
        else:
            truncate = False
        
        self.timestep += 1
        self.init_x = final_x

        return self.observation, reward, terminated, truncate, {}

    def render(self):
        pass

    def close(self):
        pass
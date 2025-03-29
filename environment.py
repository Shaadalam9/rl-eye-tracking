import gym
from gym import spaces
from tensordict import TensorDict, TensorDictBase
import torch
import torchrl
import numpy as np
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
import pickle
from custom_logger import CustomLogger
from logmod import logs

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class CustomRLEnv(gym.Env):
    def __init__(self):
        super(CustomRLEnv, self).__init__()

        # Define the action space (discrete actions in this case)
        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([1, 1]),
                                       dtype=np.float32)

        # Define the observation space (continuous states)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Initialize the environment state
        self.state = np.random.uniform(-1, 1, (4,))
        self.done = False
        self.steps = 0
        self.max_steps = 100

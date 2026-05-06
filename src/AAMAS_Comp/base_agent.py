from ns_gym.base import Agent
import gymnasium as gym
from typing import Dict
import numpy as np
import time

"""Make sure your agents inherit from these classes so that they adhere to the required interfaces. 

Don't change this code as we have our own versions for evaluation. 
"""

class ModelBasedAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, obs: Dict, planning_env: gym.Env, **kwargs):
        raise NotImplementedError("Please implement get_action method")

    def act(self, obs, planning_env, **kwargs):
        """Agent sub-class requries this method to be implemented
        """
        return self.get_action(obs, planning_env, **kwargs)

    def validate_and_get_action(self, obs: Dict, planning_env: gym.Env, reward=None, done=None):
        """Called by the competition evaluator."""
        start_time = time.perf_counter()
        action = self.get_action(obs, planning_env, reward=reward, done=done)
        end_time = time.perf_counter()
        if not isinstance(action, (np.ndarray, int, np.integer)):
            raise TypeError(f"Action must be a numpy array or int, got {type(action)}")

        if not planning_env.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the bounds of {planning_env.action_space}")
        

        decision_time = end_time - start_time
    
        return action, decision_time
    


class ModelFreeAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, obs: Dict, **kwargs):
        raise NotImplementedError("Please implement get_action method")

    def act(self, obs, **kwargs):
        return self.get_action(obs, **kwargs)

    def validate_and_get_action(self, obs: Dict, action_space: gym.Space, reward=None, done=None):
        """Called by the competition evaluator."""
        start_time = time.perf_counter()
        action = self.get_action(obs, reward=reward, done=done)
        end_time = time.perf_counter()

        assert action_space.contains(action), f"Invalid action: {action}"

        decision_time = end_time - start_time

        return action, decision_time
    
    


class SB3Agent(ModelFreeAgent):
    """Generic wrapper for any pre-trained Stable Baselines 3 model.

    Args:
        model: A trained SB3 model instance (e.g. PPO, A2C, DQN).
        deterministic (bool): Use deterministic actions. Defaults to True.
        vec_normalize: A VecNormalize instance for observation normalization.
            Required if the model was trained with VecNormalize.
    """

    def __init__(self, model, deterministic=True, vec_normalize=None) -> None:
        super().__init__()
        self.model = model
        self.deterministic = deterministic
        self.vec_normalize = vec_normalize

    def get_action(self, obs: Dict, **kwargs):
        state = obs["state"]
        if self.vec_normalize is not None:
            state = self.vec_normalize.normalize_obs(state)
        action, _ = self.model.predict(state, deterministic=self.deterministic)
        return action
    
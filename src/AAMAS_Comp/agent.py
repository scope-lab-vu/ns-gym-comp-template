
from typing import Dict
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent


"""Implement your code here. 
"""

class MyModelBasedAgent(ModelBasedAgent):

    def __init__(self):
        """YOUR CODE HERE
        """
        raise NotImplementedError


    def get_action(self, obs: Dict, planning_env, **kwargs):
        """YOUR CODE HERE

        Optional kwargs forwarded by the evaluator: `reward`, `done` (the
        result of the previous step). Useful for in-context RL agents that
        need to store transitions; ignore them otherwise.
        """
        raise NotImplementedError
    
    def set_seed(self, seed):
        raise NotImplementedError
    

class MyModelFreeAgent(ModelFreeAgent):

    def __init__(self):
        """YOUR CODE HERE
        """
        raise NotImplementedError

    def get_action(self, obs, **kwargs):
        """YOUR CODE HERE

        Optional kwargs forwarded by the evaluator: `reward`, `done` (the
        result of the previous step). Useful for in-context RL agents that
        need to store transitions; ignore them otherwise.
        """
        return None
    
    def set_seed(self, seed):
        raise NotImplementedError

    





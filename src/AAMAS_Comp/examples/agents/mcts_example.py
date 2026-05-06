from typing import Dict
from ns_gym.benchmark_algorithms import MCTS
from AAMAS_Comp.base_agent import ModelBasedAgent

import gymnasium as gym


"""This is an example evaluates Monte Carlo Tree Search (MCTS) with chance nodes on the non-stationary FrozenLake environment. Here we show how to wrap the existing NS-Gym MCTS implimentation in our AAMAS_Comp Agent classes for standardized interfaces. 

This MCTS implementation does `m` iterations. The we do a "random roll-out" once the tree reaches as leaf node till depth `d` or until reaching a terminal node. The action selection policy is the standard UCT equation. `c` is the UCT exploration constant. `gamma` is the MCTS cummulative reward discount factor applied to both the rollout policy and backpropogation steps of MCTS. 

"""

class AAMASCompBaselineMCTS(ModelBasedAgent):
    """AAMAS Competition Baseline MCTS

    An example model based agent following competition interfaces. 

    Args:
        d (int): Random roll out depth
        m (int): Number of MCTS Iterations
        c (float): UCT exploration constant. Defaults to 1.4.
        gamma (float): Cummulative reward discount factor. Defaults to 0.99
    """

    def __init__(self, d, m, c=1.4, gamma=0.99) -> None:
        super().__init__()

        self.d = d
        self.m = m
        self.c = c
        self.gamma = gamma 

    
    def get_action(self, obs: Dict, planning_env: gym.Env, **kwargs):

        state = obs["state"]

        mcts_solver = MCTS(env=planning_env,
                           state=state,
                           d=self.d,
                           m=self.m, 
                           c=self.c, 
                           gamma=self.gamma)
        
        action, _ = mcts_solver.search()

        return action
    







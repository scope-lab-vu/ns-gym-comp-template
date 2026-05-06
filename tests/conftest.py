import pytest
import gymnasium as gym
import AAMAS_Comp  # triggers environment registration
from AAMAS_Comp.base_agent import ModelBasedAgent, ModelFreeAgent


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (e.g. MuJoCo envs)")


# ---------------------------------------------------------------------------
# Dummy agents for testing interface conformance
# ---------------------------------------------------------------------------

class DummyModelBasedAgent(ModelBasedAgent):
    def __init__(self):
        super().__init__()

    def get_action(self, obs, planning_env, **kwargs):
        return planning_env.action_space.sample()


class DummyModelFreeAgent(ModelFreeAgent):
    def __init__(self, action_space):
        super().__init__()
        self._action_space = action_space

    def get_action(self, obs, **kwargs):
        return self._action_space.sample()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_model_based_agent():
    return DummyModelBasedAgent()


@pytest.fixture
def dummy_model_free_agent_factory():
    """Factory fixture — call with an action_space to get a DummyModelFreeAgent."""
    def _make(action_space):
        return DummyModelFreeAgent(action_space)
    return _make


@pytest.fixture
def ns_frozenlake_env():
    env = gym.make(
        "ExampleNSFrozenLake-v0",
        change_notification=True,
        delta_change_notification=True,
    )
    yield env
    env.close()


@pytest.fixture
def ns_cartpole_env():
    env = gym.make(
        "ExampleNSCartPole-v0",
        change_notification=True,
        delta_change_notification=True,
    )
    yield env
    env.close()

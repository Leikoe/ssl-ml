import warnings
from typing import Any, Union, Tuple, Dict, List

from gymnasium.vector.utils import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from abc import ABC


class PettingZooEnv(AECEnv, ABC):
    """The interface for petting zoo environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.PettingZooEnv`. Here is the usage:
    ::

        env = PettingZooEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, trunc, term, info = env.step(action)
        env.close()

    The available action's mask is set to True, otherwise it is set to False.
    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(self.env.observation_space(agent) == self.observation_space
                   for agent in self.agents), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(self.env.action_space(agent) == self.action_space
                   for agent in self.agents), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Union[dict, Tuple[dict, dict]]:
        self.env.reset(*args, **kwargs)

        # Here, we do not label the return values explicitly to keep compatibility with
        # old step API. TODO: Change once PettingZoo>=1.21.0 is required
        last_return = self.env.reset(self)
        print(last_return)

        if len(last_return) == 4:
            warnings.warn(
                "The PettingZoo environment is using the old step API. "
                "This API may not be supported in future versions of tianshou. "
                "We recommend that you update the environment code or apply a "
                "compatibility wrapper.", DeprecationWarning
            )
        try:
            observation, info = last_return[0], last_return[-1]
        except:
            observation, info = last_return, {}

        if isinstance(observation, dict) and 'action_mask' in observation:
            observation_dict = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                observation_dict = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                }

        if "return_info" in kwargs and kwargs["return_info"]:
            return observation_dict, info
        else:
            return observation_dict

    def step(
        self, action: Any
    ) -> Union[Tuple[Dict, List[int], bool, Dict], Tuple[Dict, List[int], bool, bool,
                                                         Dict]]:
        self.env.step(action)

        # Here, we do not label the return values explicitly to keep compatibility with
        # old step API. TODO: Change once PettingZoo>=1.21.0 is required
        last_return = self.env.last()
        observation = last_return[0]
        if isinstance(observation, dict) and 'action_mask' in observation:
            obs = {
                'agent_id': self.env.agent_selection,
                'obs': observation['observation'],
                'mask':
                [True if obm == 1 else False for obm in observation['action_mask']]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                obs = {
                    'agent_id': self.env.agent_selection,
                    'obs': observation,
                    'mask': [True] * self.env.action_space(self.env.agent_selection).n
                }
            else:
                obs = {'agent_id': self.env.agent_selection, 'obs': observation}

        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        return (obs, self.rewards, *last_return[2:])  # type: ignore

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()

import math

import gymnasium
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium.utils import EzPickle, seeding


from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .manual_policy import ManualPolicy  # noqa: F401

_image_library = {}

FPS = 20


def get_image(path):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    if env.continuous:
        env = wrappers.ClipOutOfBoundsWrapper(env)
    else:
        env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "ants",
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self,
                 n_ants=5,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True):

        super().__init__()
        self.dt = 1.0 / FPS
        self.n_ants = n_ants
        self.piston_head_height = 11
        self.piston_width = 40
        self.piston_height = 40
        self.piston_body_height = 23
        self.piston_radius = 5
        self.wall_width = 40
        self.ball_radius = 40
        self.screen_width = (2 * self.wall_width) + (self.piston_width * self.n_ants)
        self.screen_height = 560
        y_high = self.screen_height - self.wall_width - self.piston_body_height
        y_low = self.wall_width
        obs_height = y_high - y_low

        assert (
            self.piston_width == self.wall_width
        ), "Wall width and piston width must be equal for observation calculation"
        assert self.n_pistons > 1, "n_pistons must be greater than 1"

        self.agents = ["piston_" + str(r) for r in range(self.n_pistons)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        self._agent_selector = agent_selector(self.agents)

        obs_shape = 27
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
                    )
                ]
                * self.n_pistons,
            )
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        observation = pygame.surfarray.pixels3d(self.screen)
        i = self.agent_name_mapping[agent]
        # Set x bounds to include 40px left and 40px right of piston
        x_high = self.wall_width + self.piston_width * (i + 2)
        x_low = self.wall_width + self.piston_width * (i - 1)
        y_high = self.screen_height - self.wall_width - self.piston_body_height
        y_low = self.wall_width
        cropped = np.array(observation[x_low:x_high, y_low:y_high, :])
        observation = np.rot90(cropped, k=3)
        observation = np.fliplr(observation)
        return observation

    def state(self):
        """Returns an observation of the global environment."""
        pass

    def reset(self, seed=None, return_info=False, options=None):
        pass

    def render(self):
        pass

    def step(self, action):
        pass
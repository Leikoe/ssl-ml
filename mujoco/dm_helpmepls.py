from dm_control import suite
from dm_control import viewer
import numpy as np

from dm_control.locomotion import soccer as dm_soccer
from dm_env import TimeStep

env = dm_soccer.load(team_size=5,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)
action_specs = env.action_spec()


# Define a uniform random policy.
def random_policy(time_step: TimeStep):
    # del time_step  # Unused.
    # print(time_step.observation)
    return [np.random.uniform(low=action_spec.minimum,
                              high=action_spec.maximum,
                              size=action_spec.shape) for action_spec in action_specs]


# Launch the viewer application.
viewer.launch(env, policy=random_policy)

"""
Source: https://github.com/Farama-Foundation/PettingZoo/blob/master/README.md
"""

from pettingzoo.butterfly import pistonball_v6
env = pistonball_v6.env(render_mode="human")

env.reset()
for i in range(100):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
        env.step(action)
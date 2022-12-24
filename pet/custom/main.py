from pettingzoo.utils import agent_selector

import custom

env = custom.env(render_mode="human")

env.reset()
selector = agent_selector(env.agents)
agent = selector.reset()
# agent_selection will be "agent_1"
for i in range(3):
    agent = selector.next()
    print(f"agent: {agent}")
    observation, reward, termination, truncation, info = env.last()
    print(observation, reward, termination, truncation, info)
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)

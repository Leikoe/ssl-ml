import os

from custom import ssl_single_env
env = ssl_single_env.AntEnv(render_mode="human", xml_file=os.getcwd()+"/../robotics/onshape/ssl_bot.xml")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

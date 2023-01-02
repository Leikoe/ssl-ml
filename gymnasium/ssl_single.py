from custom import ssl_single_env
env = ssl_single_env.AntEnv(render_mode="human", xml_file="/Users/leo/project/ssl/ml/onshape/mjmodel.xml")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
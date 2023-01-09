from custom import ant_env
import gymnasium
# env = ant_env.AntEnv(render_mode="human", xml_file="/Users/leo/project/ssl/ml/onshape/mjmodel.xml")
env = gymnasium.make("ALE/Breakout-v5")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
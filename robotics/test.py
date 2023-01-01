import gymnasium as gym
env = gym.make("FetchPickAndPlace-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   # action = policy(observation)  # User-defined policy function
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
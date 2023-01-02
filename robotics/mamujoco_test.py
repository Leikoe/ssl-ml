import numpy
from gymnasium_robotics import mamujoco_v0

if __name__ == "__main__":
    env = mamujoco_v0.parallel_env(scenario='Ant', agent_conf='2x4', agent_obsk=0, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Humanoid', agent_conf='9|8', agent_obsk=0, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Reacher', agent_conf='2x1', agent_obsk=1, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='coupled_half_cheetah', agent_conf='1p1', agent_obsk=1, render_mode=None)
    # env = mamujoco_v0.parallel_env(scenario='Swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='manyagent_swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='coupled_half_cheetah', agent_conf='1p1', agent_obsk=0, render_mode='human')
    # env = mamujoco_v0.parallel_env(scenario='manyagent_swimmer', agent_conf='2x1', agent_obsk=0, render_mode='human')

    n_episodes = 1
    debug_step = 0

    for e in range(n_episodes):
        obs = env.reset()
        terminated = {'agent_0': False}
        truncated = {'agent_0': False}
        episode_reward = 0

        while not terminated['agent_0'] and not truncated['agent_0']:
            state = env.state()

            actions = {}
            for agent_id in env.agents:
                avail_actions = env.action_space(agent_id)
                action = numpy.random.uniform(avail_actions.low[0], avail_actions.high[0], avail_actions.shape[0])
                actions[str(agent_id)] = action

            obs, reward, terminated, truncated, info = env.step(actions)
            print(reward)
            episode_reward += reward['agent_0']

        print("Total reward in episode {} = {}".format(e, episode_reward))
    env.close()
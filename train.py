from ssl_env import new_env, step_env, Action, Observation, Env
import jax
import jax.numpy as jnp

N_ENVS = 100
EPISODES = 10

if __name__ == "__main__":
    for i in range(EPISODES):
        rngs = jax.random.split(jax.random.PRNGKey(30), N_ENVS)
        envs, observations = jax.vmap(lambda key: new_env(200, key))(rngs)

        jitted_vmapped_step = jax.jit(jax.vmap(step_env, in_axes=(0, None)))
        for i in range(1):
            action = Action(target_vel=jnp.array([1., 0.]), kick=False)
            envs, observations, rewards, terminateds, truncateds = jitted_vmapped_step(envs, action)
            print(rewards.min())

        exit(0)
        env, obs = new_env(200)

        done = False
        while not done:
            action = Action(target_vel=jnp.array([1., 0.]), kick=False)
            env, observation, reward, terminated, truncated = step_env(env, action)

            print(env.step)

            done = terminated or truncated

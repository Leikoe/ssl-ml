from ssl_env import new_env, step_env, Action, Observation, Env
import jax
import jax.numpy as jnp

N_ENVS = 100
EPISODES = 10

from flax import nnx

class ActorCritic(nnx.Module):
    def __init__(self, observation_size, action_size, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(observation_size, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, action_size, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        out = self.linear3(x)
        return out



if __name__ == "__main__":
    jitted_step = jax.jit(step_env)
    model = ActorCritic(2, 2, nnx.Rngs(0))
    jitted_policy = jax.jit(lambda obs: model(obs))

    for i in range(EPISODES):
        # rngs = jax.random.split(jax.random.PRNGKey(30), N_ENVS)
        # envs, observations = jax.vmap(lambda key: new_env(200, key))(rngs)

        # jitted_vmapped_step = jax.jit(jax.vmap(step_env, in_axes=(0, None)))
        # for i in range(1):
        #     action = Action(target_vel=jnp.array([1., 0.]), kick=False)
        #     envs, observations, rewards, terminateds, truncateds = jitted_vmapped_step(envs, action)
        #     print(rewards.min())

        # exit(0)
        env, obs = new_env(200, jax.random.PRNGKey(0))

        done = False
        while not done:
            action = jitted_policy(obs.pos)
            action = Action(target_vel=action, kick=False)
            env, obs, reward, terminated, truncated = jitted_step(env, action)

            print(reward)

            done = terminated or truncated

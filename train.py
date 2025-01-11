from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import distrax
import optax
import functools
from flax import nnx
from flax.nnx import initializers
from ssl_env import new_env, step_env, Action, Observation, Env


SEED = 42
N_ENVS = 1
NUM_STEPS = 100
EPISODES = 1


class ActorCritic(nnx.Module):
    def __init__(self, observation_size, action_size, rngs: nnx.Rngs):
        self.actor = nnx.Sequential(
            nnx.Linear(observation_size, 64, kernel_init=initializers.orthogonal(np.sqrt(2)),
                        bias_init=initializers.constant(0.0), rngs=rngs),
            jax.nn.relu,
            nnx.Linear(64, 64, kernel_init=initializers.orthogonal(np.sqrt(2)),
                        bias_init=initializers.constant(0.0), rngs=rngs),
            jax.nn.relu,
            nnx.Linear(64, action_size, kernel_init=initializers.orthogonal(0.01),
                        bias_init=initializers.constant(0.0), rngs=rngs)
        )
        self.log_std = nnx.Param(jnp.zeros((action_size,)))

        self.critic = nnx.Sequential(
            nnx.Linear(observation_size, 64, kernel_init=initializers.orthogonal(np.sqrt(2)),
                        bias_init=initializers.constant(0.0), rngs=rngs),
            jax.nn.relu,
            nnx.Linear(64, 64, kernel_init=initializers.orthogonal(np.sqrt(2)),
                        bias_init=initializers.constant(0.0), rngs=rngs),
            jax.nn.relu,
            nnx.Linear(64, 1, kernel_init=initializers.orthogonal(1.0),
                        bias_init=initializers.constant(0.0), rngs=rngs)
        )

    def __call__(self, x):
        # actor
        actor_mean = self.actor(x)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))

        # value
        return pi, self.critic(x)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


@functools.partial(jax.jit, static_argnums=0)
def rollout(model, n_steps, rng):
    rngs = jax.random.split(rng, N_ENVS)
    envs, obsv = jax.vmap(lambda key: new_env(key))(rngs)

    def _env_step(runner_state, unused):
        env_state, last_obs, rng = runner_state
        last_obs = last_obs.pos

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = model(last_obs)
        actions = pi.sample(seed=_rng)
        log_prob = pi.log_prob(actions)

        # STEP ENV
        actions_formatted = jax.vmap(lambda action: Action(target_vel=action, kick=False))(actions)
        env_state, obsv, reward, done = jax.vmap(step_env)(env_state, actions_formatted)
        jax.debug.print("rew={rew}", rew=reward)
        transition = Transition(
            done, actions, value, reward, log_prob, last_obs
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, transition

    runner_state = (envs, obsv, rng)
    return jax.lax.scan(
        _env_step, runner_state, None, NUM_STEPS
    )


if __name__ == "__main__":
    rng = jax.random.PRNGKey(SEED)
    model = ActorCritic(2, 2, nnx.Rngs(SEED))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.1))

    for i in range(EPISODES):
        runner_state, traj_batch = rollout(model, 200, rng)

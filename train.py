from ssl_env import new_env, step_env, Action, Observation, Env
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
import distrax

N_ENVS = 100
EPISODES = 10


class ActorCritic(nnx.Module):
    def __init__(self, observation_size, action_size, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(observation_size, 64, 
                                  kernel_init=initializers.orthogonal(np.sqrt(2)), 
                                  bias_init=initializers.constant(0.0), rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, 
                                  kernel_init=initializers.orthogonal(np.sqrt(2)), 
                                  bias_init=initializers.constant(0.0), rngs=rngs)
        self.linear3 = nnx.Linear(64, action_size, 
                                  kernel_init=initializers.orthogonal(0.01), 
                                  bias_init=initializers.constant(0.0), rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros((action_size,)))

    def __call__(self, x):
        actor_mean = self.linear1(x)
        actor_mean = nnx.relu(actor_mean)
        actor_mean = self.linear2(actor_mean)
        actor_mean = nnx.relu(actor_mean)
        actor_mean = self.linear3(actor_mean)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))


if __name__ == "__main__":
    jitted_step = jax.jit(step_env)
    model = ActorCritic(2, 2, nnx.Rngs(0))
    jitted_policy = model #jax.jit(lambda obs: model(obs))

    # def _env_step(runner_state, unused):
    #     """
    #     Steps the environment across ``NUM_ENVS``.
    #     Returns the updated runner state and transition tuple.
    #     """
    #     train_state, env_state, last_obs, rng = runner_state

    #     # action selection
    #     rng, _rng = jax.random.split(rng)
    #     # here obs has shape (obs, n_env) 
    #     # => pi contains n_env probability distributions
    #     pi, value = jitted_policy(last_obs)
    #     actions = pi.sample(seed=_rng)  # returns one action per env
    #     log_prob = pi.log_prob(actions)

    #     # step
    #     rng, _rng = jax.random.split(rng)
    #     rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    #     obsv, env_state, reward, done, info = jax.vmap(
    #         env.step, in_axes=(0, 0, 0, None)
    #     )(rng_step, env_state, actions, env_params)
        
    #     # collecting data
    #     transition = Transition(
    #         done, actions, value, reward, log_prob, last_obs, info
    #     )
    #     runner_state = (train_state, env_state, obsv, rng)
    #     return runner_state, transition

    # # scanning the _env_step function across NUM_STEPS iterations
    # runner_state, traj_batch = jax.lax.scan(
    #     _env_step, runner_state, None, config["NUM_STEPS"]
    # )

    # exit()

    for i in range(EPISODES):
        # rngs = jax.random.split(jax.random.PRNGKey(30), N_ENVS)
        # envs, observations = jax.vmap(lambda key: new_env(200, key))(rngs)

        # jitted_vmapped_step = jax.jit(jax.vmap(step_env, in_axes=(0, None)))
        # for i in range(1):
        #     action = Action(target_vel=jnp.array([1., 0.]), kick=False)
        #     envs, observations, rewards, terminateds, truncateds = jitted_vmapped_step(envs, action)
        #     print(rewards.min())

        # exit(0)
        rng = jax.random.PRNGKey(0)
        env, obs = new_env(200, rng)

        done = False
        while not done:
            obs = jnp.array([obs.pos]) # fake a batch
            action = jitted_policy(obs).sample(seed=rng)[0]
            print(obs, action)
            action = Action(target_vel=action, kick=False)
            
            env, obs, reward, terminated, truncated = jitted_step(env, action)
            print(env.step, reward)

            done = terminated or truncated

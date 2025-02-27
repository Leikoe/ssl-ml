# based on https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py
# and https://towardsdatascience.com/breaking-down-state-of-the-art-ppo-implementations-in-jax-6f102c06c149

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
LR = 2.5e-4
TOTAL_TIMESTEPS = 5_000_000
N = 2048  # number of envs
NUM_STEPS = 10
NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // N
K = 4  # number of epochs
M = 32  # number of minibatches
MINIBATCH_SIZE = N * NUM_STEPS // M
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01

# gae
GAMMA = 0.99
GAE_LAMBDA = 0.95


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
        actor_mean = self.actor(x)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std.value))
        return pi, jnp.squeeze(self.critic(x), axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


@functools.partial(jax.jit, static_argnums=0)
def rollout(model, rng):
    rngs = jax.random.split(rng, N)
    envs, obsv = jax.vmap(lambda key: new_env(key))(rngs)
    obsv = obsv.pos  # silly fix for now

    def _env_step(runner_state, unused):
        env_state, last_obs, rng = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = model(last_obs)
        actions = pi.sample(seed=_rng)
        log_prob = pi.log_prob(actions)

        # STEP ENV
        actions_formatted = jax.vmap(lambda action: Action(target_vel=action, kick=False))(actions)
        # print(actions_formatted)
        env_state, obsv, reward, done = jax.vmap(step_env)(env_state, actions_formatted)
        obsv = obsv.pos  # silly fix for now

        # jax.debug.print("rew={rew}", rew=reward)
        transition = Transition(
            done, actions, value, reward, log_prob, last_obs
        )
        runner_state = (env_state, obsv, rng)
        return runner_state, transition

    runner_state = (envs, obsv, rng)
    return jax.lax.scan(
        _env_step, runner_state, None, NUM_STEPS
    )

@jax.jit
def calculate_gae(traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        # print(gae.shape, next_value.shape)
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        # print(done.shape, value.shape, reward.shape)
        delta = reward + GAMMA * next_value * (1 - done) - value
        # print("delta:", delta.shape, reward.shape, next_value.shape, done.shape, value.shape)
        gae = delta + GAMMA * GAE_LAMBDA * (1 - done) * gae
        # print(gae.shape, value.shape)
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value

def loss_fn(model, traj_batch, gae, targets):
    # RERUN NETWORK
    pi, value = model(traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    # CALCULATE VALUE LOSS
    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-CLIP_EPS, CLIP_EPS)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()  # PureJaxRL's clipped MSE

    # CALCULATE ACTOR LOSS
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    unclipped_surrogate = ratio * gae
    clipped_surrogate = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * gae
    loss_actor = -jnp.minimum(unclipped_surrogate, clipped_surrogate) # -1*L because gradient ascent in ppo paper and we are doing gradient descent here
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + VF_COEF * value_loss - ENT_COEF * entropy
    return total_loss, (value_loss, loss_actor, entropy)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(SEED)
    model = ActorCritic(2, 2, nnx.Rngs(SEED))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LR, eps=1e-5))

    def _step(rng, x):
        # # COLLECT TRAJECTORIES
        runner_state, traj_batch = rollout(model, rng)
        jax.debug.print("mean last reward: {rew}", rew=traj_batch.reward[-1].mean())
        # print(jax.tree.map(lambda x: x.shape, traj_batch))

        # CALCULATE ADVANTAGE
        env_state, last_obs, rng = runner_state
        _, last_val = model(last_obs)

        advantages, targets = calculate_gae(traj_batch, last_val)

        def _epoch(update_state, _x):
            def _update_minbatch(_carry, batch_info):
                traj_batch, advantages, targets = batch_info

                grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
                (total_loss, (value_loss, loss_actor, entropy)), grad = grad_fn(model, traj_batch, advantages, targets)
                optimizer.update(grad)
                return _carry, total_loss

            traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = MINIBATCH_SIZE * M
            assert (
                batch_size == NUM_STEPS * N
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [M, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            _, total_loss = jax.lax.scan(
                _update_minbatch, None, minibatches
            )
            update_state = (traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _epoch, update_state, None, K

        )
        runner_state = (env_state, last_obs, rng)
        return rng, None

    train_fn = lambda rng: jax.lax.scan(
        _step,
        rng,
        length=NUM_UPDATES,
    )

    jitted_train_fn = jax.jit(train_fn)
    train_fn(rng)

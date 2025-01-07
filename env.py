import time
from typing import List, NamedTuple, Optional, Tuple
import numpy as np
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx


# mujoco consts
_MJ_MODEL = mujoco.MjModel.from_xml_path("scene.xml")
_MJX_MODEL = mjx.put_model(_MJ_MODEL)
ROBOT_QPOS_ADRS = jnp.array([_MJ_MODEL.joint("x_pos").qposadr[0], _MJ_MODEL.joint("y_pos").qposadr[0], _MJ_MODEL.joint("orientation").qposadr[0]]) # (x, y, orienation)
ROBOT_QVEL_ADRS = jnp.array([_MJ_MODEL.joint("x_pos").dofadr[0], _MJ_MODEL.joint("y_pos").dofadr[0], _MJ_MODEL.joint("orientation").dofadr[0]])  # (vx, vy, vangular)
ROBOT_CTRL_ADRS = jnp.array([_MJ_MODEL.actuator("forward_motor").id, _MJ_MODEL.actuator("left_motor").id, _MJ_MODEL.actuator("orientation_motor").id])  # (x, y, angular) motors
BALL_QPOS_ADRS = jnp.arange(3) + _MJ_MODEL.joint("free_ball").qposadr[0]  # (x, y, z) pos
BALL_QVEL_ADRS = jnp.arange(6) + _MJ_MODEL.joint("free_ball").dofadr[0]  # linear speed, rotational speed

# ctrl consts
A_MAX = 5.
K = 4.
M = _MJ_MODEL.body("robot_base").mass


def new_env() -> mjx.Data:
    mjx_data = mjx.make_data(_MJX_MODEL)
    # init code here ...
    return mjx_data.replace(qvel=mjx_data.qvel.at[BALL_QVEL_ADRS[0]].set(-1.)) # TODO: fix kick orientation

class Action(NamedTuple):
    target_vel: jax.Array
    kick: bool

@jax.jit
def step_env(env: mjx.Data, action: Action) -> mjx.Data:
    """
    Steps the env using the given `action`.

    Args:
        env (mjx.Data): The env state to step.
        action (jax.Array): The action taken. (target_vel_x, target_vel_y)

    Returns:
        mjx.Data: The new env state.
    """

    new_qvel = env.qvel

    # kick
    robot_pos = env.qpos[ROBOT_QPOS_ADRS][:2]
    ball_pos = env.qpos[BALL_QPOS_ADRS][:2]

    robot_to_ball = ball_pos-robot_pos
    robot_to_ball_angle = jnp.arctan2(robot_to_ball[1], robot_to_ball[0])
    robot_to_ball_distance = jnp.linalg.norm(robot_to_ball)
    robot_to_ball_normalized = robot_to_ball / robot_to_ball_distance

    REACH = 0.09 + 0.025# robot radius + ball radius
    kick_would_hit_ball = (robot_to_ball_distance < REACH) & ((robot_to_ball_angle < 0.2) & (robot_to_ball_angle > -0.2)) 
    new_qvel = jax.lax.cond(
        jnp.logical_and(action.kick, kick_would_hit_ball), # if we want to kick and the kick can hit the ball, apply vel
        lambda: new_qvel.at[BALL_QVEL_ADRS[:2]].set(robot_to_ball_normalized * 5.),
        lambda: new_qvel
    )
    env = env.replace(qvel=new_qvel)

    vel = env.qvel[ROBOT_QVEL_ADRS][:2]

    target_vel = action.target_vel # for now
    vel_err = target_vel - vel

    # clip speed
    a_target = K*vel_err
    a_target_norm = jnp.linalg.norm(a_target)
    a_target = jax.lax.cond(jnp.linalg.norm(a_target) > A_MAX, lambda: a_target * (A_MAX / a_target_norm), lambda: a_target)

    # compute force
    f = M * a_target

    env = env.replace(ctrl=env.ctrl.at[ROBOT_CTRL_ADRS[:2]].set(f))
    return mjx.step(_MJX_MODEL, env)

if __name__ == "__main__":
    env = new_env()
    mj_data = mjx.get_data(_MJ_MODEL, env)

    duration = 2.  # (seconds)
    framerate = 25  # (Hz)

    frames = []
    renderer = mujoco.Renderer(_MJ_MODEL, width=720, height=480)
    while env.time < duration:
        print(f"step {len(frames)}")
        step_start = time.time()

        robot_pos = env.qpos[ROBOT_QPOS_ADRS][:2]
        ball_pos = env.qpos[BALL_QPOS_ADRS][:2]

        robot_to_ball = ball_pos-robot_pos
        robot_to_ball_angle = jnp.arctan2(robot_to_ball[1], robot_to_ball[0])
        robot_to_ball_distance = np.linalg.norm(robot_to_ball)
        kick = bool(robot_to_ball_distance < (0.09 + 0.025))

        target_vel = jnp.array([0., 0.]) # placeholder for policy output
        action = Action(target_vel, kick)
        print(action)

        env = step_env(env, action)
        if len(frames) < env.time * framerate:
            mj_data = mjx.get_data(_MJ_MODEL, env)
            renderer.update_scene(mj_data)
            pixels = renderer.render()
            frames.append(pixels)
    renderer.close()
    
    from PIL import Image
    imgs = [Image.fromarray(img) for img in frames]
    # duration is the number of milliseconds between frames; this is 25 fps
    imgs[0].save("render.gif", save_all=True, append_images=imgs[1:], duration=40, loop=0)

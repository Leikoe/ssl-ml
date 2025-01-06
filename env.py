import time
from typing import List, NamedTuple, Optional, Tuple
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx


# mujoco consts
_MJ_MODEL = mujoco.MjModel.from_xml_path("scene.xml")
_MJX_MODEL = mjx.put_model(_MJ_MODEL)
ROBOT_BASE_ID = mjx.name2id(_MJ_MODEL, mjx.ObjType.BODY, 'robot_base')
ROBOT_QPOS_HINGE = jnp.array([_MJ_MODEL.joint("x_pos").qposadr[0], _MJ_MODEL.joint("y_pos").qposadr[0]])
ROBOT_QVEL_HINGE = jnp.array([_MJ_MODEL.joint("x_pos").dofadr[0], _MJ_MODEL.joint("y_pos").dofadr[0]])
ROBOT_CTRL_IDS = jnp.array([_MJ_MODEL.actuator("forward_motor").id, _MJ_MODEL.actuator("left_motor").id])
BALL_QVEL_IDS = jnp.arange(6) + _MJ_MODEL.joint("free_ball").dofadr[0]

# ctrl consts
A_MAX = 5.
K = 4.
M = _MJ_MODEL.body("robot_base").mass


def new_env() -> mjx.Data:
    mjx_data = mjx.make_data(_MJX_MODEL)
    # init code here ...
    qvel_ball_go_brr = mjx_data.qvel.at[BALL_QVEL_IDS[:3]].set(jnp.array([5., 0., 0.]))
    return mjx_data.replace(qvel=qvel_ball_go_brr)

@jax.jit
def step_env(env: mjx.Data, action: jax.Array) -> mjx.Data:
    """
    Steps the env using the given `action`.

    Args:
        env (mjx.Data): The env state to step.
        action (jax.Array): The action taken. (target_vel_x, target_vel_y)

    Returns:
        mjx.Data: The new env state.
    """
    vel = env.qvel[ROBOT_QVEL_HINGE]

    target_vel = action # for now
    vel_err = target_vel - vel

    # clip speed
    a_target = K*vel_err
    a_target_norm = jnp.linalg.norm(a_target)
    a_target = jax.lax.cond(jnp.linalg.norm(a_target) > A_MAX, lambda: a_target * (A_MAX / a_target_norm), lambda: a_target)

    # compute force
    f = M * a_target

    env = env.replace(ctrl=env.ctrl.at[ROBOT_CTRL_IDS].set(f))
    return mjx.step(_MJX_MODEL, env)

if __name__ == "__main__":
    env = new_env()
    mj_data = mjx.get_data(_MJ_MODEL, env)

    duration = 2.  # (seconds)
    framerate = 25  # (Hz)

    frames = []
    renderer = mujoco.Renderer(_MJ_MODEL, width=1920, height=1080)
    while env.time < duration:
        step_start = time.time()

        target_vel = jnp.array([5., 0.]) # placeholder for policy output
        env = step_env(env, target_vel)
        print(len(frames), env.qvel[ROBOT_QVEL_HINGE])
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

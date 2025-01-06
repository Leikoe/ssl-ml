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

# ctrl consts
A_MAX = 5.
K = 4.
M = _MJ_MODEL.body("robot_base").mass


def new_env() -> mjx.Data:
    mjx_data = mjx.make_data(_MJX_MODEL)
    # init code here ...
    return mjx_data

# @jax.jit
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

    frames = []
    renderer = mujoco.Renderer(_MJ_MODEL, height=360, width=480)
    for i in range(100):
        step_start = time.time()
        env = step_env(env, jnp.array([5., 0.]))
        vel = env.qvel[ROBOT_QVEL_HINGE]
        print(vel)
        mj_data = mjx.get_data(_MJ_MODEL, env)

        renderer.update_scene(mj_data)
        pixels = renderer.render()
        frames.append(pixels)
        
    import numpy as np
    from PIL import Image

    imgs = [Image.fromarray(img) for img in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=len(frames) * (1/60.), loop=0)

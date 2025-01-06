import mujoco
import numpy as np
import mujoco.viewer
import time

A_MAX = 5.
TARGET_VEL = np.array([1.0, 0.0])
K = 4.


model = mujoco.MjModel.from_xml_path("scene.xml")
M = model.body("robot_base").mass
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
while data.time < 10:
    step_start = time.time()
    mujoco.mj_step(model, data)

    vel = np.array([data.joint("x_pos").qvel[0], data.joint("y_pos").qvel[0]])
    print(vel)
    vel_err = TARGET_VEL - vel

    # clip speed
    a_target = K*vel_err
    a_target_norm = np.linalg.norm(a_target)
    if np.linalg.norm(a_target) > A_MAX:
       a_target *= (A_MAX / a_target_norm)

    f = M * a_target
    # print("force applied:", f)
    data.actuator("forward_motor").ctrl = f[0]
    data.actuator("left_motor").ctrl = f[1]
    # print("pos:", model.body("robot_base").pos)
    viewer.sync()
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
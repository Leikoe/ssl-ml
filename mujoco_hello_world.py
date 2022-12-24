import mujoco

XML=r"""
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""


model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)

"""
Source: https://github.com/deepmind/dm_control/tree/main/dm_control/mjcf
"""

from dm_control import mjcf

arena = mjcf.RootElement(model="ssl_env")
arena.worldbody.add('geom', name='ground', type='plane', size=[10, 10, 1])

robot = mjcf.from_path('../robotics/onshape/mjmodel.xml')
robot.set_attributes(model=f"robot")


print(robot)
arena.attach(robot)

with open("ssl_bot.xml", "w") as f:
    f.write(arena.to_xml_string())

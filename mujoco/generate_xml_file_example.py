"""
Source: https://github.com/deepmind/dm_control/tree/main/dm_control/mjcf
"""

from dm_control import mjcf


class Arm(object):

    def __init__(self, name):
        self.mjcf_model = mjcf.RootElement(model=name)

        self.upper_arm = self.mjcf_model.worldbody.add('body', name='upper_arm')
        self.shoulder = self.upper_arm.add('joint', name='shoulder', type='ball')
        self.upper_arm.add('geom', name='upper_arm', type='capsule',
                           pos=[0, 0, -0.15], size=[0.045, 0.15])

        self.forearm = self.upper_arm.add('body', name='forearm', pos=[0, 0, -0.3])
        self.elbow = self.forearm.add('joint', name='elbow',
                                      type='hinge', axis=[0, 1, 0])
        self.forearm.add('geom', name='forearm', type='capsule',
                         pos=[0, 0, -0.15], size=[0.045, 0.15])


class UpperBody(object):

    def __init__(self):
        self.mjcf_model = mjcf.RootElement()
        self.mjcf_model.worldbody.add(
            'geom', name='torso', type='box', size=[0.15, 0.045, 0.25])
        left_shoulder_site = self.mjcf_model.worldbody.add(
            'site', size=[1e-6] * 3, pos=[-0.15, 0, 0.25])
        right_shoulder_site = self.mjcf_model.worldbody.add(
            'site', size=[1e-6] * 3, pos=[0.15, 0, 0.25])

        self.left_arm = Arm(name='left_arm')
        left_shoulder_site.attach(self.left_arm.mjcf_model)

        self.right_arm = Arm(name='right_arm')
        right_shoulder_site.attach(self.right_arm.mjcf_model)


body = UpperBody()
physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)

print(body.mjcf_model.to_xml_string())
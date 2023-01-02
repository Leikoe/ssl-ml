# Requirements
> note: this is tested on m1 pro macbook macOS 13.1, with python 3.9 venv
- numpy (1.24.0)
- mujoco (2.3.1.post1)
- gymnasium (0.27.0)
- gymnasium robotics (mamujoco PR)
```shell
git clone https://github.com/Kallinteris-Andreas/Gymnasium-Robotics-Kalli
cd Gymnasium-Robotics-Kalli
pip install .
```
- box2d-py (?)
- dm_control (generating xml mujoco files)

# Quick start

```shell
python gymnasium_hello_world.py

python mujoco_hello_world.py
```

# MacOS specific install
- https://developer.apple.com/metal/pytorch/
- https://developer.apple.com/metal/tensorflow-plugin/

# Exact pip list
```
absl-py            1.3.0
Box2D              2.3.2
cloudpickle        2.2.0
contourpy          1.0.6
cycler             0.11.0
fonttools          4.38.0
glfw               2.5.5
gym                0.26.2
gym-notices        0.0.8
gymnasium          0.27.0
gymnasium-notices  0.0.1
imageio            2.23.0
importlib-metadata 5.2.0
jax-jumpy          0.2.0
kiwisolver         1.4.4
matplotlib         3.6.2
mujoco             2.3.1.post1
numpy              1.24.0
packaging          22.0
pandas             1.5.2
Pillow             9.3.0
pip                21.3.1
pygame             2.1.3.dev8
PyOpenGL           3.1.6
pyparsing          3.0.9
python-dateutil    2.8.2
pytz               2022.7
seaborn            0.12.1
setuptools         60.2.0
Shimmy             0.2.0
six                1.16.0
swig               4.1.1
torch              1.13.1
typing_extensions  4.4.0
wheel              0.37.1
zipp               3.11.0
```
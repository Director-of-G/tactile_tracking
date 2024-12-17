import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='tactile_model',
    version='1.0.0',
    packages=find_packages(),
    description='scripts and utilities for collecting, learning tactile models in Mujoco',
    author='Yongpeng Jiang, IRM Lab, Tsinghua University',
    install_requires=[
        'gym==0.25.2',
        'mujoco==3.2.3',
        'joblib',
        'torch',
        'scipy',
        'pytorch_kinematics',
        'pinocchio',
        'roma'
    ],
)
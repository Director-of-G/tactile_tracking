import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as SciR

import mujoco
import mujoco.viewer

from matplotlib import pyplot as plt

xml_path = './Adroit/my_sensor_panel_v2.xml'
dataset_path = './data/dataset_v1'

FLAG_APPYCONTROL = False
FLAG_COLLECTTOUCHGRID = False
FLAG_COLLECTSENSORARRAY = True
FLAG_SAVEFIGURE = False
FLAG_SAVEDATA = False

TIME_STEP = 0.002

int_euler_seq = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"] + ["XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]
ext_euler_seq = [seq.lower() for seq in int_euler_seq]
euler_seq = int_euler_seq + ext_euler_seq

m = mujoco.MjModel.from_xml_path(xml_path)

# Shadow Hand joint poses
# FFJ3， FFJ2， FFJ1， FFJ0 = 0.0, 0.4, 1.6, 0.288
shadow_hand_dof_targets = np.array([
  0, 0.183,                     # 0~1
  0.0, 0.4, 1.6, 0.288,         # 2~5
  0.0, 0.4, 1.6, 0.288,         # 6~9
  0.0, 0.4, 1.6, 0.288,         # 10~13
  0.0, 0.0, 0.4, 1.6, 0.288,    # 14~18
  1.0, 1.17, 0.0, 0.0, 0.0      # 19~23
])
shadow_hand_dof_targets[3:6] += 0.05
shadow_hand_dof_targets[7:10] += 0.05
shadow_hand_dof_targets[11:14] += 0.05
shadow_hand_dof_targets[16:19] += 0.05

# Shadow Hand Touch Grids
shadow_hand_touch_grid_names = [
    'thdistal_touch', 'thmiddle_touch', 'thproximal_touch', \
    'ffdistal_touch', 'ffmiddle_touch', 'ffproximal_touch', \
    'mfdistal_touch', 'mfmiddle_touch', 'mfproximal_touch', \
    'rfdistal_touch', 'rfmiddle_touch', 'rfproximal_touch', \
    'lfdistal_touch', 'lfmiddle_touch', 'lfproximal_touch'
]

# Tactile Sensor Array config
finger_keys = ["ff", "mf", "rf", "lf", "th"]
knuckle_keys = ["distal", "middle", "proximal"]
taxel_size = (4, 4)
global_idx = 0

# buffer for data collection
taxel_reading_buf = []
panel_state_buf = []
panel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panel_base")

def read_sensor_array(data:mujoco.MjData):
  raw = np.zeros(taxel_size)
  taxel_width, taxel_height = taxel_size
  for tx in range(taxel_width):
    for ty in range(taxel_height):
        raw[tx, ty] = data.sensor(f'panel_S_r{tx}c{ty}').data
  return raw

def generate_ctrl_target_se3(t):
    """
      t: time param. for warm start
    """
    x_range = [0.0, 0.0]
    y_range = [0.0, 0.0]
    
    # z_range = [-0.05, -0.026]
    z_range = [-0.35, -0.026]
    # overide z, needs warm start, since sudden decrease of z will cause severe penetration
    z_range[0] = -0.026 - min((t / 10.0), 1.0) * (-0.026-(-0.35))
    
    rx_range = [-0.2, 0.2]
    ry_range = [-0.2, 0.2]
    rz_range = [0.0, 0.0]
    se3_range = np.array([x_range, y_range, z_range, rx_range, ry_range, rz_range])
    sample = np.random.uniform(se3_range[:, 0], se3_range[:, 1])

    return sample

d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 3000:
    step_start = time.time()

    # set desired sensor base se3
    if ((time.time() - start) % 2.0) < TIME_STEP:
      if FLAG_SAVEFIGURE:
        plt.figure()
        raw = read_sensor_array(d)
        print(f"raw value min - max = {raw.min()} - {raw.max()}")
        plt.imshow(raw*10, cmap='gray', vmin=0, vmax=1, norm=None)
        plt.grid("off")
        plt.axis("off")
        plt.savefig(f'./debug/sample_{str(global_idx).zfill(3)}.png', dpi=300)
        global_idx += 1
      if 'sensor_panel' in xml_path:
        passed_euler_seq = []
        for seq in euler_seq:
          _euler_as_qpos = d.qpos[3:6]
          _quat = d.xquat[1]
          _converted_quat = SciR.from_euler(seq, _euler_as_qpos).as_quat()[[-1, 0, 1, 2]]
          if np.allclose(_converted_quat, _quat):
            passed_euler_seq.append(seq)
        print("these euler seq match quat: ", passed_euler_seq)
        # try:
        #   assert(np.allclose(_converted_quat, _quat))
        #   print("euler and quat match!")
        # except:
        #   print("euler and quat mismatch")
        #   print("converted euler from qpos: ", _converted_quat)
        #   print("quat: ", _quat)

      # read panel state
      panel_pos = d.xpos[panel_id]
      panel_rot = d.xmat[panel_id].reshape(3, 3)
      panel_rot = SciR.from_matrix(panel_rot).as_quat()
      # print(f"panel pos: {panel_pos}, panel rot: {panel_rot}")

      # read sensor array
      panel_data = read_sensor_array(d)
      # print(f"panel data min - max = {panel_data.min()} - {panel_data.max()}")

      # save to buffer
      taxel_reading_buf.append(panel_data.flatten())
      panel_state_buf.append(np.concatenate((panel_pos, panel_rot)))

      # autogen panel pose
      # if time.time() - start < TIME_STEP:
      #   # this warm start does not work, now implements in generate_ctrl_target_se3
      #   print("ctrl initialize to home position")
      #   d.ctrl[:] = np.zeros(6,)
      # else:
      d.ctrl[:] = generate_ctrl_target_se3(time.time()-start)

      # print("all dof pos: ", d.qpos)

      if len(taxel_reading_buf) >= 10 and FLAG_SAVEDATA:
        npz_idx = len(os.listdir(dataset_path))
        np.savez(os.path.join(dataset_path, str(npz_idx).zfill(3)), taxel_reading=np.array(taxel_reading_buf), panel_state=np.array(panel_state_buf))
        taxel_reading_buf.clear()
        panel_state_buf.clear()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

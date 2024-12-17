import time
import numpy as np
from scipy.spatial.transform import Rotation as SciR

import mujoco
import mujoco.viewer

from matplotlib import pyplot as plt

from tactile_model_learning.tactile_model.hand_model import HandModel


# xml_path = './Adroit/Adroit_hand.xml'
# xml_path = './Adroit/resources/touch_grid.xml'
# xml_path = './Adroit/my_sensor_panel_v2.xml'
# xml_path = "/home/yongpeng/research/dexterous/mj_envs/mj_envs/hand_manipulation_suite/assets/DAPG_relocate_v1.xml"
xml_path = "./Leap/Leap_hand.xml"

TIME_STEP = 0.002

FLAG_APPYCONTROL = False
FLAG_COLLECTTOUCHGRID = False
FLAG_COLLECTSENSORARRAY = True

def get_actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
m = mujoco.MjModel.from_xml_path(xml_path)
m.actuator_gainprm[get_actuator_id(m, "A_WRJ1"):get_actuator_id(m, "A_WRJ0")+1,:3] = np.array([10, 0, 0])
m.actuator_gainprm[get_actuator_id(m, "A_FFJ3"):get_actuator_id(m, 'A_THJ0')+1,:3] = np.array([1, 0, 0])
m.actuator_biasprm[get_actuator_id(m, 'A_WRJ1'):get_actuator_id(m, 'A_WRJ0')+1,:3] = np.array([0, -10, 0])
m.actuator_biasprm[get_actuator_id(m, 'A_FFJ3'):get_actuator_id(m, 'A_THJ0')+1,:3] = np.array([0, -1, 0])

m.actuator_ctrlrange[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_ARTy")] = np.array([-0.2, 0.2])

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

def read_sensor_array(data:mujoco.MjData, knuckle_key):
  raw = np.zeros(taxel_size)
  taxel_width, taxel_height = taxel_size
  try:
    for tx in range(taxel_width):
      for ty in range(taxel_height):
          raw[tx, ty] = data.sensor(f'{knuckle_key}_S_r{tx}c{ty}').data
  except:
    print(f"sensor {knuckle_key} not found")
  return raw

def check_torch_fk(hand:HandModel, mj_m:mujoco.MjModel, mj_d:mujoco.MjData):
  # find all actuated joints
  actuator_joint_angles = {}
  for i in range(mj_m.nu):
    joint_id = mj_m.actuator_trnid[i][0]  # 获取 actuator 驱动的关节 ID
    joint_name = mujoco.mj_id2name(mj_m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    qpos_address = mj_m.jnt_qposadr[joint_id]  # 获取该关节在 qpos 中的索引
    actuator_joint_angles[joint_name] = mj_d.qpos[qpos_address]  # 存储关节角度
  test_fk = hand.chain.forward_kinematics(actuator_joint_angles)
  print("joint pos: ", actuator_joint_angles)
  for key in hand.panel_base_keys:
    body_address = mujoco.mj_name2id(mj_m, mujoco.mjtObj.mjOBJ_BODY, key)
    
    pos_match = np.allclose(mj_d.xpos[body_address], test_fk[key].get_matrix()[0, :3, 3])
    quat_match = np.allclose(mj_d.xmat[body_address].reshape(3, 3), test_fk[key].get_matrix()[0, :3, :3])

    if not (pos_match and quat_match):
      print(f"fk mismatch for {key}, body_address is {body_address}")
      print(f"mujoco pos: {mj_d.xpos[body_address]}, fk pos: {test_fk[key].get_matrix()[0, :3, 3]}")
      print(f"mujoco quat: {mj_d.xmat[body_address].reshape(3, 3)}, fk quat: {test_fk[key].get_matrix()[0, :3, :3]}")

d = mujoco.MjData(m)

hand_model = HandModel(model_path="./Adroit/Adroit_hand_kin_v2.xml")

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 3000:
    step_start = time.time()

    # if time.time() - start > 2:
    #    import pdb; pdb.set_trace()
    #    check_torch_fk(hand_model, m, d)
    #    exit(0)

    if ((time.time() - start) % 2.0) < TIME_STEP:
        panel_data = read_sensor_array(d, "ffdistal")
        print(f"panel data min - max = {panel_data.min()} - {panel_data.max()}")
        # check how qpos is related to euler
        if 'sensor_panel' in xml_path:
          _euler_as_qpos = d.qpos[3:6]
          _quat = d.xquat[1]
          try:
            assert(np.allclose(SciR.from_euler("XYZ", _euler_as_qpos).as_quat()[[-1, 0, 1, 2]], _quat))
          except:
            print("euler and quat mismatch")
            print("converted euler from qpos: ", SciR.from_euler("XYZ", _euler_as_qpos).as_quat()[[-1, 0, 1, 2]])
            print("quat: ", _quat)

    if time.time() - start > 200:
        if 'hand' in xml_path:
            if FLAG_COLLECTTOUCHGRID:
              plt.figure()
              for i, name in enumerate(shadow_hand_touch_grid_names):
                  # import pdb; pdb.set_trace()
                  raw = np.asarray(d.sensor(name).data)
                  size = int(np.sqrt(len(raw) // 3))
                  channel = raw[:size**2].reshape(size, size)
                  plt.subplot(4, 4, i+1)
                  plt.imshow(channel, cmap='gray')
                  plt.grid("off")
              plt.savefig('./touch_grid_z.png', dpi=300)
            if FLAG_COLLECTSENSORARRAY:
              plt.figure()
              for i, finger in enumerate(finger_keys):
                  for j, knuckle in enumerate(knuckle_keys):
                      name = f'{finger}{knuckle}'
                      raw = read_sensor_array(d, name)
                      # channel = np.linalg.norm(raw, axis=-1)
                      channel = raw
                      print(f"{name}: {channel.min()} - {channel.max()}")
                      # normalize
                      # channel = (channel - channel.min()) / (channel.max() - channel.min())
                      plt.subplot(5, 3, i*3+j+1)
                      plt.imshow(channel, cmap='gray', vmin=0, vmax=1, norm=None)
                      plt.grid("off")
              plt.savefig('./sensor_array_z.png', dpi=300)

        exit(0)

    # set desired joint pos
    if 'hand' in xml_path:
      if FLAG_APPYCONTROL:
        d.ctrl[:] = shadow_hand_dof_targets

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
